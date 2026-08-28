import { buildRoute, type Waypoint } from './domain';
import { analyzePhotoPosition } from './photo-analysis';
import { evaluatePhotoCompliance } from './photo-compliance';
import { preparePhotoJpeg } from './photo-image';
import {
  applyManualValue,
  extractPhotoMetadata,
  inferClassification,
  markRepeatedGps,
} from './photo-metadata';
import {
  type ExamplePhotoManifest,
  type ExamplePhotoManifestItem,
  missingValue,
  PHOTO_CLASSIFICATIONS,
  type PhotoClassification,
  type PhotoComplianceSummary,
  type PhotoMetadata,
  type PhotoRecord,
} from './photo-types';

export interface PhotoLayerOptions {
  exactDots: boolean;
  projectedMarkers: boolean;
  headingArrows: boolean;
  connectors: boolean;
  legend: boolean;
  includeInCrop: boolean;
}

export interface PhotoHandoutOptions {
  splitWaypoint: string;
  splitAfterM: number | null;
  includeSummary: boolean;
}

interface ImportOverride {
  item?: ExamplePhotoManifestItem;
}

const CLASS_LABELS: Record<PhotoClassification, string> = {
  enroute: 'En-route photo',
  'control-correct': 'Correct control photo',
  'control-false': 'False control photo',
  'sign-task': 'Sign task',
  reference: 'Reference only',
};

export const PHOTO_IMPORT_LIMITS = {
  maximumCount: 60,
  maximumFileBytes: 25 * 1024 * 1024,
  maximumTotalBytes: 250 * 1024 * 1024,
  maximumMegapixels: 50,
  thumbnailEdge: 480,
} as const;

function required<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Required photo workflow element is missing: #${id}`);
  return element as T;
}

export function validateManualNumber(
  field: string,
  value: string
): { value: number | null; error: string | null } {
  if (!value.trim()) return { value: null, error: null };
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return { value: null, error: 'Enter a finite number.' };
  const ranges: Record<string, [number, number, boolean]> = {
    latitude: [-90, 90, true],
    taskLatitude: [-90, 90, true],
    longitude: [-180, 180, true],
    taskLongitude: [-180, 180, true],
    headingDeg: [0, 360, false],
  };
  const range = ranges[field];
  if (range && (parsed < range[0] || (range[2] ? parsed > range[1] : parsed >= range[1]))) {
    return {
      value: null,
      error: `${field.toLowerCase().includes('latitude') ? 'Latitude' : field.toLowerCase().includes('longitude') ? 'Longitude' : 'Heading'} must be ${range[2] ? `between ${range[0]} and ${range[1]}` : `at least ${range[0]} and below ${range[1]}`}.`,
    };
  }
  if (['altitudeAglFt', 'focalLength35Mm'].includes(field) && parsed <= 0) {
    return { value: null, error: 'Enter a value greater than zero.' };
  }
  return { value: parsed, error: null };
}

export function alphabeticIdentifier(index: number): string {
  if (!Number.isInteger(index) || index < 0)
    throw new Error('Identifier index must be a non-negative integer.');
  let value = index + 1;
  let result = '';
  while (value > 0) {
    value -= 1;
    result = String.fromCharCode(65 + (value % 26)) + result;
    value = Math.floor(value / 26);
  }
  return result;
}

function emptyMetadata(note: string): PhotoMetadata {
  return {
    latitude: missingValue(note),
    longitude: missingValue(note),
    gpsAltitudeMslM: missingValue(note),
    altitudeAglFt: missingValue(note),
    headingDeg: missingValue(note),
    headingReference: missingValue(note),
    focalLengthMm: missingValue(note),
    focalLength35Mm: missingValue(note),
    captureTime: missingValue(note),
    orientation: missingValue(note),
    cameraMake: missingValue(note),
    cameraModel: missingValue(note),
    lensModel: missingValue(note),
  };
}

export async function hashArrayBufferSha256(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

export async function hashFileSha256(file: File): Promise<string> {
  return hashArrayBufferSha256(await file.arrayBuffer());
}

function randomId(): string {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  return Array.from(bytes, (byte, index) => {
    const separator = [4, 6, 8, 10].includes(index) ? '-' : '';
    return `${separator}${byte.toString(16).padStart(2, '0')}`;
  }).join('');
}

function cloneMetadata(metadata: PhotoMetadata): PhotoMetadata {
  return JSON.parse(JSON.stringify(metadata)) as PhotoMetadata;
}

export function isDuplicatePhoto(file: File, hash: string, records: PhotoRecord[]): boolean {
  return records.some(
    (record) =>
      record.contentHash === hash || (record.fileName === file.name && record.fileSize === file.size)
  );
}

export function applyPhotoClassification(
  record: PhotoRecord,
  classification: PhotoClassification,
  linkedWaypoint: string | null
): PhotoRecord {
  return { ...record, classification, linkedWaypoint };
}

function imageDimensions(url: string): Promise<[number, number]> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve([image.naturalWidth, image.naturalHeight]);
    image.onerror = () => reject(new Error('The JPEG could not be decoded.'));
    image.src = url;
  });
}

function input(
  label: string,
  value: string,
  field: string,
  id: string,
  type = 'text',
  error: string | null = null
): HTMLElement {
  const wrapper = document.createElement('label');
  wrapper.className = 'photo-field';
  wrapper.append(document.createTextNode(label));
  const control = document.createElement('input');
  control.type = type;
  control.value = value;
  control.dataset.photoId = id;
  control.dataset.field = field;
  if (type === 'number') control.step = 'any';
  if (error) {
    control.setAttribute('aria-invalid', 'true');
    control.setCustomValidity(error);
    const message = document.createElement('small');
    message.className = 'danger-text';
    message.textContent = error;
    wrapper.append(control, message);
    return wrapper;
  }
  wrapper.appendChild(control);
  return wrapper;
}

function select(
  label: string,
  value: string,
  field: string,
  id: string,
  values: Array<[string, string]>
): HTMLElement {
  const wrapper = document.createElement('label');
  wrapper.className = 'photo-field';
  wrapper.append(document.createTextNode(label));
  const control = document.createElement('select');
  control.dataset.photoId = id;
  control.dataset.field = field;
  for (const [optionValue, optionLabel] of values) {
    const option = document.createElement('option');
    option.value = optionValue;
    option.textContent = optionLabel;
    option.selected = optionValue === value;
    control.appendChild(option);
  }
  wrapper.appendChild(control);
  return wrapper;
}

export class PhotoWorkflow {
  records: PhotoRecord[] = [];
  compliance: PhotoComplianceSummary = evaluatePhotoCompliance([], []);
  private route: Waypoint[] = [];
  private readonly list = required<HTMLElement>('photoList');
  private readonly empty = required<HTMLElement>('photoEmpty');
  private readonly progress = required<HTMLProgressElement>('photoProgress');
  private readonly progressText = required<HTMLElement>('photoProgressText');
  private readonly input = required<HTMLInputElement>('photoFiles');
  private readonly drop = required<HTMLElement>('photoDropzone');
  private readonly status = required<HTMLElement>('photoComplianceStatus');
  private readonly findings = required<HTMLElement>('photoFindings');
  private readonly loadExampleButton = required<HTMLButtonElement>('loadPhotoExample');
  private importing = false;
  private externallyBusy = false;

  get isBusy(): boolean {
    return this.importing || this.externallyBusy;
  }

  setExternalBusy(busy: boolean): void {
    this.externallyBusy = busy;
    this.input.disabled = busy || this.importing;
    this.loadExampleButton.disabled = busy || this.importing;
    this.drop.setAttribute('aria-disabled', String(busy));
    if (busy) {
      this.list
        .querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>('input,select,button')
        .forEach((control) => {
          control.disabled = true;
        });
    } else {
      this.render();
    }
  }

  constructor(
    private readonly onMessage: (message: string, tone?: 'neutral' | 'success' | 'warning' | 'error') => void
  ) {
    this.input.addEventListener('change', () => void this.importFiles(Array.from(this.input.files ?? [])));
    this.drop.addEventListener('dragover', (event) => {
      event.preventDefault();
      this.drop.classList.add('is-dragging');
    });
    this.drop.addEventListener('dragleave', () => this.drop.classList.remove('is-dragging'));
    this.drop.addEventListener('drop', (event) => {
      event.preventDefault();
      this.drop.classList.remove('is-dragging');
      void this.importFiles(Array.from(event.dataTransfer?.files ?? []));
    });
    this.list.addEventListener('click', (event) => this.handleClick(event));
    this.list.addEventListener('change', (event) => this.handleChange(event));
    this.loadExampleButton.addEventListener('click', () => void this.loadExample());
    this.render();
  }

  get layerOptions(): PhotoLayerOptions {
    return {
      exactDots: required<HTMLInputElement>('photoExactDots').checked,
      projectedMarkers: required<HTMLInputElement>('photoProjectedMarkers').checked,
      headingArrows: required<HTMLInputElement>('photoHeadingArrows').checked,
      connectors: required<HTMLInputElement>('photoConnectors').checked,
      legend: required<HTMLInputElement>('photoLegend').checked,
      includeInCrop: required<HTMLInputElement>('photoCropBounds').checked,
    };
  }

  get handoutOptions(): PhotoHandoutOptions {
    const splitWaypoint = required<HTMLSelectElement>('handoutSplit').value;
    const splitIndex = this.route.findIndex(([name]) => name === splitWaypoint);
    const route = this.route.length >= 2 ? buildRoute(this.route) : null;
    const splitAfterM =
      route && splitIndex > 0 && splitIndex < this.route.length - 1
        ? route.legs[splitIndex - 1].cumulativeStart + route.legs[splitIndex - 1].length
        : null;
    return {
      splitWaypoint: splitWaypoint || 'No valid internal boundary',
      splitAfterM,
      includeSummary: required<HTMLInputElement>('handoutSummary').checked,
    };
  }

  private async importOne(
    file: File,
    override: ImportOverride = {},
    existingRecords = this.records
  ): Promise<PhotoRecord | null> {
    if (!/^image\/jpeg$/i.test(file.type) && !/\.jpe?g$/i.test(file.name)) {
      this.onMessage(`Skipped ${file.name}: only JPEG photos are supported.`, 'warning');
      return null;
    }
    if (file.size > PHOTO_IMPORT_LIMITS.maximumFileBytes) {
      throw new Error(
        `${file.name} exceeds the ${PHOTO_IMPORT_LIMITS.maximumFileBytes / 1024 / 1024} MB limit.`
      );
    }
    const buffer = await file.arrayBuffer();
    const hash = await hashArrayBufferSha256(buffer);
    if (isDuplicatePhoto(file, hash, existingRecords)) {
      this.onMessage(`Skipped duplicate photo: ${file.name}.`, 'warning');
      return null;
    }
    let previewUrl = '';
    let width = 0;
    let height = 0;
    let importError: string | null = null;
    let metadata: PhotoMetadata;
    try {
      const sourceUrl = URL.createObjectURL(file);
      try {
        [width, height] = await imageDimensions(sourceUrl);
      } finally {
        URL.revokeObjectURL(sourceUrl);
      }
      if ((width * height) / 1_000_000 > PHOTO_IMPORT_LIMITS.maximumMegapixels) {
        throw new Error(`${file.name} exceeds the ${PHOTO_IMPORT_LIMITS.maximumMegapixels} megapixel limit.`);
      }
      metadata = await extractPhotoMetadata(file, buffer);
      const thumbnail = await preparePhotoJpeg(
        file,
        metadata.orientation.value ?? 1,
        PHOTO_IMPORT_LIMITS.thumbnailEdge
      );
      previewUrl = URL.createObjectURL(new Blob([Uint8Array.from(thumbnail).buffer], { type: 'image/jpeg' }));
    } catch (error) {
      importError = error instanceof Error ? error.message : String(error);
      metadata = emptyMetadata(importError);
      previewUrl = URL.createObjectURL(file);
    }
    const inferred = inferClassification(file.name);
    const item = override.item;
    const originalMetadata = cloneMetadata(metadata);
    if (item) {
      metadata.latitude = {
        value: item.manualOverrides.latitude,
        source: 'example-track',
        reliable: true,
        note: 'Interpolated from the original example GPX track.',
      };
      metadata.longitude = {
        value: item.manualOverrides.longitude,
        source: 'example-track',
        reliable: true,
        note: 'Interpolated from the original example GPX track.',
      };
    }
    return {
      id: randomId(),
      file,
      fileName: file.name,
      fileSize: file.size,
      contentHash: hash,
      previewUrl,
      width,
      height,
      originalMetadata,
      metadata,
      classification: item?.classification ?? inferred.classification,
      identifier: item?.identifier ?? alphabeticIdentifier(existingRecords.length),
      linkedWaypoint: item?.linkedWaypoint ?? inferred.linkedWaypoint,
      taskLatitude: missingValue('Task/object latitude has not been supplied.'),
      taskLongitude: missingValue('Task/object longitude has not been supplied.'),
      manualLegIndex: null,
      order: existingRecords.length,
      importError,
      analysis: null,
      taskAnalysis: null,
      findings: [],
      exceptionAccepted: false,
      exceptionAcceptedAt: null,
      isExample: Boolean(item),
      fieldErrors: {},
      fieldDrafts: {},
    };
  }

  async importFiles(files: File[], overrides = new Map<string, ImportOverride>()): Promise<void> {
    if (files.length === 0) return;
    if (this.importing) {
      this.onMessage('Photo import is already in progress.', 'warning');
      return;
    }
    if (this.records.length + files.length > PHOTO_IMPORT_LIMITS.maximumCount) {
      this.onMessage(`Error: a maximum of ${PHOTO_IMPORT_LIMITS.maximumCount} photos is allowed.`, 'error');
      return;
    }
    const totalBytes = [
      ...this.records.map((record) => record.fileSize),
      ...files.map((file) => file.size),
    ].reduce((sum, bytes) => sum + bytes, 0);
    if (totalBytes > PHOTO_IMPORT_LIMITS.maximumTotalBytes) {
      this.onMessage(
        `Error: selected photos exceed the ${PHOTO_IMPORT_LIMITS.maximumTotalBytes / 1024 / 1024} MB total limit.`,
        'error'
      );
      return;
    }
    this.setImporting(true);
    this.progress.hidden = false;
    this.progress.max = files.length;
    this.progress.value = 0;
    let imported = 0;
    const errors: string[] = [];
    try {
      for (const [index, file] of files.entries()) {
        this.progressText.textContent = `Reading photo ${index + 1} of ${files.length}: ${file.name}`;
        try {
          const record = await this.importOne(file, overrides.get(file.name));
          if (record) {
            this.records.push(record);
            imported += 1;
          }
        } catch (error) {
          errors.push(`${file.name}: ${error instanceof Error ? error.message : String(error)}`);
        }
        this.progress.value = index + 1;
      }
      const normalized = markRepeatedGps(this.records.map((record) => record.metadata));
      this.records.forEach((record, index) => {
        if (record.metadata.latitude.source === 'exif') record.metadata.latitude = normalized[index].latitude;
        if (record.metadata.longitude.source === 'exif')
          record.metadata.longitude = normalized[index].longitude;
        record.order = index;
      });
      this.analyze(this.route);
      this.progressText.textContent = `${imported} of ${files.length} photos imported.${errors.length ? ` ${errors.length} failed: ${errors.join(' | ')}` : ''}`;
      this.onMessage(
        errors.length
          ? `Imported ${imported} photo(s); ${errors.length} failed. Review the persistent import details.`
          : `Imported ${imported} photo${imported === 1 ? '' : 's'} locally. Nothing was uploaded.`,
        errors.length ? 'warning' : imported ? 'success' : 'warning'
      );
    } finally {
      this.progress.hidden = true;
      this.input.value = '';
      this.setImporting(false);
    }
  }

  async loadExample(): Promise<void> {
    if (this.importing) {
      this.onMessage('Photo import is already in progress.', 'warning');
      return;
    }
    this.setImporting(true);
    const staged: PhotoRecord[] = [];
    try {
      const response = await fetch(new URL('examples/photo-manifest.json', document.baseURI));
      if (!response.ok) throw new Error(`manifest returned ${response.status}`);
      const manifest = (await response.json()) as ExamplePhotoManifest;
      this.progress.hidden = false;
      this.progress.max = manifest.items.length;
      if (manifest.items.length > PHOTO_IMPORT_LIMITS.maximumCount) {
        throw new Error(`manifest exceeds the ${PHOTO_IMPORT_LIMITS.maximumCount} photo limit`);
      }
      let totalBytes = 0;
      for (const [index, item] of manifest.items.entries()) {
        this.progressText.textContent = `Loading example ${index + 1} of ${manifest.items.length}: ${item.fileName}`;
        const photoResponse = await fetch(new URL(item.path, document.baseURI));
        if (!photoResponse.ok) throw new Error(`${item.fileName} returned ${photoResponse.status}`);
        const blob = await photoResponse.blob();
        totalBytes += blob.size;
        if (totalBytes > PHOTO_IMPORT_LIMITS.maximumTotalBytes) {
          throw new Error(
            `example exceeds the ${PHOTO_IMPORT_LIMITS.maximumTotalBytes / 1024 / 1024} MB limit`
          );
        }
        const file = new File([blob], item.fileName, { type: 'image/jpeg' });
        const record = await this.importOne(file, { item }, staged);
        if (record) staged.push(record);
        this.progress.value = index + 1;
      }
      const routeTextarea = required<HTMLTextAreaElement>('waypoints');
      routeTextarea.value = manifest.route
        .map(([name, latitude, longitude]) => `${name},${latitude},${longitude}`)
        .join('\n');
      this.records.forEach((record) => {
        URL.revokeObjectURL(record.previewUrl);
      });
      this.records = staged;
      this.analyze(manifest.route);
      this.onMessage(`Loaded ${manifest.label}. Its 29 photos remain in this browser tab only.`, 'success');
    } catch (error) {
      staged.forEach((record) => {
        URL.revokeObjectURL(record.previewUrl);
      });
      this.onMessage(
        `Error: example photos could not be loaded (${error instanceof Error ? error.message : String(error)}).`,
        'error'
      );
    } finally {
      this.progress.hidden = true;
      this.setImporting(false);
    }
  }

  private setImporting(importing: boolean): void {
    this.importing = importing;
    this.input.disabled = importing || this.externallyBusy;
    this.loadExampleButton.disabled = importing || this.externallyBusy;
    this.drop.setAttribute('aria-busy', String(importing));
    this.list.setAttribute('aria-busy', String(importing));
  }

  analyze(points: Waypoint[]): PhotoComplianceSummary {
    this.route = points;
    const splitSelect = required<HTMLSelectElement>('handoutSplit');
    const previousSplit = splitSelect.value;
    splitSelect.replaceChildren(
      ...points.slice(1, -1).map(([name]) => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        return option;
      })
    );
    splitSelect.disabled = points.length < 3;
    if (Array.from(splitSelect.options).some((option) => option.value === previousSplit)) {
      splitSelect.value = previousSplit;
    } else if (splitSelect.options.length > 0) {
      splitSelect.selectedIndex = Math.floor(splitSelect.options.length / 2);
    }
    const route = points.length >= 2 ? buildRoute(points) : null;
    for (const record of this.records) {
      const latitude = record.metadata.latitude.value;
      const longitude = record.metadata.longitude.value;
      const heading =
        record.metadata.headingReference.value === 'true' && record.metadata.headingReference.reliable
          ? record.metadata.headingDeg.value
          : null;
      record.analysis =
        latitude !== null && longitude !== null && route
          ? analyzePhotoPosition(latitude, longitude, heading, route, points, record.manualLegIndex)
          : null;
      const linked = record.linkedWaypoint
        ? points.find(([name]) => name.toUpperCase() === record.linkedWaypoint?.toUpperCase())
        : undefined;
      const taskLatitude = record.taskLatitude.reliable ? record.taskLatitude.value : (linked?.[1] ?? null);
      const taskLongitude = record.taskLongitude.reliable
        ? record.taskLongitude.value
        : (linked?.[2] ?? null);
      record.taskAnalysis =
        taskLatitude !== null && taskLongitude !== null && route
          ? analyzePhotoPosition(taskLatitude, taskLongitude, null, route, points, record.manualLegIndex)
          : null;
    }
    this.compliance = evaluatePhotoCompliance(this.records, points);
    const enroute = this.records.filter((record) => record.classification === 'enroute');
    if (enroute.length > 0) {
      const splitAfterM = this.handoutOptions.splitAfterM;
      const positioned = enroute.filter((record) => record.analysis);
      const before = positioned.filter((record) => (record.analysis?.alongRouteM ?? 0) <= (splitAfterM ?? 0));
      const after = positioned.filter((record) => (record.analysis?.alongRouteM ?? 0) > (splitAfterM ?? 0));
      const complete = splitAfterM !== null && positioned.length === enroute.length;
      const splitValid = complete && before.length > 0 && after.length > 0;
      this.compliance.findings.push({
        photoId: null,
        severity: splitValid ? 'pass' : complete ? 'violation' : 'warning',
        code: 'handout-two-set-split',
        rule: 'A2.4.5',
        affected: 'route',
        message: splitValid
          ? 'The selected handout boundary produces two non-empty en-route photo sets.'
          : complete
            ? 'The selected handout boundary leaves one en-route photo set empty.'
            : 'The required two-set handout split needs a valid boundary and reliable route positions.',
        measured: `${before.length} before / ${after.length} after / ${enroute.length - positioned.length} unavailable`,
        permitted: 'two non-empty sets separated at a valid internal route boundary',
      });
      this.compliance.violationCount = this.compliance.findings.filter(
        (finding) => finding.severity === 'violation'
      ).length;
      this.compliance.warningCount = this.compliance.findings.filter(
        (finding) => finding.severity === 'warning'
      ).length;
      this.compliance.status =
        this.compliance.violationCount > 0
          ? 'against-rules'
          : this.compliance.warningCount > 0
            ? 'manual-review'
            : 'ok';
    }
    for (const record of this.records) {
      record.findings = this.compliance.findings.filter((finding) => finding.photoId === record.id);
    }
    this.render();
    return this.compliance;
  }

  private handleClick(event: Event): void {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>('button[data-action]');
    if (!button) return;
    const index = this.records.findIndex((record) => record.id === button.dataset.photoId);
    if (index < 0) return;
    const action = button.dataset.action;
    if (action === 'remove') {
      const removedName = this.records[index].fileName;
      URL.revokeObjectURL(this.records[index].previewUrl);
      this.records.splice(index, 1);
      this.onMessage(`Removed ${removedName}.`, 'neutral');
    } else if (action === 'up' && index > 0) {
      [this.records[index - 1], this.records[index]] = [this.records[index], this.records[index - 1]];
      this.onMessage(`Moved ${this.records[index - 1].fileName} up.`, 'neutral');
    } else if (action === 'down' && index < this.records.length - 1) {
      [this.records[index], this.records[index + 1]] = [this.records[index + 1], this.records[index]];
      this.onMessage(`Moved ${this.records[index + 1].fileName} down.`, 'neutral');
    } else if (action === 'accept-exception') {
      const record = this.records[index];
      record.exceptionAccepted = !record.exceptionAccepted;
      record.exceptionAcceptedAt = record.exceptionAccepted ? new Date().toISOString() : null;
      this.onMessage(
        record.exceptionAccepted
          ? `Accepted ${record.fileName} as a documented judging exception; its violations remain recorded.`
          : `Removed the judging exception from ${record.fileName}.`,
        record.exceptionAccepted ? 'warning' : 'neutral'
      );
    }
    this.records.forEach((record, order) => {
      record.order = order;
    });
    this.analyze(this.route);
  }

  private handleChange(event: Event): void {
    const control = event.target as HTMLInputElement | HTMLSelectElement;
    const record = this.records.find((item) => item.id === control.dataset.photoId);
    const field = control.dataset.field;
    if (!record || !field) return;
    const focusRecordId = record.id;
    const focusField = field;
    record.fieldErrors ??= {};
    record.fieldDrafts ??= {};
    const fieldErrors = record.fieldErrors;
    const fieldDrafts = record.fieldDrafts;
    const applyNumber = (key: keyof PhotoMetadata | 'taskLatitude' | 'taskLongitude') => {
      const result = validateManualNumber(field, control.value);
      if (result.error) {
        fieldErrors[field] = result.error;
        fieldDrafts[field] = control.value;
      } else {
        delete fieldErrors[field];
        delete fieldDrafts[field];
      }
      if (key === 'taskLatitude' || key === 'taskLongitude') {
        record[key] = applyManualValue(record[key], result.value);
      } else {
        record.metadata[key] = applyManualValue(record.metadata[key] as never, result.value) as never;
      }
    };
    if (field === 'classification') {
      const updated = applyPhotoClassification(
        record,
        control.value as PhotoClassification,
        record.linkedWaypoint
      );
      record.classification = updated.classification;
    } else if (field === 'identifier') {
      const identifier = control.value.trim().toUpperCase();
      const duplicate = this.records.some(
        (item) => item.id !== record.id && item.identifier.trim().toUpperCase() === identifier
      );
      const error = !/^[A-Z]$/.test(identifier)
        ? 'Use one letter from A to Z.'
        : duplicate
          ? 'Identifier must be unique.'
          : null;
      if (error) {
        fieldErrors[field] = error;
        fieldDrafts[field] = control.value;
        record.identifier = '';
      } else {
        delete fieldErrors[field];
        delete fieldDrafts[field];
        record.identifier = identifier;
      }
    } else if (field === 'linkedWaypoint') record.linkedWaypoint = control.value || null;
    else if (field === 'manualLegIndex') {
      const parsed = control.value === '' ? null : Number(control.value);
      record.manualLegIndex = Number.isInteger(parsed) ? parsed : null;
    } else if (field === 'latitude') applyNumber('latitude');
    else if (field === 'longitude') applyNumber('longitude');
    else if (field === 'headingDeg') {
      applyNumber('headingDeg');
      record.metadata.headingReference =
        record.metadata.headingDeg.value === null
          ? missingValue('Manual heading is unavailable.')
          : { value: 'true', source: 'manual', reliable: true };
    } else if (field === 'altitudeAglFt') applyNumber('altitudeAglFt');
    else if (field === 'focalLength35Mm') applyNumber('focalLength35Mm');
    else if (field === 'captureTime')
      record.metadata.captureTime = applyManualValue(
        record.metadata.captureTime,
        control.value.trim() || null
      );
    else if (field === 'taskLatitude') applyNumber('taskLatitude');
    else if (field === 'taskLongitude') applyNumber('taskLongitude');
    this.analyze(this.route);
    requestAnimationFrame(() => {
      Array.from(this.list.querySelectorAll<HTMLElement>('[data-photo-id][data-field]'))
        .find((element) => element.dataset.photoId === focusRecordId && element.dataset.field === focusField)
        ?.focus();
    });
  }

  private render(): void {
    this.empty.hidden = this.records.length > 0;
    const fragment = document.createDocumentFragment();
    const waypointOptions: Array<[string, string]> = [
      ['', 'Not linked'],
      ...this.route.map(([name]) => [name, name] as [string, string]),
    ];
    const legOptions: Array<[string, string]> = [
      ['', 'Automatic leg matching'],
      ...this.route
        .slice(0, -1)
        .map(
          ([name], index) =>
            [String(index), `${index + 1}: ${name}-${this.route[index + 1][0]}`] as [string, string]
        ),
    ];
    for (const [index, record] of this.records.entries()) {
      const article = document.createElement('article');
      article.className = 'photo-card';
      article.dataset.photoId = record.id;
      const image = document.createElement('img');
      image.src = record.previewUrl;
      image.alt = `Preview of ${record.fileName}`;
      image.className = 'photo-thumbnail';
      const body = document.createElement('div');
      body.className = 'photo-card-body';
      const heading = document.createElement('div');
      heading.className = 'photo-card-heading';
      const title = document.createElement('strong');
      title.textContent = `${index + 1}. ${record.fileName}`;
      const controls = document.createElement('div');
      controls.className = 'photo-order-controls';
      for (const [action, label] of [
        ['up', '↑'],
        ['down', '↓'],
        ['remove', 'Remove'],
      ]) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn btn-small';
        button.dataset.action = action;
        button.dataset.photoId = record.id;
        button.textContent = label;
        button.disabled =
          (action === 'up' && index === 0) || (action === 'down' && index === this.records.length - 1);
        button.setAttribute(
          'aria-label',
          action === 'up'
            ? `Move ${record.fileName} up`
            : action === 'down'
              ? `Move ${record.fileName} down`
              : `Remove ${record.fileName}`
        );
        controls.appendChild(button);
      }
      heading.append(title, controls);
      const status = document.createElement('span');
      status.className = 'photo-status';
      const hasViolation = record.findings.some((finding) => finding.severity === 'violation');
      const hasWarning = record.findings.some((finding) => finding.severity === 'warning');
      if (!hasViolation && record.exceptionAccepted) {
        record.exceptionAccepted = false;
        record.exceptionAcceptedAt = null;
      }
      status.dataset.tone = hasViolation
        ? record.exceptionAccepted
          ? 'accepted'
          : 'fail'
        : hasWarning
          ? 'review'
          : 'ok';
      status.textContent = hasViolation
        ? record.exceptionAccepted
          ? 'Accepted exception · still against the rules'
          : 'Against the rules'
        : hasWarning
          ? 'Manual review required'
          : 'OK for automated checks';
      const exceptionButton = document.createElement('button');
      exceptionButton.type = 'button';
      exceptionButton.className = 'btn btn-small photo-exception-button';
      exceptionButton.dataset.action = 'accept-exception';
      exceptionButton.dataset.photoId = record.id;
      exceptionButton.textContent = record.exceptionAccepted
        ? 'Revoke accepted exception'
        : 'Accept exception';
      exceptionButton.hidden = !hasViolation;
      exceptionButton.setAttribute('aria-pressed', String(record.exceptionAccepted));
      exceptionButton.title =
        'Keeps all violations in the audit record and accepts this photo for judging output.';
      const grid = document.createElement('div');
      grid.className = 'photo-field-grid';
      const numericInput = (label: string, field: string, value: number | null) =>
        input(
          label,
          record.fieldDrafts?.[field] ?? value?.toString() ?? '',
          field,
          record.id,
          'number',
          record.fieldErrors?.[field] ?? null
        );
      grid.append(
        select(
          'Classification',
          record.classification,
          'classification',
          record.id,
          PHOTO_CLASSIFICATIONS.map((value) => [value, CLASS_LABELS[value]])
        ),
        input(
          'Identifier',
          record.fieldDrafts?.identifier ?? record.identifier,
          'identifier',
          record.id,
          'text',
          record.fieldErrors?.identifier ?? null
        ),
        select('Linked waypoint', record.linkedWaypoint ?? '', 'linkedWaypoint', record.id, waypointOptions),
        select(
          'Route leg override',
          record.manualLegIndex?.toString() ?? '',
          'manualLegIndex',
          record.id,
          legOptions
        ),
        numericInput('Latitude', 'latitude', record.metadata.latitude.value),
        numericInput('Longitude', 'longitude', record.metadata.longitude.value),
        numericInput('Heading (°)', 'headingDeg', record.metadata.headingDeg.value),
        numericInput('Altitude AGL (ft)', 'altitudeAglFt', record.metadata.altitudeAglFt.value),
        numericInput('35 mm equiv. focal (mm)', 'focalLength35Mm', record.metadata.focalLength35Mm.value),
        input('Capture time', record.metadata.captureTime.value ?? '', 'captureTime', record.id),
        numericInput('Task/object latitude', 'taskLatitude', record.taskLatitude.value),
        numericInput('Task/object longitude', 'taskLongitude', record.taskLongitude.value)
      );
      const provenance = document.createElement('p');
      provenance.className = 'photo-provenance';
      const originalPosition = [
        record.originalMetadata.latitude.value,
        record.originalMetadata.longitude.value,
      ].every((value) => value !== null)
        ? `; original EXIF GPS: ${record.originalMetadata.latitude.value}, ${record.originalMetadata.longitude.value}`
        : '';
      provenance.textContent = `${record.width}×${record.height}px · Position: ${record.metadata.latitude.source}/${record.metadata.longitude.source}${originalPosition}; EXIF altitude: ${record.metadata.gpsAltitudeMslM.value ?? 'missing'} m MSL; camera: ${[record.metadata.cameraMake.value, record.metadata.cameraModel.value].filter(Boolean).join(' ') || 'missing'}; orientation: ${record.metadata.orientation.value ?? 'missing'}.`;
      const metrics = document.createElement('p');
      metrics.className = 'photo-metrics';
      metrics.textContent = record.analysis
        ? `${record.analysis.legId} · ${(record.analysis.alongRouteM / 1852).toFixed(2)} NM along route · ${record.analysis.lateralDistanceM.toFixed(0)} m lateral · ${(record.analysis.distanceAfterPreviousControlPointM / 1852).toFixed(2)} NM after ${record.analysis.previousControlPoint}`
        : 'Route metrics unavailable until both position and a valid route are present.';
      const issueList = document.createElement('ul');
      issueList.className = 'photo-card-findings';
      for (const finding of record.findings.filter((item) => item.severity !== 'pass')) {
        const item = document.createElement('li');
        item.textContent = `${finding.rule}: ${finding.measured}; permitted ${finding.permitted}.${record.exceptionAccepted && finding.severity === 'violation' ? ' Accepted as a judge exception; violation retained.' : ''}`;
        issueList.appendChild(item);
      }
      if (record.importError) {
        const error = document.createElement('p');
        error.className = 'danger-text';
        error.textContent = `Metadata error: ${record.importError}`;
        body.append(heading, status, exceptionButton, error, grid, provenance, metrics, issueList);
      } else body.append(heading, status, exceptionButton, grid, provenance, metrics, issueList);
      article.append(image, body);
      fragment.appendChild(article);
    }
    this.list.replaceChildren(fragment);
    if (this.externallyBusy) {
      this.list
        .querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>('input,select,button')
        .forEach((control) => {
          control.disabled = true;
        });
    }
    const tone =
      this.compliance.status === 'against-rules'
        ? 'fail'
        : this.compliance.status === 'manual-review'
          ? 'review'
          : 'ok';
    this.status.dataset.tone = tone;
    this.status.textContent =
      this.compliance.status === 'against-rules'
        ? `Against the rules · ${this.compliance.violationCount} violation(s), ${this.compliance.warningCount} manual review item(s)`
        : this.compliance.status === 'manual-review'
          ? `Manual review required · ${this.compliance.warningCount} item(s) could not be automated`
          : 'OK for automated photo checks';
    const findingFragment = document.createDocumentFragment();
    for (const finding of this.compliance.findings.filter((item) => item.severity !== 'pass')) {
      const item = document.createElement('li');
      const accepted =
        finding.photoId !== null &&
        finding.severity === 'violation' &&
        this.records.some((record) => record.id === finding.photoId && record.exceptionAccepted);
      item.textContent = `${finding.rule} · ${finding.affected}: ${finding.measured}; permitted ${finding.permitted}.${accepted ? ' Accepted as a judge exception; violation retained.' : ''}`;
      findingFragment.appendChild(item);
    }
    this.findings.replaceChildren(findingFragment);
  }
}
