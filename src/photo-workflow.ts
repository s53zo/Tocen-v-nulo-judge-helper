import { buildRoute, type Waypoint } from './domain';
import { analyzePhotoPosition } from './photo-analysis';
import { evaluatePhotoCompliance } from './photo-compliance';
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

function required<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Required photo workflow element is missing: #${id}`);
  return element as T;
}

function numberValue(value: string): number | null {
  if (!value.trim()) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function emptyMetadata(note: string): PhotoMetadata {
  return {
    latitude: missingValue(note),
    longitude: missingValue(note),
    gpsAltitudeMslM: missingValue(note),
    altitudeAglFt: missingValue(note),
    headingDeg: missingValue(note),
    focalLengthMm: missingValue(note),
    focalLength35Mm: missingValue(note),
    captureTime: missingValue(note),
    orientation: missingValue(note),
    cameraMake: missingValue(note),
    cameraModel: missingValue(note),
    lensModel: missingValue(note),
  };
}

export async function hashFileSha256(file: File): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
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

function input(label: string, value: string, field: string, id: string, type = 'text'): HTMLElement {
  const wrapper = document.createElement('label');
  wrapper.className = 'photo-field';
  wrapper.append(document.createTextNode(label));
  const control = document.createElement('input');
  control.type = type;
  control.value = value;
  control.dataset.photoId = id;
  control.dataset.field = field;
  if (type === 'number') control.step = 'any';
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
    return {
      splitWaypoint: required<HTMLSelectElement>('handoutSplit').value || 'TP5',
      includeSummary: required<HTMLInputElement>('handoutSummary').checked,
    };
  }

  private async importOne(file: File, override: ImportOverride = {}): Promise<PhotoRecord | null> {
    if (!/^image\/jpeg$/i.test(file.type) && !/\.jpe?g$/i.test(file.name)) {
      this.onMessage(`Skipped ${file.name}: only JPEG photos are supported.`, 'warning');
      return null;
    }
    const hash = await hashFileSha256(file);
    if (isDuplicatePhoto(file, hash, this.records)) {
      this.onMessage(`Skipped duplicate photo: ${file.name}.`, 'warning');
      return null;
    }
    const previewUrl = URL.createObjectURL(file);
    let width = 0;
    let height = 0;
    let importError: string | null = null;
    let metadata: PhotoMetadata;
    try {
      [width, height] = await imageDimensions(previewUrl);
      metadata = await extractPhotoMetadata(file);
    } catch (error) {
      importError = error instanceof Error ? error.message : String(error);
      metadata = emptyMetadata(importError);
    }
    const inferred = inferClassification(file.name);
    const item = override.item;
    const originalMetadata = structuredClone(metadata);
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
      id: crypto.randomUUID(),
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
      identifier: item?.identifier ?? String.fromCharCode(65 + (this.records.length % 26)),
      linkedWaypoint: item?.linkedWaypoint ?? inferred.linkedWaypoint,
      subjectLatitude: null,
      subjectLongitude: null,
      order: this.records.length,
      importError,
      analysis: null,
      findings: [],
      isExample: Boolean(item),
    };
  }

  async importFiles(files: File[], overrides = new Map<string, ImportOverride>()): Promise<void> {
    if (files.length === 0) return;
    this.progress.hidden = false;
    this.progress.max = files.length;
    this.progress.value = 0;
    let imported = 0;
    for (const [index, file] of files.entries()) {
      this.progressText.textContent = `Reading photo ${index + 1} of ${files.length}: ${file.name}`;
      try {
        const record = await this.importOne(file, overrides.get(file.name));
        if (record) {
          this.records.push(record);
          imported += 1;
        }
      } catch (error) {
        this.onMessage(
          `Could not import ${file.name}: ${error instanceof Error ? error.message : String(error)}`,
          'error'
        );
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
    this.progress.hidden = true;
    this.progressText.textContent = `${imported} of ${files.length} photos imported.`;
    this.analyze(this.route);
    this.onMessage(
      `Imported ${imported} photo${imported === 1 ? '' : 's'} locally. Nothing was uploaded.`,
      imported ? 'success' : 'warning'
    );
    this.input.value = '';
  }

  async loadExample(): Promise<void> {
    this.loadExampleButton.disabled = true;
    try {
      const response = await fetch(new URL('examples/photo-manifest.json', document.baseURI));
      if (!response.ok) throw new Error(`manifest returned ${response.status}`);
      const manifest = (await response.json()) as ExamplePhotoManifest;
      this.progress.hidden = false;
      this.progress.max = manifest.items.length;
      const files: File[] = [];
      const overrides = new Map<string, ImportOverride>();
      for (const [index, item] of manifest.items.entries()) {
        this.progressText.textContent = `Loading example ${index + 1} of ${manifest.items.length}: ${item.fileName}`;
        const photoResponse = await fetch(new URL(item.path, document.baseURI));
        if (!photoResponse.ok) throw new Error(`${item.fileName} returned ${photoResponse.status}`);
        files.push(new File([await photoResponse.blob()], item.fileName, { type: 'image/jpeg' }));
        overrides.set(item.fileName, { item });
        this.progress.value = index + 1;
      }
      const routeTextarea = required<HTMLTextAreaElement>('waypoints');
      routeTextarea.value = manifest.route
        .map(([name, latitude, longitude]) => `${name},${latitude},${longitude}`)
        .join('\n');
      await this.importFiles(files, overrides);
      this.onMessage(`Loaded ${manifest.label}. Its 29 photos remain in this browser tab only.`, 'success');
    } catch (error) {
      this.onMessage(
        `Error: example photos could not be loaded (${error instanceof Error ? error.message : String(error)}).`,
        'error'
      );
    } finally {
      this.loadExampleButton.disabled = false;
    }
  }

  analyze(points: Waypoint[]): PhotoComplianceSummary {
    this.route = points;
    const route = points.length >= 2 ? buildRoute(points) : null;
    for (const record of this.records) {
      const latitude = record.metadata.latitude.value;
      const longitude = record.metadata.longitude.value;
      record.analysis =
        latitude !== null && longitude !== null && route
          ? analyzePhotoPosition(latitude, longitude, record.metadata.headingDeg.value, route, points)
          : null;
    }
    this.compliance = evaluatePhotoCompliance(this.records, points);
    for (const record of this.records) {
      record.findings = this.compliance.findings.filter((finding) =>
        finding.affected.includes(`(${record.fileName})`)
      );
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
      URL.revokeObjectURL(this.records[index].previewUrl);
      this.records.splice(index, 1);
    } else if (action === 'up' && index > 0) {
      [this.records[index - 1], this.records[index]] = [this.records[index], this.records[index - 1]];
    } else if (action === 'down' && index < this.records.length - 1) {
      [this.records[index], this.records[index + 1]] = [this.records[index + 1], this.records[index]];
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
    if (field === 'classification') {
      const updated = applyPhotoClassification(
        record,
        control.value as PhotoClassification,
        record.linkedWaypoint
      );
      record.classification = updated.classification;
    } else if (field === 'identifier') record.identifier = control.value.trim();
    else if (field === 'linkedWaypoint') record.linkedWaypoint = control.value || null;
    else if (field === 'latitude')
      record.metadata.latitude = applyManualValue(record.metadata.latitude, numberValue(control.value));
    else if (field === 'longitude')
      record.metadata.longitude = applyManualValue(record.metadata.longitude, numberValue(control.value));
    else if (field === 'headingDeg')
      record.metadata.headingDeg = applyManualValue(record.metadata.headingDeg, numberValue(control.value));
    else if (field === 'altitudeAglFt')
      record.metadata.altitudeAglFt = applyManualValue(
        record.metadata.altitudeAglFt,
        numberValue(control.value)
      );
    else if (field === 'focalLength35Mm')
      record.metadata.focalLength35Mm = applyManualValue(
        record.metadata.focalLength35Mm,
        numberValue(control.value)
      );
    else if (field === 'captureTime')
      record.metadata.captureTime = applyManualValue(
        record.metadata.captureTime,
        control.value.trim() || null
      );
    else if (field === 'subjectLatitude') record.subjectLatitude = numberValue(control.value);
    else if (field === 'subjectLongitude') record.subjectLongitude = numberValue(control.value);
    this.analyze(this.route);
  }

  private render(): void {
    this.empty.hidden = this.records.length > 0;
    const fragment = document.createDocumentFragment();
    const waypointOptions: Array<[string, string]> = [
      ['', 'Not linked'],
      ...this.route.map(([name]) => [name, name] as [string, string]),
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
      status.dataset.tone = hasViolation ? 'fail' : hasWarning ? 'review' : 'ok';
      status.textContent = hasViolation
        ? 'Against the rules'
        : hasWarning
          ? 'Manual review required'
          : 'OK for automated checks';
      const grid = document.createElement('div');
      grid.className = 'photo-field-grid';
      grid.append(
        select(
          'Classification',
          record.classification,
          'classification',
          record.id,
          PHOTO_CLASSIFICATIONS.map((value) => [value, CLASS_LABELS[value]])
        ),
        input('Identifier', record.identifier, 'identifier', record.id),
        select('Linked waypoint', record.linkedWaypoint ?? '', 'linkedWaypoint', record.id, waypointOptions),
        input('Latitude', record.metadata.latitude.value?.toString() ?? '', 'latitude', record.id, 'number'),
        input(
          'Longitude',
          record.metadata.longitude.value?.toString() ?? '',
          'longitude',
          record.id,
          'number'
        ),
        input(
          'Heading (°)',
          record.metadata.headingDeg.value?.toString() ?? '',
          'headingDeg',
          record.id,
          'number'
        ),
        input(
          'Altitude AGL (ft)',
          record.metadata.altitudeAglFt.value?.toString() ?? '',
          'altitudeAglFt',
          record.id,
          'number'
        ),
        input(
          '35 mm equiv. focal (mm)',
          record.metadata.focalLength35Mm.value?.toString() ?? '',
          'focalLength35Mm',
          record.id,
          'number'
        ),
        input('Capture time', record.metadata.captureTime.value ?? '', 'captureTime', record.id),
        input(
          'False-object latitude',
          record.subjectLatitude?.toString() ?? '',
          'subjectLatitude',
          record.id,
          'number'
        ),
        input(
          'False-object longitude',
          record.subjectLongitude?.toString() ?? '',
          'subjectLongitude',
          record.id,
          'number'
        )
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
        item.textContent = `${finding.rule}: ${finding.measured}; permitted ${finding.permitted}.`;
        issueList.appendChild(item);
      }
      if (record.importError) {
        const error = document.createElement('p');
        error.className = 'danger-text';
        error.textContent = `Metadata error: ${record.importError}`;
        body.append(heading, status, error, grid, provenance, metrics, issueList);
      } else body.append(heading, status, grid, provenance, metrics, issueList);
      article.append(image, body);
      fragment.appendChild(article);
    }
    this.list.replaceChildren(fragment);
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
      item.textContent = `${finding.rule} · ${finding.affected}: ${finding.measured}; permitted ${finding.permitted}.`;
      findingFragment.appendChild(item);
    }
    this.findings.replaceChildren(findingFragment);
  }
}
