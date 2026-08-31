import { buildRoute, type Waypoint } from './domain';
import {
  GURS_DOF025_ATTRIBUTION,
  type OrthophotoCaptureModel,
  type OrthophotoTarget,
  orthophotoCoverage,
  orthophotoRequestUrl,
} from './orthophoto';
import { analyzePhotoPosition } from './photo-analysis';
import { evaluatePhotoCompliance, findingPresentation, operationalFindingCounts } from './photo-compliance';
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
  sourcedValue,
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
  orthophoto?: {
    target: OrthophotoTarget;
    model: OrthophotoCaptureModel;
  };
}

const CLASS_LABELS: Record<PhotoClassification, string> = {
  enroute: 'En-route photo',
  'control-correct': 'Correct control photo',
  'control-false': 'False control photo',
  'sign-task': 'Sign task',
  reference: 'Reference only',
};
const ORTHOPHOTO_BATCH_TIMEOUT_MS = 120_000;

export const PHOTO_IMPORT_LIMITS = {
  maximumCount: 60,
  maximumFileBytes: 25 * 1024 * 1024,
  maximumTotalBytes: 250 * 1024 * 1024,
  maximumMegapixels: 50,
  thumbnailEdge: 480,
} as const;

export interface OrthophotoImportResult {
  importedTargets: OrthophotoTarget[];
  failedTargets: Array<{ target: OrthophotoTarget; error: string }>;
  cancelled: boolean;
}

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

export function mixedRouteIndices(count: number, random: () => number = Math.random): number[] {
  if (!Number.isInteger(count) || count < 0) throw new Error('Photo count must be a non-negative integer.');
  const indices = Array.from({ length: count }, (_, index) => index);
  for (let index = indices.length - 1; index > 0; index -= 1) {
    const target = Math.floor(random() * (index + 1));
    [indices[index], indices[target]] = [indices[target], indices[index]];
  }
  const revealsForwardOrder = indices.every((value, index) => value === index);
  const revealsReverseOrder = indices.every((value, index) => value === count - index - 1);
  if (count === 2 && revealsForwardOrder) return [1, 0];
  if (count >= 3 && (revealsForwardOrder || revealsReverseOrder)) {
    const offset = Math.ceil(count / 2);
    return indices.map((_, index) => (index + offset) % count);
  }
  return indices;
}

export function createIdentifierMixRandom(salt: string): () => number {
  let seed = 0x811c9dc5;
  for (const character of salt) {
    seed ^= character.charCodeAt(0);
    seed = Math.imul(seed, 0x01000193);
  }
  return () => {
    seed += 0x6d2b79f5;
    let value = seed;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function identifierMixSalt(): string {
  return Array.from(crypto.getRandomValues(new Uint32Array(2)), (value) =>
    value.toString(16).padStart(8, '0')
  ).join('');
}

function routePosition(photo: PhotoRecord): number | null {
  return photo.taskAnalysis?.alongRouteM ?? photo.analysis?.alongRouteM ?? null;
}

export function lettersRevealRouteOrder(photos: PhotoRecord[]): boolean {
  if (photos.length < 2) return false;
  const routeOrdered = [...photos].sort(
    (left, right) => (routePosition(left) ?? 0) - (routePosition(right) ?? 0)
  );
  return routeOrdered.every(
    (photo, index) =>
      index === 0 ||
      photo.identifier.localeCompare(routeOrdered[index - 1].identifier, 'en', { sensitivity: 'base' }) > 0
  );
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

export function jpegDimensions(buffer: ArrayBuffer): [number, number] | null {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 4 || bytes[0] !== 0xff || bytes[1] !== 0xd8) return null;
  const startOfFrameMarkers = new Set([
    0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf,
  ]);
  let offset = 2;
  while (offset + 3 < bytes.length) {
    while (offset < bytes.length && bytes[offset] !== 0xff) offset += 1;
    while (offset < bytes.length && bytes[offset] === 0xff) offset += 1;
    if (offset >= bytes.length) break;
    const marker = bytes[offset];
    offset += 1;
    if (marker === 0xd9 || marker === 0xda) break;
    if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
    if (offset + 1 >= bytes.length) break;
    const length = (bytes[offset] << 8) | bytes[offset + 1];
    if (length < 2 || offset + length > bytes.length) break;
    if (startOfFrameMarkers.has(marker) && length >= 7) {
      const height = (bytes[offset + 3] << 8) | bytes[offset + 4];
      const width = (bytes[offset + 5] << 8) | bytes[offset + 6];
      return width > 0 && height > 0 ? [width, height] : null;
    }
    offset += length;
  }
  return null;
}

function abortMessage(signal: AbortSignal): Error {
  return new Error(signal.reason instanceof Error ? signal.reason.message : 'Operation cancelled.');
}

function withAbort<T>(operation: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return operation;
  if (signal.aborted) return Promise.reject(abortMessage(signal));
  return new Promise<T>((resolve, reject) => {
    const abort = () => reject(abortMessage(signal));
    signal.addEventListener('abort', abort, { once: true });
    operation.then(resolve, reject).finally(() => signal.removeEventListener('abort', abort));
  });
}

function imageDimensions(url: string, signal?: AbortSignal): Promise<[number, number]> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const cleanup = () => signal?.removeEventListener('abort', abort);
    const abort = () => {
      image.src = '';
      cleanup();
      reject(signal ? abortMessage(signal) : new Error('Operation cancelled.'));
    };
    image.onload = () => {
      cleanup();
      resolve([image.naturalWidth, image.naturalHeight]);
    };
    image.onerror = () => {
      cleanup();
      reject(new Error('The JPEG could not be decoded.'));
    };
    signal?.addEventListener('abort', abort, { once: true });
    if (signal?.aborted) return abort();
    image.src = url;
  });
}

async function imageDetailMetrics(
  blob: Blob,
  signal?: AbortSignal
): Promise<{ standardDeviation: number; edgeMean: number }> {
  const url = URL.createObjectURL(blob);
  try {
    const image = await new Promise<HTMLImageElement>((resolve, reject) => {
      const element = new Image();
      const cleanup = () => signal?.removeEventListener('abort', abort);
      const abort = () => {
        element.src = '';
        cleanup();
        reject(signal ? abortMessage(signal) : new Error('Operation cancelled.'));
      };
      element.onload = () => {
        cleanup();
        resolve(element);
      };
      element.onerror = () => {
        cleanup();
        reject(new Error('The DOF025 crop could not be decoded.'));
      };
      signal?.addEventListener('abort', abort, { once: true });
      if (signal?.aborted) return abort();
      element.src = url;
    });
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('The browser could not inspect the DOF025 crop.');
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    const grayscale = new Float32Array(canvas.width * canvas.height);
    let sum = 0;
    for (let index = 0; index < grayscale.length; index += 1) {
      const offset = index * 4;
      const value = pixels[offset] * 0.299 + pixels[offset + 1] * 0.587 + pixels[offset + 2] * 0.114;
      grayscale[index] = value;
      sum += value;
    }
    const mean = sum / grayscale.length;
    let squaredDifference = 0;
    let edgeSum = 0;
    let edgeCount = 0;
    for (let y = 0; y < canvas.height; y += 1) {
      for (let x = 0; x < canvas.width; x += 1) {
        const index = y * canvas.width + x;
        squaredDifference += (grayscale[index] - mean) ** 2;
        if (x > 0) {
          edgeSum += Math.abs(grayscale[index] - grayscale[index - 1]);
          edgeCount += 1;
        }
        if (y > 0) {
          edgeSum += Math.abs(grayscale[index] - grayscale[index - canvas.width]);
          edgeCount += 1;
        }
      }
    }
    return {
      standardDeviation: Math.sqrt(squaredDifference / grayscale.length),
      edgeMean: edgeCount ? edgeSum / edgeCount : 0,
    };
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function boundedResponseBlob(
  response: Response,
  maximumBytes: number,
  signal: AbortSignal
): Promise<Blob> {
  const declaredLength = Number(response.headers.get('content-length'));
  if (Number.isFinite(declaredLength) && declaredLength > maximumBytes) {
    throw new Error(`GURS WMS response exceeds the ${Math.round(maximumBytes / 1024 / 1024)} MB limit.`);
  }
  if (!response.body) {
    const blob = await response.blob();
    if (blob.size > maximumBytes) throw new Error('GURS WMS response exceeds the image byte limit.');
    return blob;
  }
  const reader = response.body.getReader();
  const chunks: ArrayBuffer[] = [];
  let total = 0;
  try {
    while (true) {
      if (signal.aborted) throw abortMessage(signal);
      const { done, value } = await withAbort(reader.read(), signal);
      if (done) break;
      total += value.byteLength;
      if (total > maximumBytes) throw new Error('GURS WMS response exceeds the image byte limit.');
      chunks.push(value.slice().buffer);
    }
  } finally {
    if (signal.aborted || total > maximumBytes) await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
  return new Blob(chunks, { type: response.headers.get('content-type') ?? '' });
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
  private readonly cancelImportButton = required<HTMLButtonElement>('cancelPhotoImport');
  private readonly input = required<HTMLInputElement>('photoFiles');
  private readonly drop = required<HTMLElement>('photoDropzone');
  private readonly status = required<HTMLElement>('photoComplianceStatus');
  private readonly findings = required<HTMLElement>('photoFindings');
  private readonly auditDetails = required<HTMLDetailsElement>('photoAuditDetails');
  private readonly auditSummary = required<HTMLElement>('photoAuditSummary');
  private readonly auditFindings = required<HTMLElement>('photoAuditFindings');
  private readonly loadExampleButton = required<HTMLButtonElement>('loadPhotoExample');
  private readonly mixLettersButton = required<HTMLButtonElement>('mixPhotoLetters');
  private importing = false;
  private externallyBusy = false;
  private activeImportController: AbortController | null = null;

  get isBusy(): boolean {
    return this.importing || this.externallyBusy;
  }

  setExternalBusy(busy: boolean): void {
    this.externallyBusy = busy;
    this.input.disabled = busy || this.importing;
    this.loadExampleButton.disabled = busy || this.importing;
    this.mixLettersButton.disabled = busy || this.importing;
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
    this.cancelImportButton.addEventListener('click', () => {
      this.activeImportController?.abort(new Error('Photo import cancelled by the user.'));
    });
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
    this.mixLettersButton.addEventListener('click', () => this.mixEnrouteIdentifiers());
    required<HTMLSelectElement>('handoutSplit').addEventListener('change', () => this.analyze(this.route));
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

  private mixEnrouteIdentifiers(announce = true): boolean {
    const enroute = this.records.filter((record) => record.classification === 'enroute');
    if (enroute.length === 0) {
      if (announce) this.onMessage('There are no en-route photos to mix.', 'warning');
      return false;
    }
    if (enroute.length > 26) {
      if (announce) this.onMessage('A maximum of 26 unique single-letter identifiers can be mixed.', 'error');
      return false;
    }
    const splitAfterM = this.handoutOptions.splitAfterM;
    if (splitAfterM === null || enroute.some((record) => routePosition(record) === null)) {
      if (announce) {
        this.onMessage(
          'Choose a valid internal split and provide reliable route positions before mixing letters.',
          'warning'
        );
      }
      return false;
    }
    const routeOrdered = [...enroute].sort(
      (left, right) => (routePosition(left) ?? 0) - (routePosition(right) ?? 0)
    );
    const groups = [
      routeOrdered.filter((record) => (routePosition(record) ?? 0) <= splitAfterM),
      routeOrdered.filter((record) => (routePosition(record) ?? 0) > splitAfterM),
    ];
    if (groups.some((group) => group.length === 0)) {
      if (announce)
        this.onMessage('The selected split must leave en-route photos in both route parts.', 'warning');
      return false;
    }
    const mixSalt = identifierMixSalt();
    const random = createIdentifierMixRandom(mixSalt);
    let letterOffset = 0;
    for (const group of groups) {
      const mixedIndices = mixedRouteIndices(group.length, random);
      mixedIndices.forEach((recordIndex, letterIndex) => {
        const record = group[recordIndex];
        record.identifier = alphabeticIdentifier(letterOffset + letterIndex);
        record.identifierMixSalt = mixSalt;
        record.identifierMixAlgorithm = 'fnv1a-mulberry32-fisher-yates-v1';
        delete record.fieldErrors?.identifier;
        delete record.fieldDrafts?.identifier;
      });
      letterOffset += group.length;
    }
    this.analyze(this.route);
    if (announce) {
      this.onMessage(
        `Mixed ${enroute.length} en-route photo letters independently before and after ${this.handoutOptions.splitWaypoint}.`,
        'success'
      );
    }
    return true;
  }

  private async importOne(
    file: File,
    override: ImportOverride = {},
    existingRecords = this.records,
    signal?: AbortSignal
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
    if (signal?.aborted) throw abortMessage(signal);
    const encodedDimensions = jpegDimensions(buffer);
    if (
      encodedDimensions &&
      (encodedDimensions[0] * encodedDimensions[1]) / 1_000_000 > PHOTO_IMPORT_LIMITS.maximumMegapixels
    ) {
      throw new Error(`${file.name} exceeds the ${PHOTO_IMPORT_LIMITS.maximumMegapixels} megapixel limit.`);
    }
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
        [width, height] = await imageDimensions(sourceUrl, signal);
      } finally {
        URL.revokeObjectURL(sourceUrl);
      }
      if ((width * height) / 1_000_000 > PHOTO_IMPORT_LIMITS.maximumMegapixels) {
        throw new Error(`${file.name} exceeds the ${PHOTO_IMPORT_LIMITS.maximumMegapixels} megapixel limit.`);
      }
      metadata = await withAbort(extractPhotoMetadata(file, buffer), signal);
      if (signal?.aborted) throw abortMessage(signal);
      const thumbnail = await withAbort(
        preparePhotoJpeg(file, metadata.orientation.value ?? 1, PHOTO_IMPORT_LIMITS.thumbnailEdge),
        signal
      );
      if (signal?.aborted) throw abortMessage(signal);
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
    const orthophoto = override.orthophoto;
    if (orthophoto) {
      const sourceNote = 'Centre of the requested GURS DOF025 crop; not camera EXIF.';
      metadata.latitude = sourcedValue(orthophoto.target.latitude, 'manual', true, sourceNote);
      metadata.longitude = sourcedValue(orthophoto.target.longitude, 'manual', true, sourceNote);
      metadata.focalLength35Mm = sourcedValue(
        orthophoto.model.focalLength35Mm,
        'manual',
        true,
        'Synthetic coverage model, not camera EXIF.'
      );
      metadata.cameraMake = sourcedValue('GURS', 'manual', true);
      metadata.cameraModel = sourcedValue('DOF025 orthophoto', 'manual', true);
    }
    const coverage = orthophoto ? orthophotoCoverage(orthophoto.model) : null;
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
      classification: orthophoto ? 'enroute' : (item?.classification ?? inferred.classification),
      identifier: item?.identifier ?? alphabeticIdentifier(existingRecords.length),
      linkedWaypoint: item?.linkedWaypoint ?? inferred.linkedWaypoint,
      taskLatitude: orthophoto
        ? sourcedValue(orthophoto.target.latitude, 'manual', true, 'Centre of the DOF025 crop.')
        : missingValue('Task/object latitude has not been supplied.'),
      taskLongitude: orthophoto
        ? sourcedValue(orthophoto.target.longitude, 'manual', true, 'Centre of the DOF025 crop.')
        : missingValue('Task/object longitude has not been supplied.'),
      manualLegIndex: null,
      order: existingRecords.length,
      importError,
      analysis: null,
      taskAnalysis: null,
      findings: [],
      exceptionAccepted: false,
      exceptionAcceptedAt: null,
      isExample: Boolean(item),
      ...(orthophoto && coverage
        ? {
            generatedOrthophoto: {
              provider: 'GURS' as const,
              layer: 'DOF025' as const,
              targetLabel: orthophoto.target.label,
              requestedAt: new Date().toISOString(),
              coverageWidthM: coverage.widthM,
              coverageHeightM: coverage.heightM,
              modeledAltitudeM: orthophoto.model.altitudeM,
              modeledFocalLength35Mm: orthophoto.model.focalLength35Mm,
              modeledDepressionDeg: orthophoto.model.depressionDeg,
              attribution: GURS_DOF025_ATTRIBUTION,
              ...(orthophoto.target.source ? { targetSource: orthophoto.target.source } : {}),
            },
          }
        : {}),
      fieldErrors: {},
      fieldDrafts: {},
    };
  }

  async importOrthophotoTargets(
    targets: OrthophotoTarget[],
    model: OrthophotoCaptureModel
  ): Promise<OrthophotoImportResult> {
    const emptyResult = (): OrthophotoImportResult => ({
      importedTargets: [],
      failedTargets: [],
      cancelled: false,
    });
    if (targets.length === 0) return emptyResult();
    if (this.isBusy) {
      this.onMessage('Another photo operation is already in progress.', 'warning');
      return emptyResult();
    }
    if (this.records.length + targets.length > PHOTO_IMPORT_LIMITS.maximumCount) {
      this.onMessage(`Error: a maximum of ${PHOTO_IMPORT_LIMITS.maximumCount} photos is allowed.`, 'error');
      return emptyResult();
    }
    const coverage = orthophotoCoverage(model);
    const batchController = new AbortController();
    const batchTimeout = window.setTimeout(
      () => batchController.abort(new Error('DOF025 batch exceeded two minutes.')),
      ORTHOPHOTO_BATCH_TIMEOUT_MS
    );
    this.activeImportController = batchController;
    this.setImporting(true);
    this.progress.hidden = false;
    this.progress.max = targets.length;
    this.progress.value = 0;
    const importedTargets: OrthophotoTarget[] = [];
    const failedTargets: Array<{ target: OrthophotoTarget; error: string }> = [];
    let cancelled = false;
    let importedBytes = this.records.reduce((total, record) => total + record.fileSize, 0);
    try {
      for (const [index, target] of targets.entries()) {
        if (batchController.signal.aborted) {
          cancelled = true;
          break;
        }
        this.progressText.textContent = `Requesting DOF025 crop ${index + 1} of ${targets.length}: ${target.label}`;
        const itemController = new AbortController();
        const abortItem = () => itemController.abort(batchController.signal.reason);
        batchController.signal.addEventListener('abort', abortItem, { once: true });
        const timeout = window.setTimeout(
          () => itemController.abort(new Error('DOF025 import timed out after 30 seconds.')),
          30_000
        );
        try {
          const response = await fetch(orthophotoRequestUrl(target.latitude, target.longitude, coverage), {
            signal: itemController.signal,
          });
          if (!response.ok) throw new Error(`GURS WMS returned HTTP ${response.status}`);
          const remainingBytes = PHOTO_IMPORT_LIMITS.maximumTotalBytes - importedBytes;
          if (remainingBytes <= 0) throw new Error('The total photo byte limit has been reached.');
          const blob = await boundedResponseBlob(
            response,
            Math.min(PHOTO_IMPORT_LIMITS.maximumFileBytes, remainingBytes),
            itemController.signal
          );
          if (!/^image\/jpeg(?:;|$)/i.test(blob.type)) {
            throw new Error(`GURS WMS returned ${blob.type || 'an unknown content type'} instead of JPEG`);
          }
          const signature = new Uint8Array(await blob.slice(0, 2).arrayBuffer());
          if (signature[0] !== 0xff || signature[1] !== 0xd8) {
            throw new Error('GURS WMS response does not contain JPEG bytes.');
          }
          const detail = await imageDetailMetrics(blob, itemController.signal);
          if (detail.standardDeviation < 4 && detail.edgeMean < 2) {
            throw new Error('DOF025 crop is nearly uniform and has too little visible detail.');
          }
          const safeLabel = target.label.replace(/[^A-Za-z0-9_-]+/g, '_').slice(0, 40) || 'TARGET';
          const file = new File(
            [blob],
            `DOF025_${safeLabel}_${target.latitude.toFixed(6)}_${target.longitude.toFixed(6)}.jpg`,
            { type: 'image/jpeg' }
          );
          const record = await this.importOne(
            file,
            { orthophoto: { target, model } },
            this.records,
            itemController.signal
          );
          if (record) {
            this.records.push(record);
            importedTargets.push(target);
            importedBytes += file.size;
          } else {
            throw new Error('The crop was not imported, usually because it duplicates an existing photo.');
          }
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          if (batchController.signal.aborted) {
            failedTargets.push({ target, error: message });
            cancelled = true;
            break;
          }
          failedTargets.push({ target, error: message });
        } finally {
          window.clearTimeout(timeout);
          batchController.signal.removeEventListener('abort', abortItem);
        }
        this.progress.value = index + 1;
      }
      if (cancelled) {
        const accounted = new Set([
          ...importedTargets.map((target) => target.label),
          ...failedTargets.map(({ target }) => target.label),
        ]);
        for (const target of targets) {
          if (!accounted.has(target.label)) {
            failedTargets.push({ target, error: 'Import cancelled before this crop was requested.' });
          }
        }
      }
      this.records.forEach((record, index) => {
        record.order = index;
      });
      this.analyze(this.route);
      this.mixEnrouteIdentifiers(false);
      const errors = failedTargets.map(({ target, error }) => `${target.label}: ${error}`);
      this.progressText.textContent = `${importedTargets.length} of ${targets.length} DOF025 crops imported.${cancelled ? ' Import cancelled.' : ''}${errors.length ? ` ${errors.length} failed: ${errors.join(' | ')}` : ''}`;
      this.onMessage(
        cancelled
          ? `DOF025 import cancelled after ${importedTargets.length} successful crop(s).`
          : errors.length
            ? `Imported ${importedTargets.length} DOF025 crop(s); ${errors.length} failed. Coordinates were sent to GURS.`
            : `Imported ${importedTargets.length} DOF025 crop${importedTargets.length === 1 ? '' : 's'} at approximately ${coverage.widthM.toFixed(0)} × ${coverage.heightM.toFixed(0)} m coverage.`,
        cancelled || errors.length ? 'warning' : 'success'
      );
      return { importedTargets, failedTargets, cancelled };
    } finally {
      window.clearTimeout(batchTimeout);
      this.progress.hidden = true;
      this.activeImportController = null;
      this.setImporting(false);
    }
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
      this.mixEnrouteIdentifiers(false);
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
      this.mixEnrouteIdentifiers(false);
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
    this.mixLettersButton.disabled = importing || this.externallyBusy;
    this.drop.setAttribute('aria-busy', String(importing));
    this.list.setAttribute('aria-busy', String(importing));
    this.cancelImportButton.hidden = !importing || this.activeImportController === null;
    this.cancelImportButton.disabled = !importing || this.activeImportController === null;
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
      const positioned = enroute.filter((record) => routePosition(record) !== null);
      const before = positioned.filter((record) => (routePosition(record) ?? 0) <= (splitAfterM ?? 0));
      const after = positioned.filter((record) => (routePosition(record) ?? 0) > (splitAfterM ?? 0));
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
      const groups = [before, after];
      const revealsOrder = complete && groups.some(lettersRevealRouteOrder);
      this.compliance.findings.push({
        photoId: null,
        severity: complete ? (revealsOrder ? 'violation' : 'pass') : 'warning',
        code: 'enroute-letter-order',
        rule: 'A2.4.5',
        affected: 'route',
        message: complete
          ? revealsOrder
            ? 'En-route photo letters disclose route order within at least one handout part.'
            : 'En-route photo letters are mixed independently within both handout parts.'
          : 'Letter mixing requires a valid split and reliable route positions.',
        measured: complete
          ? revealsOrder
            ? 'alphabetical route sequence detected'
            : 'mixed sequence in both route parts'
          : 'route sequence unavailable',
        permitted: 'lettered photos in non-route order within each of the two route parts',
      });
      const counts = operationalFindingCounts(this.compliance.findings);
      this.compliance.violationCount = counts.violationCount;
      this.compliance.warningCount = counts.warningCount;
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
      const primaryFindings = record.findings.filter(
        (finding) => finding.severity !== 'pass' && findingPresentation(finding) === 'primary'
      );
      const actionFindings = record.findings.filter(
        (finding) => finding.severity !== 'pass' && findingPresentation(finding) === 'action'
      );
      const auditFindings = record.findings.filter(
        (finding) => finding.severity !== 'pass' && findingPresentation(finding) === 'audit'
      );
      const hasViolation = primaryFindings.some((finding) => finding.severity === 'violation');
      const hasWarning = primaryFindings.some((finding) => finding.severity === 'warning');
      const hasAction = actionFindings.length > 0;
      if (!hasViolation && record.exceptionAccepted) {
        record.exceptionAccepted = false;
        record.exceptionAcceptedAt = null;
      }
      status.dataset.tone = hasViolation
        ? record.exceptionAccepted
          ? 'accepted'
          : 'fail'
        : hasAction || hasWarning
          ? 'review'
          : 'ok';
      status.textContent = hasViolation
        ? record.exceptionAccepted
          ? 'Accepted exception · still against the rules'
          : 'Against the rules'
        : hasAction
          ? 'Action required'
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
      const metadataDetails = document.createElement('details');
      metadataDetails.className = 'photo-metadata-details';
      const metadataSummary = document.createElement('summary');
      metadataSummary.className = 'photo-metadata-summary';
      metadataSummary.textContent = 'Camera metadata & source';
      const metadataGrid = document.createElement('div');
      metadataGrid.className = 'photo-field-grid';
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
        numericInput('Task/object latitude', 'taskLatitude', record.taskLatitude.value),
        numericInput('Task/object longitude', 'taskLongitude', record.taskLongitude.value)
      );
      metadataGrid.append(
        numericInput('Latitude', 'latitude', record.metadata.latitude.value),
        numericInput('Longitude', 'longitude', record.metadata.longitude.value),
        numericInput('Heading (°)', 'headingDeg', record.metadata.headingDeg.value),
        numericInput('Altitude AGL (ft)', 'altitudeAglFt', record.metadata.altitudeAglFt.value),
        numericInput('35 mm equiv. focal (mm)', 'focalLength35Mm', record.metadata.focalLength35Mm.value),
        input('Capture time', record.metadata.captureTime.value ?? '', 'captureTime', record.id)
      );
      const provenance = document.createElement('p');
      provenance.className = 'photo-provenance';
      const originalPosition = [
        record.originalMetadata.latitude.value,
        record.originalMetadata.longitude.value,
      ].every((value) => value !== null)
        ? `; original EXIF GPS: ${record.originalMetadata.latitude.value}, ${record.originalMetadata.longitude.value}`
        : '';
      const orthophotoProvenance = record.generatedOrthophoto
        ? `; GURS DOF025 crop ${record.generatedOrthophoto.coverageWidthM.toFixed(0)}×${record.generatedOrthophoto.coverageHeightM.toFixed(0)} m modeled at ${record.generatedOrthophoto.modeledAltitudeM} m AGL, ${record.generatedOrthophoto.modeledFocalLength35Mm} mm, ${record.generatedOrthophoto.modeledDepressionDeg}° depression`
        : '';
      const targetProvenance = record.generatedOrthophoto?.targetSource
        ? `; OSM target: ${record.generatedOrthophoto.targetSource.name ?? record.generatedOrthophoto.targetSource.featureType} (${record.generatedOrthophoto.targetSource.featureType}, score ${record.generatedOrthophoto.targetSource.score}${record.generatedOrthophoto.targetSource.selectionSalt ? `, selection mix ${record.generatedOrthophoto.targetSource.selectionSalt.slice(0, 8)}` : ''})`
        : '';
      provenance.textContent = `${record.width}×${record.height}px · Position: ${record.metadata.latitude.source}/${record.metadata.longitude.source}${originalPosition}; EXIF altitude: ${record.metadata.gpsAltitudeMslM.value ?? 'missing'} m MSL; camera: ${[record.metadata.cameraMake.value, record.metadata.cameraModel.value].filter(Boolean).join(' ') || 'missing'}; orientation: ${record.metadata.orientation.value ?? 'missing'}${orthophotoProvenance}${targetProvenance}.`;
      metadataDetails.append(metadataSummary, metadataGrid, provenance);
      const metrics = document.createElement('p');
      metrics.className = 'photo-metrics';
      metrics.textContent = record.analysis
        ? `${record.analysis.legId} · ${(record.analysis.alongRouteM / 1852).toFixed(2)} NM along route · ${record.analysis.lateralDistanceM.toFixed(0)} m lateral · ${(record.analysis.distanceAfterPreviousControlPointM / 1852).toFixed(2)} NM after ${record.analysis.previousControlPoint}`
        : 'Route metrics unavailable until both position and a valid route are present.';
      const issueList = document.createElement('ul');
      issueList.className = 'photo-card-findings';
      for (const finding of primaryFindings) {
        const item = document.createElement('li');
        item.textContent = `${finding.rule}: ${finding.measured}; permitted ${finding.permitted}.${record.exceptionAccepted && finding.severity === 'violation' ? ' Accepted as a judge exception; violation retained.' : ''}`;
        issueList.appendChild(item);
      }
      issueList.hidden = primaryFindings.length === 0;
      const actionList = document.createElement('ul');
      actionList.className = 'photo-card-actions';
      for (const finding of actionFindings) {
        const item = document.createElement('li');
        item.textContent =
          finding.code === 'task-position-missing'
            ? 'Mark the photographed task/object with coordinates or link a valid waypoint before generating the marked map.'
            : finding.message;
        actionList.appendChild(item);
      }
      actionList.hidden = actionFindings.length === 0;
      const auditDetails = document.createElement('details');
      auditDetails.className = 'photo-audit-details';
      auditDetails.hidden = auditFindings.length === 0;
      const auditSummary = document.createElement('summary');
      auditSummary.className = 'photo-audit-summary';
      auditSummary.textContent = `Technical & rulebook audit (${auditFindings.length})`;
      const auditList = document.createElement('ul');
      auditList.className = 'photo-audit-findings';
      for (const finding of auditFindings) {
        const item = document.createElement('li');
        item.textContent = `${finding.rule}: ${finding.message} ${finding.measured}; expected ${finding.permitted}.`;
        auditList.appendChild(item);
      }
      auditDetails.append(auditSummary, auditList);
      if (record.importError) {
        const error = document.createElement('p');
        error.className = 'danger-text';
        error.textContent = `Metadata error: ${record.importError}`;
        body.append(
          heading,
          status,
          exceptionButton,
          error,
          grid,
          metrics,
          issueList,
          actionList,
          auditDetails,
          metadataDetails
        );
      } else
        body.append(
          heading,
          status,
          exceptionButton,
          grid,
          metrics,
          issueList,
          actionList,
          auditDetails,
          metadataDetails
        );
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
    const operationalFindings = this.compliance.findings.filter(
      (finding) => finding.severity !== 'pass' && findingPresentation(finding) !== 'audit'
    );
    for (const finding of operationalFindings) {
      const item = document.createElement('li');
      const accepted =
        finding.photoId !== null &&
        finding.severity === 'violation' &&
        this.records.some((record) => record.id === finding.photoId && record.exceptionAccepted);
      item.textContent = `${finding.rule} · ${finding.affected}: ${finding.measured}; permitted ${finding.permitted}.${accepted ? ' Accepted as a judge exception; violation retained.' : ''}`;
      findingFragment.appendChild(item);
    }
    this.findings.replaceChildren(findingFragment);
    this.findings.hidden = operationalFindings.length === 0;
    const auditFindings = this.compliance.findings.filter(
      (finding) => finding.severity !== 'pass' && findingPresentation(finding) === 'audit'
    );
    const auditFragment = document.createDocumentFragment();
    for (const finding of auditFindings) {
      const item = document.createElement('li');
      item.textContent = `${finding.rule} · ${finding.affected}: ${finding.message} ${finding.measured}; expected ${finding.permitted}.`;
      auditFragment.appendChild(item);
    }
    this.auditFindings.replaceChildren(auditFragment);
    this.auditSummary.textContent = `Technical & rulebook audit details (${auditFindings.length})`;
    this.auditDetails.hidden = auditFindings.length === 0;
  }
}
