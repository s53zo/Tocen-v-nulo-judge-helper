import ExifReader from 'exifreader';
import {
  missingValue,
  type PhotoClassification,
  type PhotoMetadata,
  type SourcedValue,
  sourcedValue,
} from './photo-types';

type Tag = { value?: unknown; description?: unknown } | undefined;
type TagGroup = Record<string, Tag>;

function finiteNumber(value: unknown): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (Array.isArray(value) && value.length === 2 && value.every((part) => typeof part === 'number')) {
    const denominator = Number(value[1]);
    return denominator ? Number(value[0]) / denominator : null;
  }
  if (Array.isArray(value) && value.length === 1) return finiteNumber(value[0]);
  if (typeof value === 'string') {
    const match = value.replace(',', '.').match(/-?\d+(?:\.\d+)?/);
    if (match) {
      const number = Number(match[0]);
      return Number.isFinite(number) ? number : null;
    }
  }
  return null;
}

function tagNumber(...tags: Tag[]): number | null {
  for (const tag of tags) {
    const value = finiteNumber(tag?.value) ?? finiteNumber(tag?.description);
    if (value !== null) return value;
  }
  return null;
}

function tagText(...tags: Tag[]): string | null {
  for (const tag of tags) {
    const candidate = tag?.description ?? tag?.value;
    if (typeof candidate === 'string' && candidate.trim()) return candidate.trim();
    if (Array.isArray(candidate) && candidate.length === 1 && typeof candidate[0] === 'string') {
      return candidate[0].trim() || null;
    }
  }
  return null;
}

export function normalizeExifDate(value: string | null, offset: string | null): SourcedValue<string> {
  if (!value) return missingValue('Capture time is missing from EXIF.');
  const match = value.match(/^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})/);
  if (!match) return sourcedValue(value, 'exif', false, 'Unrecognized EXIF date format.');
  const [, year, month, day, hour, minute, second] = match;
  const normalizedOffset = offset?.match(/^[+-]\d{2}:\d{2}$/)?.[0] ?? '';
  return sourcedValue(
    `${year}-${month}-${day}T${hour}:${minute}:${second}${normalizedOffset}`,
    'exif',
    Boolean(normalizedOffset),
    normalizedOffset ? undefined : 'EXIF capture time has no timezone offset.'
  );
}

export function normalizeExifTags(raw: unknown): PhotoMetadata {
  const groups = (raw ?? {}) as Record<string, unknown>;
  const exif = (groups.exif ?? {}) as TagGroup;
  const file = (groups.file ?? {}) as TagGroup;
  const gps = (groups.gps ?? {}) as Record<string, unknown>;
  const latitude = finiteNumber(gps.Latitude);
  const longitude = finiteNumber(gps.Longitude);
  const altitude = finiteNumber(gps.Altitude);
  const heading = tagNumber(exif.GPSImgDirection, exif.CameraDirection);
  const focal = tagNumber(exif.FocalLength);
  const focal35 = tagNumber(exif.FocalLengthIn35mmFilm, exif.FocalLengthIn35mmFormat);
  const orientation = tagNumber(exif.Orientation, file.Orientation) ?? 1;
  const date = tagText(exif.DateTimeOriginal, exif.CreateDate, file['File Modified Date']);
  const offset = tagText(exif.OffsetTimeOriginal, exif.OffsetTime);

  return {
    latitude: sourcedValue(
      latitude,
      'exif',
      latitude !== null,
      latitude === null ? 'GPS latitude is missing.' : undefined
    ),
    longitude: sourcedValue(
      longitude,
      'exif',
      longitude !== null,
      longitude === null ? 'GPS longitude is missing.' : undefined
    ),
    gpsAltitudeMslM: sourcedValue(
      altitude,
      'exif',
      altitude !== null,
      altitude === null ? 'GPS altitude is missing.' : 'EXIF GPS altitude is MSL, not AGL.'
    ),
    altitudeAglFt: missingValue('AGL cannot be inferred safely from EXIF MSL altitude.'),
    headingDeg: sourcedValue(
      heading === null ? null : ((heading % 360) + 360) % 360,
      'exif',
      heading !== null,
      heading === null ? 'Camera direction is missing.' : undefined
    ),
    focalLengthMm: sourcedValue(
      focal,
      'exif',
      focal !== null,
      focal === null ? 'Focal length is missing.' : undefined
    ),
    focalLength35Mm: sourcedValue(
      focal35,
      'exif',
      focal35 !== null,
      focal35 === null ? '35 mm-equivalent focal length is missing.' : undefined
    ),
    captureTime: normalizeExifDate(date, offset),
    orientation: sourcedValue(orientation, 'exif', orientation >= 1 && orientation <= 8),
    cameraMake: sourcedValue(tagText(exif.Make), 'exif'),
    cameraModel: sourcedValue(tagText(exif.Model), 'exif'),
    lensModel: sourcedValue(tagText(exif.LensModel), 'exif'),
  };
}

export async function extractPhotoMetadata(file: File): Promise<PhotoMetadata> {
  const buffer = await file.arrayBuffer();
  const tags = await ExifReader.load(buffer, { expanded: true, async: true });
  return normalizeExifTags(tags);
}

export function applyManualValue<T>(field: SourcedValue<T>, value: T | null): SourcedValue<T> {
  if (value === null) return { ...field, value: null, source: 'missing', reliable: false };
  return sourcedValue(value, 'manual', true);
}

export function inferClassification(fileName: string): {
  classification: PhotoClassification;
  linkedWaypoint: string | null;
} {
  const tokens = fileName
    .normalize('NFKD')
    .replace(/[^A-Za-z0-9]+/g, ' ')
    .toUpperCase()
    .split(' ')
    .filter(Boolean);
  const linkedWaypoint =
    tokens.find((token) => token === 'SP' || token === 'FP' || /^TP\d+$/.test(token)) ?? null;
  return { classification: linkedWaypoint ? 'control-correct' : 'enroute', linkedWaypoint };
}

export function markRepeatedGps(metadata: PhotoMetadata[]): PhotoMetadata[] {
  const groups = new Map<string, number[]>();
  metadata.forEach((item, index) => {
    if (item.latitude.value === null || item.longitude.value === null) return;
    const key = `${item.latitude.value.toFixed(6)},${item.longitude.value.toFixed(6)}`;
    groups.set(key, [...(groups.get(key) ?? []), index]);
  });
  const result = metadata.map((item) => structuredClone(item));
  for (const [coordinate, indexes] of groups) {
    if (indexes.length < 3) continue;
    const note = `Identical EXIF GPS (${coordinate}) appears in ${indexes.length} photos; verify that it is not stale.`;
    for (const index of indexes) {
      result[index].latitude.reliable = false;
      result[index].longitude.reliable = false;
      result[index].latitude.note = note;
      result[index].longitude.note = note;
    }
  }
  return result;
}

export function orientationOutputSize(width: number, height: number, orientation: number): [number, number] {
  return orientation >= 5 && orientation <= 8 ? [height, width] : [width, height];
}
