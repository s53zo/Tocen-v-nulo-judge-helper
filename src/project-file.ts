import type { SavedProjectPhoto } from './photo-workflow';

export const PROJECT_FILE_SCHEMA_VERSION = 2;
export const PROJECT_FILE_FORMAT = 'tocen-v-nulo-route-project';
export const SPEED_EDITION_KNOTS = Array.from({ length: 11 }, (_, index) => 50 + index * 5);
export const STANDARD_SPEED_PRESETS = SPEED_EDITION_KNOTS.map((speed) => `${speed}kt`);

export type CustomSpeedUnit = 'kt' | 'kmh';

export interface SavedCustomSpeed {
  value: string;
  unit: CustomSpeedUnit;
}

export const DEFAULT_CUSTOM_SPEEDS: SavedCustomSpeed[] = Array.from({ length: 4 }, () => ({
  value: '',
  unit: 'kt',
}));

export interface SavedProjectSettings {
  waypoints: string;
  speed: string;
  speedPreset: string;
  customSpeeds: SavedCustomSpeed[];
  takeoffBuffer: string;
  minuteInterval: string;
  mapKey: string;
  handoutSplit: string;
  orthophotoRandomCount: string;
  orthophotoAltitude: string;
  orthophotoFocalLength: string;
  orthophotoDepression: string;
  styleRouteWidth: string;
  styleWaypointFont: string;
  styleHeadingFont: string;
  styleMinuteLabelFont: string;
  styleMinuteMarkerSize: string;
  styleMinuteLineWidth: string;
  photoExactDots: boolean;
  photoProjectedMarkers: boolean;
  photoHeadingArrows: boolean;
  photoConnectors: boolean;
  photoLegend: boolean;
  photoCropBounds: boolean;
  handoutSummary: boolean;
}

interface ProjectEnvelope {
  format?: string;
  schemaVersion?: unknown;
  appVersion?: unknown;
  savedAt?: unknown;
  settings?: unknown;
  photos?: unknown;
}

export interface SavedRouteProject {
  format: typeof PROJECT_FILE_FORMAT;
  schemaVersion: 2;
  appVersion: string;
  savedAt: string;
  settings: SavedProjectSettings;
  photos: SavedProjectPhoto[];
}

const requiredStrings = [
  'waypoints',
  'speed',
  'takeoffBuffer',
  'minuteInterval',
  'mapKey',
  'handoutSplit',
  'orthophotoRandomCount',
  'orthophotoAltitude',
  'orthophotoFocalLength',
  'orthophotoDepression',
  'styleRouteWidth',
  'styleWaypointFont',
  'styleHeadingFont',
  'styleMinuteLabelFont',
  'styleMinuteMarkerSize',
  'styleMinuteLineWidth',
] as const;

const requiredBooleans = [
  'photoExactDots',
  'photoProjectedMarkers',
  'photoHeadingArrows',
  'photoConnectors',
  'photoLegend',
  'photoCropBounds',
  'handoutSummary',
] as const;

function validateEnvelope(value: Record<string, unknown>): {
  envelope: ProjectEnvelope;
  settings: Record<string, unknown>;
  photos: SavedProjectPhoto[];
} {
  const envelope = value as ProjectEnvelope;
  if (envelope.format !== undefined && envelope.format !== PROJECT_FILE_FORMAT) {
    throw new Error('The selected file is not a Route Overlay project.');
  }
  if (typeof envelope.appVersion !== 'string' || typeof envelope.savedAt !== 'string') {
    throw new Error('The selected project file is missing its version or save time.');
  }
  if (!Number.isFinite(Date.parse(envelope.savedAt))) {
    throw new Error('The selected project file has an invalid save time.');
  }
  if (
    !envelope.settings ||
    typeof envelope.settings !== 'object' ||
    Array.isArray(envelope.settings) ||
    !Array.isArray(envelope.photos)
  ) {
    throw new Error('The selected project file is missing settings or photos.');
  }
  const settings = envelope.settings as Record<string, unknown>;
  for (const key of requiredStrings) {
    if (typeof settings[key] !== 'string') {
      throw new Error(`The selected project file has an invalid ${key} setting.`);
    }
  }
  for (const key of requiredBooleans) {
    if (typeof settings[key] !== 'boolean') {
      throw new Error(`The selected project file has an invalid ${key} setting.`);
    }
  }
  return { envelope, settings, photos: envelope.photos as SavedProjectPhoto[] };
}

function customPresetIndex(preset: string): number | null {
  const match = preset.match(/^custom-([1-4])$/);
  return match ? Number(match[1]) - 1 : null;
}

function migrateLegacySpeed(speed: string): Pick<SavedProjectSettings, 'speedPreset' | 'customSpeeds'> {
  const compact = speed.trim().toLowerCase().replace(/\s+/g, '');
  if (STANDARD_SPEED_PRESETS.includes(compact)) {
    return { speedPreset: compact, customSpeeds: DEFAULT_CUSTOM_SPEEDS.map((entry) => ({ ...entry })) };
  }
  const match = compact.match(/^(\d+(?:[.,]\d+)?)(kt|kts|mph|km\/h|kmh|kph)?$/);
  const customSpeeds = DEFAULT_CUSTOM_SPEEDS.map((entry) => ({ ...entry }));
  if (match) {
    const sourceValue = Number(match[1].replace(',', '.'));
    const sourceUnit = match[2] ?? 'kt';
    const unit: CustomSpeedUnit = sourceUnit.startsWith('km') || sourceUnit === 'kph' ? 'kmh' : 'kt';
    const convertedValue = sourceUnit === 'mph' ? sourceValue * 0.8689762419 : sourceValue;
    const value = convertedValue.toFixed(3).replace(/\.?0+$/, '');
    const standardPreset = `${value}kt`;
    if (unit === 'kt' && STANDARD_SPEED_PRESETS.includes(standardPreset)) {
      return { speedPreset: standardPreset, customSpeeds };
    }
    customSpeeds[0] = {
      value,
      unit,
    };
    return { speedPreset: 'custom-1', customSpeeds };
  }
  return { speedPreset: '75kt', customSpeeds };
}

function validateVersionTwoSpeedSettings(settings: Record<string, unknown>): void {
  if (typeof settings.speedPreset !== 'string') {
    throw new Error('The selected project file has an invalid speedPreset setting.');
  }
  const customIndex = customPresetIndex(settings.speedPreset);
  if (!STANDARD_SPEED_PRESETS.includes(settings.speedPreset) && customIndex === null) {
    throw new Error('The selected project file has an unknown speed preset.');
  }
  if (!Array.isArray(settings.customSpeeds) || settings.customSpeeds.length !== 4) {
    throw new Error('The selected project file must contain four custom speed settings.');
  }
  for (const [index, entry] of settings.customSpeeds.entries()) {
    if (
      !entry ||
      typeof entry !== 'object' ||
      typeof entry.value !== 'string' ||
      (entry.unit !== 'kt' && entry.unit !== 'kmh')
    ) {
      throw new Error(`The selected project file has an invalid custom speed ${index + 1}.`);
    }
    if (entry.value.trim() && (!Number.isFinite(Number(entry.value)) || Number(entry.value) <= 0)) {
      throw new Error(`The selected project file has an invalid custom speed ${index + 1} value.`);
    }
  }
}

function parseVersionOne(value: Record<string, unknown>): SavedRouteProject {
  const { envelope, settings, photos } = validateEnvelope(value);
  return {
    format: PROJECT_FILE_FORMAT,
    schemaVersion: PROJECT_FILE_SCHEMA_VERSION,
    appVersion: envelope.appVersion as string,
    savedAt: envelope.savedAt as string,
    settings: {
      ...(settings as unknown as Omit<SavedProjectSettings, 'speedPreset' | 'customSpeeds'>),
      ...migrateLegacySpeed(settings.speed as string),
    },
    photos,
  };
}

function parseVersionTwo(value: Record<string, unknown>): SavedRouteProject {
  const { envelope, settings, photos } = validateEnvelope(value);
  validateVersionTwoSpeedSettings(settings);
  return {
    format: PROJECT_FILE_FORMAT,
    schemaVersion: PROJECT_FILE_SCHEMA_VERSION,
    appVersion: envelope.appVersion as string,
    savedAt: envelope.savedAt as string,
    settings: settings as unknown as SavedProjectSettings,
    photos,
  };
}

export function parseSavedRouteProject(raw: string): SavedRouteProject {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    throw new Error('The selected project file is not valid JSON.');
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('The selected project file is malformed.');
  }
  const envelope = value as Record<string, unknown>;
  switch (envelope.schemaVersion) {
    // Keep each decoder when adding a newer schema so old organizer projects remain loadable.
    case 1:
      return parseVersionOne(envelope);
    case 2:
      return parseVersionTwo(envelope);
    default:
      throw new Error(`Unsupported project schema version: ${String(envelope.schemaVersion ?? 'missing')}.`);
  }
}
