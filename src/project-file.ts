import type { SavedProjectPhoto } from './photo-workflow';

export const PROJECT_FILE_SCHEMA_VERSION = 1;
export const SPEED_EDITION_KNOTS = Array.from({ length: 11 }, (_, index) => 50 + index * 5);

export interface SavedProjectSettings {
  waypoints: string;
  speed: string;
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

export interface SavedRouteProject {
  schemaVersion: 1;
  appVersion: string;
  savedAt: string;
  settings: SavedProjectSettings;
  photos: SavedProjectPhoto[];
}

export function parseSavedRouteProject(raw: string): SavedRouteProject {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    throw new Error('The selected project file is not valid JSON.');
  }
  if (!value || typeof value !== 'object') throw new Error('The selected project file is malformed.');
  const project = value as Partial<SavedRouteProject>;
  if (project.schemaVersion !== PROJECT_FILE_SCHEMA_VERSION) {
    throw new Error(`Unsupported project schema version: ${String(project.schemaVersion ?? 'missing')}.`);
  }
  if (typeof project.appVersion !== 'string' || typeof project.savedAt !== 'string') {
    throw new Error('The selected project file is missing its version or save time.');
  }
  if (!Number.isFinite(Date.parse(project.savedAt))) {
    throw new Error('The selected project file has an invalid save time.');
  }
  if (!project.settings || typeof project.settings !== 'object' || !Array.isArray(project.photos)) {
    throw new Error('The selected project file is missing settings or photos.');
  }
  const requiredStrings: Array<keyof SavedProjectSettings> = [
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
  ];
  for (const key of requiredStrings) {
    if (typeof project.settings[key] !== 'string') {
      throw new Error(`The selected project file has an invalid ${key} setting.`);
    }
  }
  const requiredBooleans: Array<keyof SavedProjectSettings> = [
    'photoExactDots',
    'photoProjectedMarkers',
    'photoHeadingArrows',
    'photoConnectors',
    'photoLegend',
    'photoCropBounds',
    'handoutSummary',
  ];
  for (const key of requiredBooleans) {
    if (typeof project.settings[key] !== 'boolean') {
      throw new Error(`The selected project file has an invalid ${key} setting.`);
    }
  }
  return project as SavedRouteProject;
}
