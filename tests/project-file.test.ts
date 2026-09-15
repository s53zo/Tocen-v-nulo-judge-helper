import { describe, expect, it } from 'vitest';
import { base64ToBytes, bytesToBase64 } from '../src/photo-workflow';
import {
  PROJECT_FILE_FORMAT,
  PROJECT_FILE_SCHEMA_VERSION,
  parseSavedRouteProject,
  SPEED_EDITION_KNOTS,
} from '../src/project-file';

const settings = {
  waypoints: 'SP,46.6,16\nFP,46.7,16.1',
  speed: '75kt',
  speedPreset: '75kt',
  customSpeeds: [
    { value: '', unit: 'kt' },
    { value: '', unit: 'kt' },
    { value: '', unit: 'kt' },
    { value: '', unit: 'kt' },
  ],
  takeoffBuffer: '4',
  minuteInterval: '1',
  mapKey: 'vfr',
  handoutSplit: 'TP1',
  orthophotoRandomCount: '12',
  orthophotoAltitude: '100',
  orthophotoFocalLength: '50',
  orthophotoDepression: '90',
  styleRouteWidth: '2.5',
  styleWaypointFont: '12',
  styleHeadingFont: '18',
  styleMinuteLabelFont: '10',
  styleMinuteMarkerSize: '7',
  styleMinuteLineWidth: '1.2',
  photoExactDots: false,
  photoProjectedMarkers: true,
  photoHeadingArrows: false,
  photoConnectors: false,
  photoLegend: false,
  photoCropBounds: true,
  handoutSummary: true,
};

describe('saved route projects', () => {
  it('defines all 11 speed editions from 50 through 100 kt', () => {
    expect(SPEED_EDITION_KNOTS).toEqual([50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]);
  });

  it('round-trips arbitrary embedded photo bytes', () => {
    const bytes = Uint8Array.from([0, 1, 2, 127, 128, 254, 255]);
    expect(base64ToBytes(bytesToBase64(bytes))).toEqual(bytes);
  });

  it('accepts the current schema and rejects unknown versions', () => {
    const project = {
      format: PROJECT_FILE_FORMAT,
      schemaVersion: PROJECT_FILE_SCHEMA_VERSION,
      appVersion: 'test',
      savedAt: '2026-01-01T00:00:00.000Z',
      settings,
      photos: [],
    };
    expect(parseSavedRouteProject(JSON.stringify(project))).toEqual(project);
    expect(() => parseSavedRouteProject(JSON.stringify({ ...project, schemaVersion: 999 }))).toThrow(
      /Unsupported project schema version/
    );
    expect(() => parseSavedRouteProject(JSON.stringify({ ...project, savedAt: 'not-a-date' }))).toThrow(
      /invalid save time/
    );
  });

  it('migrates version 1 projects and retains a legacy custom speed', () => {
    const { speedPreset: _speedPreset, customSpeeds: _customSpeeds, ...legacySettings } = settings;
    const legacyProject = {
      schemaVersion: 1,
      appVersion: '2.4.0',
      savedAt: '2026-01-01T00:00:00.000Z',
      settings: { ...legacySettings, speed: '140kmh' },
      photos: [],
    };
    expect(parseSavedRouteProject(JSON.stringify(legacyProject))).toEqual({
      ...legacyProject,
      format: PROJECT_FILE_FORMAT,
      schemaVersion: PROJECT_FILE_SCHEMA_VERSION,
      settings: {
        ...legacyProject.settings,
        speedPreset: 'custom-1',
        customSpeeds: [
          { value: '140', unit: 'kmh' },
          { value: '', unit: 'kt' },
          { value: '', unit: 'kt' },
          { value: '', unit: 'kt' },
        ],
      },
    });
  });
});
