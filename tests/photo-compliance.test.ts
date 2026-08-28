import { describe, expect, it } from 'vitest';
import type { Waypoint } from '../src/domain';
import { evaluatePhotoCompliance } from '../src/photo-compliance';
import { missingValue, type PhotoMetadata, type PhotoRecord, sourcedValue } from '../src/photo-types';

const points: Waypoint[] = [
  ['SP', 0, 0],
  ['TP1', 0, 0.1],
  ['FP', 0, 0.2],
];

function metadata(overrides: Partial<{ agl: number; focal: number; heading: number }> = {}): PhotoMetadata {
  return {
    latitude: sourcedValue(0, 'manual'),
    longitude: sourcedValue(0.02, 'manual'),
    gpsAltitudeMslM: missingValue(),
    altitudeAglFt: overrides.agl === undefined ? missingValue() : sourcedValue(overrides.agl, 'manual'),
    headingDeg: overrides.heading === undefined ? missingValue() : sourcedValue(overrides.heading, 'manual'),
    headingReference:
      overrides.heading === undefined ? missingValue() : sourcedValue('true' as const, 'manual'),
    focalLengthMm: missingValue(),
    focalLength35Mm: overrides.focal === undefined ? missingValue() : sourcedValue(overrides.focal, 'manual'),
    captureTime: missingValue(),
    orientation: sourcedValue(1, 'exif'),
    cameraMake: missingValue(),
    cameraModel: missingValue(),
    lensModel: missingValue(),
  };
}

function photo(index: number, overrides: Partial<PhotoRecord> = {}): PhotoRecord {
  return {
    id: String(index),
    file: {} as File,
    fileName: `p${index}.jpg`,
    fileSize: 1,
    contentHash: String(index),
    previewUrl: '',
    width: 1,
    height: 1,
    originalMetadata: metadata({ agl: 750, focal: 55, heading: 90 }),
    metadata: metadata({ agl: 750, focal: 55, heading: 90 }),
    classification: 'enroute',
    identifier: String.fromCharCode(65 + index),
    linkedWaypoint: null,
    taskLatitude: sourcedValue(0, 'manual'),
    taskLongitude: sourcedValue(0.02, 'manual'),
    manualLegIndex: null,
    order: index,
    importError: null,
    analysis: {
      legIndex: 0,
      legId: 'SP-TP1',
      rawFraction: 0.2,
      fraction: 0.2,
      closestLatitude: 0,
      closestLongitude: 0.02,
      lateralDistanceM: 200,
      lateralSignedM: 200,
      alongRouteM: 2200,
      distanceOnLegM: 2200,
      routePosition: 'on-route',
      nearestControlPoint: 'SP',
      distanceFromNearestControlPointM: 2200,
      previousControlPoint: 'SP',
      distanceAfterPreviousControlPointM: 2200,
      legBearingDeg: 90,
      headingDifferenceDeg: 0,
      ambiguousLegIndices: [0],
      manuallySelectedLeg: false,
    },
    taskAnalysis: {
      legIndex: 0,
      legId: 'SP-TP1',
      rawFraction: 0.2,
      fraction: 0.2,
      closestLatitude: 0,
      closestLongitude: 0.02,
      lateralDistanceM: 0,
      lateralSignedM: 0,
      alongRouteM: 2200,
      distanceOnLegM: 2200,
      routePosition: 'on-route',
      nearestControlPoint: 'SP',
      distanceFromNearestControlPointM: 2200,
      previousControlPoint: 'SP',
      distanceAfterPreviousControlPointM: 2200,
      legBearingDeg: 90,
      headingDifferenceDeg: null,
      ambiguousLegIndices: [0],
      manuallySelectedLeg: false,
    },
    findings: [],
    exceptionAccepted: false,
    exceptionAcceptedAt: null,
    isExample: false,
    ...overrides,
  };
}

describe('photo compliance', () => {
  it('enforces 12 en-route photos and 15 route tasks', () => {
    expect(
      evaluatePhotoCompliance(
        Array.from({ length: 12 }, (_, index) => photo(index)),
        points
      ).violationCount
    ).toBe(0);
    expect(
      evaluatePhotoCompliance(
        Array.from({ length: 13 }, (_, index) => photo(index)),
        points
      ).findings.find((item) => item.code === 'enroute-photo-count')?.severity
    ).toBe('violation');
    const tasks = Array.from({ length: 16 }, (_, index) => photo(index, { classification: 'sign-task' }));
    expect(
      evaluatePhotoCompliance(tasks, points).findings.find((item) => item.code === 'route-task-count')
        ?.severity
    ).toBe('violation');
  });

  it('excludes turning-point signs from the route-task count', () => {
    const tasks = Array.from({ length: 20 }, (_, index) =>
      photo(index, { classification: 'sign-task', linkedWaypoint: 'TP1' })
    );
    expect(evaluatePhotoCompliance(tasks, points).routeTaskCount).toBe(0);
  });

  it('counts control photos at SP/FP and stale TP links as route tasks', () => {
    const startControls = Array.from({ length: 16 }, (_, index) =>
      photo(index, { classification: 'control-correct', linkedWaypoint: 'SP' })
    );
    expect(evaluatePhotoCompliance(startControls, points).routeTaskCount).toBe(16);
    const staleSigns = Array.from({ length: 16 }, (_, index) =>
      photo(index, { classification: 'sign-task', linkedWaypoint: 'TP99' })
    );
    const result = evaluatePhotoCompliance(staleSigns, points);
    expect(result.routeTaskCount).toBe(16);
    expect(result.findings.some((item) => item.code === 'stale-waypoint-link')).toBe(true);
  });

  it('checks 1 NM post-control spacing at its boundary', () => {
    const baseAnalysis = photo(1).taskAnalysis;
    if (!baseAnalysis) throw new Error('Test fixture analysis is missing.');
    const passing = photo(1, {
      taskAnalysis: { ...baseAnalysis, distanceAfterPreviousControlPointM: 1852 },
    });
    const failing = photo(2, {
      taskAnalysis: { ...baseAnalysis, distanceAfterPreviousControlPointM: 1851.9 },
    });
    expect(
      evaluatePhotoCompliance([passing], points).findings.find((item) => item.code === 'post-control-spacing')
        ?.severity
    ).toBe('pass');
    expect(
      evaluatePhotoCompliance([failing], points).findings.find((item) => item.code === 'post-control-spacing')
        ?.severity
    ).toBe('violation');
  });

  it('enforces lateral, heading, AGL and focal limits and uses manual review when unknown', () => {
    const baseAnalysis = photo(1).analysis;
    if (!baseAnalysis) throw new Error('Test fixture analysis is missing.');
    const invalid = photo(1, {
      metadata: metadata({ agl: 499, focal: 71, heading: 180 }),
      analysis: { ...baseAnalysis, lateralDistanceM: 301, headingDifferenceDeg: 90 },
    });
    const result = evaluatePhotoCompliance([invalid], points);
    for (const code of ['route-axis-distance', 'camera-angle', 'capture-altitude', 'focal-length']) {
      expect(result.findings.find((item) => item.code === code)?.severity).toBe('violation');
    }
    const unknown = photo(2, { metadata: metadata(), analysis: null });
    expect(evaluatePhotoCompliance([unknown], points).status).toBe('manual-review');
  });

  it('accepts inclusive measurement boundaries', () => {
    const baseAnalysis = photo(1).analysis;
    if (!baseAnalysis) throw new Error('Test fixture analysis is missing.');
    const lowBoundary = photo(1, {
      metadata: metadata({ agl: 500, focal: 50, heading: 135 }),
      analysis: { ...baseAnalysis, lateralDistanceM: 300, headingDifferenceDeg: 45 },
    });
    const highBoundary = photo(2, { metadata: metadata({ agl: 1000, focal: 70, heading: 90 }) });
    expect(evaluatePhotoCompliance([lowBoundary, highBoundary], points).violationCount).toBe(0);
  });

  it('enforces the 100 m sign-task route-axis boundary', () => {
    const baseTaskAnalysis = photo(1).taskAnalysis;
    if (!baseTaskAnalysis) throw new Error('Test fixture task analysis is missing.');
    const boundary = photo(1, {
      classification: 'sign-task',
      linkedWaypoint: 'TP1',
      taskAnalysis: { ...baseTaskAnalysis, lateralDistanceM: 100 },
    });
    const outside = photo(2, {
      classification: 'sign-task',
      linkedWaypoint: 'TP1',
      taskAnalysis: { ...baseTaskAnalysis, lateralDistanceM: 100.1 },
    });
    expect(
      evaluatePhotoCompliance([boundary], points).findings.find(
        (item) => item.code === 'sign-route-axis-distance'
      )?.severity
    ).toBe('pass');
    expect(
      evaluatePhotoCompliance([outside], points).findings.find(
        (item) => item.code === 'sign-route-axis-distance'
      )?.severity
    ).toBe('violation');
  });

  it('requires manual review for magnetic camera headings', () => {
    const magnetic = photo(1);
    magnetic.metadata.headingReference = sourcedValue('magnetic', 'exif', false);
    expect(
      evaluatePhotoCompliance([magnetic], points).findings.find(
        (item) => item.code === 'camera-angle-missing'
      )?.severity
    ).toBe('warning');
  });

  it('checks false control-object separation when subject coordinates are known', () => {
    const tooClose = photo(1, {
      classification: 'control-false',
      linkedWaypoint: 'TP1',
      taskLatitude: sourcedValue(0, 'manual'),
      taskLongitude: sourcedValue(0.1001, 'manual'),
    });
    const far = photo(2, {
      classification: 'control-false',
      linkedWaypoint: 'TP1',
      taskLatitude: sourcedValue(0, 'manual'),
      taskLongitude: sourcedValue(0.12, 'manual'),
    });
    expect(
      evaluatePhotoCompliance([tooClose], points).findings.find(
        (item) => item.code === 'false-control-distance'
      )?.severity
    ).toBe('violation');
    expect(
      evaluatePhotoCompliance([far], points).findings.find((item) => item.code === 'false-control-distance')
        ?.severity
    ).toBe('pass');
  });
});
