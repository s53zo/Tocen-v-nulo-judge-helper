import { describe, expect, it } from 'vitest';
import { buildRoute, type Waypoint } from '../src/domain';
import {
  analyzePhotoPosition,
  distanceBetweenSubjects,
  headingDifference,
  normalizeLongitudeDelta,
} from '../src/photo-analysis';

describe('photo route analysis', () => {
  const points: Waypoint[] = [
    ['SP', 0, 0],
    ['TP1', 0, 0.1],
    ['FP', 0.1, 0.1],
  ];
  const route = buildRoute(points);

  it('projects onto the closest leg and calculates signed lateral and heading differences', () => {
    const result = analyzePhotoPosition(0.01, 0.05, 90, route, points);
    expect(result.legId).toBe('SP-TP1');
    expect(result.fraction).toBeCloseTo(0.5, 2);
    expect(result.lateralDistanceM).toBeCloseTo(1112, -1);
    expect(result.lateralSignedM).toBeGreaterThan(0);
    expect(result.headingDifferenceDeg).toBeCloseTo(0, 4);
    expect(result.previousControlPoint).toBe('SP');
  });

  it('identifies positions before and after the route', () => {
    expect(analyzePhotoPosition(0, -0.01, null, route, points).routePosition).toBe('before-route');
    expect(analyzePhotoPosition(0.11, 0.1, null, route, points).routePosition).toBe('after-route');
  });

  it('handles circular angles and antimeridian longitude deltas', () => {
    expect(headingDifference(350, 10)).toBe(20);
    expect(normalizeLongitudeDelta(358)).toBe(-2);
    const datelinePoints: Waypoint[] = [
      ['SP', 10, 179.9],
      ['FP', 10, -179.9],
    ];
    const result = analyzePhotoPosition(10.01, 180, null, buildRoute(datelinePoints), datelinePoints);
    expect(result.fraction).toBeCloseTo(0.5, 1);
  });

  it('measures false-object separation geodesically', () => {
    expect(
      distanceBetweenSubjects({ latitude: 0, longitude: 0 }, { latitude: 0, longitude: 0.01666 })
    ).toBeCloseTo(1852, -1);
  });

  it('detects a shared turn vertex and conservatively chooses the outgoing leg', () => {
    const result = analyzePhotoPosition(0, 0.1, null, route, points);
    expect(result.ambiguousLegIndices).toEqual([0, 1]);
    expect(result.legIndex).toBe(1);
    expect(result.previousControlPoint).toBe('TP1');
    expect(result.distanceAfterPreviousControlPointM).toBeCloseTo(0, 6);
  });

  it('honors an explicit manual leg selection at an ambiguous crossing', () => {
    const crossing: Waypoint[] = [
      ['SP', -0.1, -0.1],
      ['TP1', 0.1, 0.1],
      ['TP2', -0.1, 0.1],
      ['FP', 0.1, -0.1],
    ];
    const result = analyzePhotoPosition(0, 0, null, buildRoute(crossing), crossing, 2);
    expect(result.legIndex).toBe(2);
    expect(result.manuallySelectedLeg).toBe(true);
    expect(result.ambiguousLegIndices).toContain(0);
    expect(result.ambiguousLegIndices).toContain(2);
  });

  it('reports ambiguity between closely parallel legs', () => {
    const parallel: Waypoint[] = [
      ['SP', 0, 0],
      ['TP1', 0, 0.1],
      ['TP2', 0.00001, 0.1],
      ['FP', 0.00001, 0],
    ];
    const result = analyzePhotoPosition(0.000005, 0.05, null, buildRoute(parallel), parallel);
    expect(result.ambiguousLegIndices).toContain(0);
    expect(result.ambiguousLegIndices).toContain(2);
  });
});
