import { describe, expect, it } from 'vitest';
import {
  bearingDegrees,
  buildRoute,
  computeMinuteMarkers,
  computeWaypointTimes,
  parseSpeed,
  roundedBearing,
  type Waypoint,
} from '../src/domain';

const routePoints: Waypoint[] = [
  ['SP', 46.60085, 16.180023],
  ['TP1', 46.5008, 16.155215],
  ['TP2', 46.515128, 16.008369],
];

describe('speed parsing', () => {
  it.each([
    ['75', '75 kt'],
    ['75 kts', '75 kt'],
    ['85mph', '85 mph'],
    ['140km/h', '140 km/h'],
    ['140kph', '140 km/h'],
  ])('parses %s', (input, label) => {
    expect(parseSpeed(input).label).toBe(label);
  });

  it('rejects unknown units and trailing text', () => {
    expect(() => parseSpeed('75foo')).toThrow(/kt, mph, or km\/h/);
  });
});

describe('route calculations', () => {
  it('builds deterministic legs and timing', () => {
    const route = buildRoute(routePoints);
    expect(route.legs).toHaveLength(2);
    expect(route.totalDistance).toBeCloseTo(22_636, -1);

    const speed = parseSpeed('75kt');
    const times = computeWaypointTimes(route, routePoints, 4, speed.metersPerSecond * 60);
    expect(times.get('SP')).toBe(4);
    expect(times.get('TP2')).toBeGreaterThan(times.get('TP1') ?? 0);
  });

  it('places markers on route legs', () => {
    const route = buildRoute(routePoints);
    const markers = computeMinuteMarkers(route, 4, parseSpeed('75kt').metersPerSecond * 60, 1);
    expect(markers.length).toBeGreaterThan(5);
    expect(markers.every((marker) => marker.ratio >= 0 && marker.ratio <= 1)).toBe(true);
  });

  it('normalizes a rounded north bearing to 000', () => {
    expect(roundedBearing(359.7)).toBe(0);
    expect(roundedBearing(bearingDegrees(46, 15, 47, 15))).toBe(0);
  });

  it('rejects zero-length legs', () => {
    expect(() => buildRoute([routePoints[0], routePoints[0]])).toThrow(/same position/);
  });
});
