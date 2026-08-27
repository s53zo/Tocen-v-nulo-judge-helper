import { describe, expect, it } from 'vitest';
import {
  bearingDegrees,
  buildRoute,
  computeMinuteMarkers,
  computeWaypointTimes,
  evaluateRouteCompliance,
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

describe('Slovenian rally route compliance', () => {
  const compliantPoints: Waypoint[] = [
    ['SP', 0, 0],
    ['TP1', 0, 0.15],
    ['TP2', 0, 0.3],
    ['TP3', 0, 0.45],
    ['TP4', 0, 0.6],
    ['TP5', 0, 0.75],
    ['TP6', 0, 0.9],
    ['FP', 0, 1.2],
  ];

  it('passes every automated rule for a compliant route', () => {
    const compliance = evaluateRouteCompliance(
      buildRoute(compliantPoints),
      compliantPoints,
      parseSpeed('75kt'),
      250_000
    );

    expect(compliance.status).toBe('ok');
    expect(compliance.violations).toEqual([]);
    expect(compliance.totalDistanceNm).toBeGreaterThan(70);
    expect(compliance.maximumControlPoints).toBe(8);
  });

  it('keeps the app default route compliant with automated checks', () => {
    const defaultPoints: Waypoint[] = [
      ['SP', 46.60085, 16.180022],
      ['TP1', 46.500801, 16.155215],
      ['TP2', 46.515128, 16.008368],
      ['TP3', 46.800823, 16.0368],
      ['TP4', 46.836955, 16.308447],
      ['TP5', 46.775187, 16.202892],
      ['TP6', 46.552346, 16.430293],
      ['FP', 46.608093, 16.234769],
    ];
    const compliance = evaluateRouteCompliance(
      buildRoute(defaultPoints),
      defaultPoints,
      parseSpeed('75kt'),
      250_000
    );

    expect(compliance.status).toBe('ok');
    expect(compliance.totalDistanceNm).toBeCloseTo(71.56, 2);
    expect(compliance.maximumControlPoints).toBe(8);
  });

  it('counts SP and FP in the maximum control-point rule', () => {
    const points: Waypoint[] = [
      ['SP', 0, 0],
      ['TP1', 0, 0.15],
      ['TP2', 0, 0.3],
      ['TP3', 0, 0.45],
      ['TP4', 0, 0.6],
      ['TP5', 0, 0.75],
      ['TP6', 0, 0.9],
      ['TP7', 0, 1.05],
      ['FP', 0, 1.2],
    ];
    const compliance = evaluateRouteCompliance(buildRoute(points), points, parseSpeed('75kt'), 250_000);

    expect(compliance.maximumControlPoints).toBe(8);
    expect(compliance.violations.map((check) => check.title)).toContain('Control-point limit');
  });

  it('reports all calculable rule violations without blocking generation', () => {
    const points: Waypoint[] = [
      ['START', 0, 0],
      ['FINISH', 0, 0.05],
    ];
    const compliance = evaluateRouteCompliance(buildRoute(points), points, parseSpeed('74kt'), null);

    expect(compliance.status).toBe('against-rules');
    expect(compliance.violations.map((check) => check.title)).toEqual(
      expect.arrayContaining([
        'Official chart scale',
        'Competition groundspeed',
        'Route distance',
        'Minimum leg distance',
        'Start and finish identifiers',
      ])
    );
    expect(compliance.manualChecks.length).toBeGreaterThan(0);
  });
});
