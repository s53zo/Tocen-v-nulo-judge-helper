export type SpeedUnit = 'kt' | 'mph' | 'kmh';
export type Waypoint = [name: string, latitude: number, longitude: number];

export interface ParsedSpeed {
  value: number;
  unit: SpeedUnit;
  metersPerSecond: number;
  knots: number;
  label: string;
}

export interface RouteLeg {
  fromName: string;
  toName: string;
  fromLat: number;
  fromLon: number;
  toLat: number;
  toLon: number;
  length: number;
  cumulativeStart: number;
}

export interface Route {
  legs: RouteLeg[];
  totalDistance: number;
}

export interface MinuteMarker {
  minute: number;
  leg: RouteLeg;
  ratio: number;
  lat: number;
  lon: number;
}

export interface ComplianceCheck {
  rule: string;
  title: string;
  passed: boolean;
  message: string;
}

export interface ManualComplianceCheck {
  rule: string;
  message: string;
}

export interface RouteCompliance {
  status: 'ok' | 'against-rules';
  checks: ComplianceCheck[];
  violations: ComplianceCheck[];
  manualChecks: ManualComplianceCheck[];
  totalDistanceNm: number;
  maximumControlPoints: number;
}

const KNOT_TO_MPS = 0.514444;
const MPH_TO_MPS = 0.44704;
const KMH_TO_MPS = 1 / 3.6;
const EARTH_RADIUS_M = 6_371_000;
const METERS_PER_NAUTICAL_MILE = 1852;
const ALLOWED_COMPETITION_SPEEDS_KT = [70, 75, 80, 85, 90];
const SPEED_TOLERANCE_KT = 0.05;

export const ROUTE_RULES = {
  mapScaleDenominator: 250_000,
  minimumDistanceNm: 70,
  maximumDistanceNm: 120,
  minimumLegDistanceNm: 5,
  allowedSpeedsKt: ALLOWED_COMPETITION_SPEEDS_KT,
} as const;

export function parseSpeed(input: string): ParsedSpeed {
  const token = input.trim().toLowerCase().replace(/\s+/g, '') || '75';
  const match = token.match(/^(\d+(?:[.,]\d+)?)(kt|kts|mph|km\/h|kmh|kph)?$/);
  if (!match) {
    throw new Error('Groundspeed must be a positive number followed by kt, mph, or km/h.');
  }

  const value = Number.parseFloat(match[1].replace(',', '.'));
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error('Groundspeed must be greater than zero.');
  }

  const rawUnit = match[2] ?? 'kt';
  const unit: SpeedUnit =
    rawUnit === 'mph' ? 'mph' : rawUnit.startsWith('km') || rawUnit === 'kph' ? 'kmh' : 'kt';
  const multiplier = unit === 'kt' ? KNOT_TO_MPS : unit === 'mph' ? MPH_TO_MPS : KMH_TO_MPS;
  const formattedValue = Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1);
  const unitLabel = unit === 'kmh' ? 'km/h' : unit;

  return {
    value,
    unit,
    metersPerSecond: value * multiplier,
    knots: (value * multiplier) / KNOT_TO_MPS,
    label: `${formattedValue} ${unitLabel}`,
  };
}

export function metersToNauticalMiles(meters: number): number {
  return meters / METERS_PER_NAUTICAL_MILE;
}

export function evaluateRouteCompliance(
  route: Route,
  points: Waypoint[],
  speed: ParsedSpeed,
  mapScaleDenominator: number | null
): RouteCompliance {
  const totalDistanceNm = metersToNauticalMiles(route.totalDistance);
  const maximumControlPoints = Math.ceil(totalDistanceNm / 10);
  const shortLegs = route.legs.filter(
    (leg) => metersToNauticalMiles(leg.length) < ROUTE_RULES.minimumLegDistanceNm - 1e-9
  );
  const matchedSpeed = ROUTE_RULES.allowedSpeedsKt.find(
    (allowed) => Math.abs(speed.knots - allowed) <= SPEED_TOLERANCE_KT
  );
  const firstName = points.at(0)?.[0].trim().toUpperCase();
  const finalName = points.at(-1)?.[0].trim().toUpperCase();

  const checks: ComplianceCheck[] = [
    {
      rule: 'A1.4',
      title: 'Official chart scale',
      passed: mapScaleDenominator === ROUTE_RULES.mapScaleDenominator,
      message:
        mapScaleDenominator === ROUTE_RULES.mapScaleDenominator
          ? 'Selected map is 1:250,000.'
          : mapScaleDenominator
            ? `Selected map is 1:${mapScaleDenominator.toLocaleString('en-US')}; 1:250,000 is required.`
            : 'Selected map has no fixed 1:250,000 competition scale.',
    },
    {
      rule: 'A1.5',
      title: 'Competition groundspeed',
      passed: matchedSpeed !== undefined,
      message:
        matchedSpeed !== undefined
          ? `${speed.label} equals an approved ${matchedSpeed} kt groundspeed.`
          : `${speed.label} equals ${speed.knots.toFixed(2)} kt; allowed values are 70, 75, 80, 85, or 90 kt.`,
    },
    {
      rule: 'A2.1.1',
      title: 'Route distance',
      passed:
        totalDistanceNm >= ROUTE_RULES.minimumDistanceNm - 1e-9 &&
        totalDistanceNm <= ROUTE_RULES.maximumDistanceNm + 1e-9,
      message: `${totalDistanceNm.toFixed(2)} NM total; required range is 70-120 NM.`,
    },
    {
      rule: 'A2.1.2',
      title: 'Minimum leg distance',
      passed: shortLegs.length === 0,
      message:
        shortLegs.length === 0
          ? `Every leg is at least ${ROUTE_RULES.minimumLegDistanceNm} NM.`
          : `Legs below 5 NM: ${shortLegs
              .map(
                (leg) => `${leg.fromName}-${leg.toName} (${metersToNauticalMiles(leg.length).toFixed(2)} NM)`
              )
              .join(', ')}.`,
    },
    {
      rule: 'A2.1.2',
      title: 'Control-point limit',
      passed: points.length <= maximumControlPoints,
      message: `${points.length} control points; a ${totalDistanceNm.toFixed(2)} NM route allows at most ${maximumControlPoints}.`,
    },
    {
      rule: 'A2.1.2',
      title: 'Start and finish identifiers',
      passed: firstName === 'SP' && finalName === 'FP',
      message:
        firstName === 'SP' && finalName === 'FP'
          ? 'Route starts at SP and finishes at FP.'
          : `First and last control points must be named SP and FP (currently ${firstName || 'missing'} and ${finalName || 'missing'}).`,
    },
  ];
  const violations = checks.filter((check) => !check.passed);

  return {
    status: violations.length === 0 ? 'ok' : 'against-rules',
    checks,
    violations,
    totalDistanceNm,
    maximumControlPoints,
    manualChecks: [
      {
        rule: 'A2.1.1',
        message: 'Confirm that the route finishes with a precision landing.',
      },
      {
        rule: 'A2.2.1-A2.2.3',
        message:
          'Confirm every control point has an unambiguous navigation-plan description and coordinates or course/distance.',
      },
      {
        rule: 'A2.3.1-A2.3.5',
        message:
          'Designate 3-5 timed control points and confirm direction, 500 ft AGL minimum, 0.5 NM approach tolerance, and no prohibited circling.',
      },
      {
        rule: 'A2.4.1-A2.4.7',
        message:
          'Check observation tasks: at most 12 route photos, at most 15 tasks, required photo/sign placement, and no task in the 1 NM segment after a control point.',
      },
      {
        rule: 'A2.6; A3.9',
        message:
          'Confirm the approved WGS 84 IGC logger setup, chart currency/event approval, weather, airspace, and all VFR operational requirements.',
      },
    ],
  };
}

export function haversine(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const [phi1, phi2, deltaPhi, deltaLambda] = [lat1, lat2, lat2 - lat1, lon2 - lon1].map(
    (degrees) => (degrees * Math.PI) / 180
  );
  const a = Math.sin(deltaPhi / 2) ** 2 + Math.cos(phi1) * Math.cos(phi2) * Math.sin(deltaLambda / 2) ** 2;
  return EARTH_RADIUS_M * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

export function buildRoute(points: Waypoint[]): Route {
  if (points.length < 2) {
    throw new Error('At least two waypoints are required.');
  }

  const legs: RouteLeg[] = [];
  let cumulative = 0;
  for (let index = 0; index < points.length - 1; index += 1) {
    const [fromName, fromLat, fromLon] = points[index];
    const [toName, toLat, toLon] = points[index + 1];
    const length = haversine(fromLat, fromLon, toLat, toLon);
    if (length < 0.01) {
      throw new Error(`Waypoints ${fromName} and ${toName} have the same position.`);
    }
    legs.push({
      fromName,
      toName,
      fromLat,
      fromLon,
      toLat,
      toLon,
      length,
      cumulativeStart: cumulative,
    });
    cumulative += length;
  }
  return { legs, totalDistance: cumulative };
}

export function bearingDegrees(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const [lat1Rad, lat2Rad, deltaLon] = [lat1, lat2, lon2 - lon1].map((degrees) => (degrees * Math.PI) / 180);
  const y = Math.sin(deltaLon) * Math.cos(lat2Rad);
  const x =
    Math.cos(lat1Rad) * Math.sin(lat2Rad) - Math.sin(lat1Rad) * Math.cos(lat2Rad) * Math.cos(deltaLon);
  return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
}

export function roundedBearing(degrees: number): number {
  return ((Math.round(degrees) % 360) + 360) % 360;
}

export function computeWaypointTimes(
  route: Route,
  points: Waypoint[],
  takeoffToStartMinutes: number,
  metersPerMinute: number
): Map<string, number> {
  if (!Number.isFinite(metersPerMinute) || metersPerMinute <= 0) {
    throw new Error('Groundspeed must be greater than zero.');
  }
  return new Map(
    points.map((point, index) => {
      const distance = index === 0 ? 0 : route.legs[index - 1].cumulativeStart + route.legs[index - 1].length;
      return [point[0], takeoffToStartMinutes + distance / metersPerMinute];
    })
  );
}

export function computeMinuteMarkers(
  route: Route,
  takeoffToStartMinutes: number,
  metersPerMinute: number,
  intervalMinutes: number
): MinuteMarker[] {
  if (metersPerMinute <= 0 || intervalMinutes <= 0) {
    throw new Error('Speed and minute-marker interval must be greater than zero.');
  }
  const markers: MinuteMarker[] = [];
  const finalMinute = takeoffToStartMinutes + route.totalDistance / metersPerMinute;
  for (let minute = 0; minute <= finalMinute + 1e-6; minute += intervalMinutes) {
    const distanceFromStart = (minute - takeoffToStartMinutes) * metersPerMinute;
    if (distanceFromStart < -1e-6 || distanceFromStart > route.totalDistance + 1e-6) continue;
    let remaining = distanceFromStart;
    for (const leg of route.legs) {
      if (remaining <= leg.length + 1e-6) {
        const ratio = remaining / leg.length;
        markers.push({
          minute,
          leg,
          ratio,
          lat: leg.fromLat + ratio * (leg.toLat - leg.fromLat),
          lon: leg.fromLon + ratio * (leg.toLon - leg.fromLon),
        });
        break;
      }
      remaining -= leg.length;
    }
  }
  return markers;
}
