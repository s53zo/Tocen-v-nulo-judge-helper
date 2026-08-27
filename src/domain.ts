export type SpeedUnit = 'kt' | 'mph' | 'kmh';
export type Waypoint = [name: string, latitude: number, longitude: number];

export interface ParsedSpeed {
  value: number;
  unit: SpeedUnit;
  metersPerSecond: number;
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

const KNOT_TO_MPS = 0.514444;
const MPH_TO_MPS = 0.44704;
const KMH_TO_MPS = 1 / 3.6;
const EARTH_RADIUS_M = 6_371_000;

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
    label: `${formattedValue} ${unitLabel}`,
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
