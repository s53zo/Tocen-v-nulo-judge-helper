import { bearingDegrees, haversine, type Route, type Waypoint } from './domain';
import type { PhotoRouteAnalysis } from './photo-types';

const EARTH_RADIUS_M = 6_371_000;

export function normalizeLongitudeDelta(delta: number): number {
  return ((delta + 540) % 360) - 180;
}

export function headingDifference(a: number, b: number): number {
  return Math.abs(((a - b + 540) % 360) - 180);
}

function localMeters(lat: number, lon: number, originLat: number, originLon: number): [number, number] {
  const x =
    (normalizeLongitudeDelta(lon - originLon) *
      Math.PI *
      EARTH_RADIUS_M *
      Math.cos((originLat * Math.PI) / 180)) /
    180;
  const y = ((lat - originLat) * Math.PI * EARTH_RADIUS_M) / 180;
  return [x, y];
}

function interpolateLongitude(from: number, to: number, fraction: number): number {
  const value = from + normalizeLongitudeDelta(to - from) * fraction;
  return ((value + 540) % 360) - 180;
}

export function analyzePhotoPosition(
  latitude: number,
  longitude: number,
  headingDeg: number | null,
  route: Route,
  points: Waypoint[]
): PhotoRouteAnalysis {
  if (route.legs.length === 0 || points.length < 2)
    throw new Error('A route with at least one leg is required.');
  let best:
    | {
        legIndex: number;
        rawFraction: number;
        fraction: number;
        lateralDistanceM: number;
        lateralSignedM: number;
      }
    | undefined;

  route.legs.forEach((leg, legIndex) => {
    const [bx, by] = localMeters(leg.toLat, leg.toLon, leg.fromLat, leg.fromLon);
    const [px, py] = localMeters(latitude, longitude, leg.fromLat, leg.fromLon);
    const lengthSquared = bx * bx + by * by;
    const rawFraction = lengthSquared > 0 ? (px * bx + py * by) / lengthSquared : 0;
    const fraction = Math.max(0, Math.min(1, rawFraction));
    const dx = px - bx * fraction;
    const dy = py - by * fraction;
    const lateralDistanceM = Math.hypot(dx, dy);
    const lateralSignedM = Math.sign(bx * py - by * px || 1) * lateralDistanceM;
    if (!best || lateralDistanceM < best.lateralDistanceM) {
      best = { legIndex, rawFraction, fraction, lateralDistanceM, lateralSignedM };
    }
  });
  if (!best) throw new Error('Could not project the photo onto the route.');

  const leg = route.legs[best.legIndex];
  const alongRouteM = leg.cumulativeStart + best.fraction * leg.length;
  const controlDistances = points.map((point, index) => ({
    name: point[0],
    distance: index === 0 ? 0 : route.legs[index - 1].cumulativeStart + route.legs[index - 1].length,
  }));
  const nearest = controlDistances.reduce((current, candidate) =>
    Math.abs(candidate.distance - alongRouteM) < Math.abs(current.distance - alongRouteM)
      ? candidate
      : current
  );
  const previous =
    [...controlDistances].reverse().find((point) => point.distance <= alongRouteM + 1e-6) ??
    controlDistances[0];
  const legBearingDeg = bearingDegrees(leg.fromLat, leg.fromLon, leg.toLat, leg.toLon);
  const routePosition =
    best.legIndex === 0 && best.rawFraction < 0
      ? 'before-route'
      : best.legIndex === route.legs.length - 1 && best.rawFraction > 1
        ? 'after-route'
        : 'on-route';

  return {
    legIndex: best.legIndex,
    legId: `${leg.fromName}-${leg.toName}`,
    rawFraction: best.rawFraction,
    fraction: best.fraction,
    closestLatitude: leg.fromLat + (leg.toLat - leg.fromLat) * best.fraction,
    closestLongitude: interpolateLongitude(leg.fromLon, leg.toLon, best.fraction),
    lateralDistanceM: best.lateralDistanceM,
    lateralSignedM: best.lateralSignedM,
    alongRouteM,
    distanceOnLegM: best.fraction * leg.length,
    routePosition,
    nearestControlPoint: nearest.name,
    distanceFromNearestControlPointM: alongRouteM - nearest.distance,
    previousControlPoint: previous.name,
    distanceAfterPreviousControlPointM: alongRouteM - previous.distance,
    legBearingDeg,
    headingDifferenceDeg: headingDeg === null ? null : headingDifference(headingDeg, legBearingDeg),
  };
}

export function distanceBetweenSubjects(
  first: { latitude: number; longitude: number },
  second: { latitude: number; longitude: number }
): number {
  return haversine(first.latitude, first.longitude, second.latitude, second.longitude);
}
