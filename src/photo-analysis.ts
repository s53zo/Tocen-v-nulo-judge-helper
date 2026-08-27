import { bearingDegrees, haversine, type Route, type Waypoint } from './domain';
import type { PhotoRouteAnalysis } from './photo-types';

const EARTH_RADIUS_M = 6_371_000;
const AMBIGUITY_TOLERANCE_M = 2;

export function normalizeLongitudeDelta(delta: number): number {
  return ((delta + 540) % 360) - 180;
}

export function headingDifference(a: number, b: number): number {
  return Math.abs(((a - b + 540) % 360) - 180);
}

function radians(degrees: number): number {
  return (degrees * Math.PI) / 180;
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
  points: Waypoint[],
  preferredLegIndex: number | null = null
): PhotoRouteAnalysis {
  if (route.legs.length === 0 || points.length < 2)
    throw new Error('A route with at least one leg is required.');
  const candidates = route.legs.map((leg, legIndex) => {
    const angularDistance = haversine(leg.fromLat, leg.fromLon, latitude, longitude) / EARTH_RADIUS_M;
    const legBearing = radians(bearingDegrees(leg.fromLat, leg.fromLon, leg.toLat, leg.toLon));
    const pointBearing = radians(bearingDegrees(leg.fromLat, leg.fromLon, latitude, longitude));
    const crossTrackRadians = Math.asin(
      Math.max(-1, Math.min(1, Math.sin(angularDistance) * Math.sin(pointBearing - legBearing)))
    );
    const alongTrackRadians = Math.atan2(
      Math.sin(angularDistance) * Math.cos(pointBearing - legBearing),
      Math.cos(angularDistance)
    );
    const rawFraction = (alongTrackRadians * EARTH_RADIUS_M) / leg.length;
    const fraction = Math.max(0, Math.min(1, rawFraction));
    // Preserve the app convention: positive is left of the route direction.
    const lateralSignedM = -crossTrackRadians * EARTH_RADIUS_M;
    const lateralDistanceM =
      rawFraction < 0
        ? haversine(latitude, longitude, leg.fromLat, leg.fromLon)
        : rawFraction > 1
          ? haversine(latitude, longitude, leg.toLat, leg.toLon)
          : Math.abs(lateralSignedM);
    return { legIndex, rawFraction, fraction, lateralDistanceM, lateralSignedM };
  });
  if (candidates.length === 0) throw new Error('Could not project the photo onto the route.');
  const nearestDistance = Math.min(...candidates.map((candidate) => candidate.lateralDistanceM));
  const ambiguous = candidates.filter(
    (candidate) => candidate.lateralDistanceM <= nearestDistance + AMBIGUITY_TOLERANCE_M
  );
  const preferred =
    preferredLegIndex === null
      ? undefined
      : candidates.find((candidate) => candidate.legIndex === preferredLegIndex);
  const best =
    preferred ??
    [...ambiguous].sort((first, second) => {
      const firstDistance = first.fraction * route.legs[first.legIndex].length;
      const secondDistance = second.fraction * route.legs[second.legIndex].length;
      return firstDistance - secondDistance || second.legIndex - first.legIndex;
    })[0];

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
    ambiguousLegIndices: ambiguous.map((candidate) => candidate.legIndex),
    manuallySelectedLeg: preferred !== undefined,
  };
}

export function distanceBetweenSubjects(
  first: { latitude: number; longitude: number },
  second: { latitude: number; longitude: number }
): number {
  return haversine(first.latitude, first.longitude, second.latitude, second.longitude);
}
