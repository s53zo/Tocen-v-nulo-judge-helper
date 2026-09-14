import { haversine, type Waypoint } from './domain';
import type { PhotoComplianceSummary, PhotoFinding, PhotoRecord } from './photo-types';

const NM_M = 1852;
const ENROUTE_CAMERA_ROUTE_AXIS_LIMIT_M = 500;

export type FindingPresentation = 'primary' | 'action' | 'audit';

const ACTION_FINDING_CODES = new Set(['task-position-missing']);
const AUDIT_FINDING_CODES = new Set([
  'camera-angle-missing',
  'capture-altitude-missing',
  'focal-length',
  'focal-length-missing',
  'judge-content-review',
]);

export function findingPresentation(finding: PhotoFinding): FindingPresentation {
  if (ACTION_FINDING_CODES.has(finding.code)) return 'action';
  if (AUDIT_FINDING_CODES.has(finding.code)) return 'audit';
  return 'primary';
}

export function isOperationalFinding(finding: PhotoFinding): boolean {
  return findingPresentation(finding) !== 'audit';
}

export function operationalFindingCounts(findings: PhotoFinding[]): {
  violationCount: number;
  warningCount: number;
} {
  const operational = findings.filter(isOperationalFinding);
  return {
    violationCount: operational.filter((finding) => finding.severity === 'violation').length,
    warningCount: operational.filter((finding) => finding.severity === 'warning').length,
  };
}

export function isPhotoAcceptedForJudge(photo: PhotoRecord): boolean {
  return (
    photo.exceptionAccepted ||
    !photo.findings.some((finding) => isOperationalFinding(finding) && finding.severity === 'violation')
  );
}

export type WaypointRole = 'start' | 'turning-point' | 'control-point' | 'finish';

export function waypointRole(point: Waypoint, index: number, points: Waypoint[]): WaypointRole {
  const name = point[0].trim().toUpperCase();
  if (index === 0 && name === 'SP') return 'start';
  if (index === points.length - 1 && name === 'FP') return 'finish';
  return /^TP\d+$/.test(name) ? 'turning-point' : 'control-point';
}

function linkedWaypoint(photo: PhotoRecord, points: Waypoint[]): { point: Waypoint; index: number } | null {
  if (!photo.linkedWaypoint) return null;
  const index = points.findIndex(
    ([name]) => name.trim().toUpperCase() === photo.linkedWaypoint?.trim().toUpperCase()
  );
  return index < 0 ? null : { point: points[index], index };
}

function isObservationTask(photo: PhotoRecord): boolean {
  return photo.classification !== 'reference';
}

export function isCountedRouteTask(photo: PhotoRecord, points: Waypoint[]): boolean {
  if (!isObservationTask(photo)) return false;
  const linked = linkedWaypoint(photo, points);
  return !linked || waypointRole(linked.point, linked.index, points) !== 'turning-point';
}

function finding(
  severity: PhotoFinding['severity'],
  code: string,
  rule: string,
  affected: string,
  message: string,
  measured: string,
  permitted: string,
  photoId: string | null = null
): PhotoFinding {
  return { photoId, severity, code, rule, affected, message, measured, permitted };
}

function taskCoordinates(photo: PhotoRecord, points: Waypoint[]): [number, number] | null {
  if (
    photo.taskLatitude.reliable &&
    photo.taskLongitude.reliable &&
    photo.taskLatitude.value !== null &&
    photo.taskLongitude.value !== null
  ) {
    return [photo.taskLatitude.value, photo.taskLongitude.value];
  }
  const linked = linkedWaypoint(photo, points);
  return linked ? [linked.point[1], linked.point[2]] : null;
}

export function evaluatePhotoCompliance(photos: PhotoRecord[], points: Waypoint[]): PhotoComplianceSummary {
  const findings: PhotoFinding[] = [];
  const enroute = photos.filter((photo) => photo.classification === 'enroute');
  const tasks = photos.filter((photo) => isCountedRouteTask(photo, points));
  findings.push(
    finding(
      enroute.length <= 12 ? 'pass' : 'violation',
      'enroute-photo-count',
      'A2.4.5',
      'route',
      enroute.length <= 12 ? 'En-route photo count is within the limit.' : 'Too many en-route photos.',
      `${enroute.length} en-route photos`,
      'maximum 12'
    ),
    finding(
      tasks.length <= 15 ? 'pass' : 'violation',
      'route-task-count',
      'A2.4.6',
      'route',
      tasks.length <= 15 ? 'Route task count is within the limit.' : 'Too many route tasks.',
      `${tasks.length} route tasks`,
      'maximum 15, excluding verified photos/signs at turning points'
    )
  );

  const normalizedIdentifiers = enroute.map((photo) => photo.identifier.trim().toUpperCase());
  const identifiersValid =
    normalizedIdentifiers.every((identifier) => /^[A-Z]$/.test(identifier)) &&
    new Set(normalizedIdentifiers).size === normalizedIdentifiers.length;
  findings.push(
    finding(
      identifiersValid ? 'pass' : 'violation',
      'enroute-identifiers',
      'A2.4.5',
      'route',
      identifiersValid
        ? 'En-route photo identifiers are unique single letters.'
        : 'En-route photo identifiers must be unique single letters.',
      normalizedIdentifiers.join(', ') || 'no en-route photos',
      'unique letters A-Z'
    )
  );

  const waypointLookup = new Map(
    points.map(([name, latitude, longitude]) => [name.toUpperCase(), { latitude, longitude }])
  );
  for (const photo of photos) {
    const affected = `${photo.identifier || '?'} (${photo.fileName})`;
    const photoFinding = (
      severity: PhotoFinding['severity'],
      code: string,
      rule: string,
      message: string,
      measured: string,
      permitted: string
    ) => finding(severity, code, rule, affected, message, measured, permitted, photo.id);
    const cameraPositionReliable = photo.metadata.latitude.reliable && photo.metadata.longitude.reliable;
    const linked = linkedWaypoint(photo, points);

    if (photo.importError) {
      findings.push(
        photoFinding(
          'warning',
          'photo-import-error',
          'A2.4.1-A2.4.8',
          'The image or metadata import failed and requires manual review.',
          photo.importError,
          'readable image and reviewable metadata'
        )
      );
    }
    if (photo.linkedWaypoint && !linked) {
      findings.push(
        photoFinding(
          'warning',
          'stale-waypoint-link',
          'A2.4.2-A2.4.7',
          'The linked waypoint does not exist in the current route.',
          photo.linkedWaypoint,
          'an existing route waypoint'
        )
      );
    }

    const isLinkedControlPhoto =
      Boolean(linked) &&
      (photo.classification === 'control-correct' || photo.classification === 'control-false');
    if (isCountedRouteTask(photo, points) && !isLinkedControlPhoto) {
      if (photo.taskAnalysis) {
        if (photo.taskAnalysis.routePosition !== 'on-route') {
          findings.push(
            photoFinding(
              'violation',
              'task-outside-route',
              'A2.4.6-A2.4.7',
              'The task lies before SP or after FP.',
              photo.taskAnalysis.routePosition,
              'a task on the competition route'
            )
          );
        } else {
          const after = photo.taskAnalysis.distanceAfterPreviousControlPointM;
          findings.push(
            photoFinding(
              after >= NM_M ? 'pass' : 'violation',
              'post-control-spacing',
              'A2.4.6',
              after >= NM_M
                ? 'Task is outside the prohibited segment after the previous control point.'
                : `Task is too close after ${photo.taskAnalysis.previousControlPoint}.`,
              `${(after / NM_M).toFixed(2)} NM after ${photo.taskAnalysis.previousControlPoint}`,
              'at least 1.00 NM after a control point'
            )
          );
        }
        if (photo.taskAnalysis.ambiguousLegIndices.length > 1 && !photo.taskAnalysis.manuallySelectedLeg) {
          findings.push(
            photoFinding(
              'warning',
              'task-leg-ambiguous',
              'A2.4.6-A2.4.7',
              'Task position matches multiple route legs; confirm its intended sequence.',
              `candidate legs ${photo.taskAnalysis.ambiguousLegIndices.map((index) => index + 1).join(', ')}`,
              'one confirmed route leg'
            )
          );
        }
      } else {
        findings.push(
          photoFinding(
            'warning',
            'task-position-missing',
            'A2.4.6-A2.4.7',
            'Task spacing and map placement require a separate task/object position.',
            'reliable task position unavailable',
            'task/object coordinates or a valid linked waypoint'
          )
        );
      }
    }

    if (photo.classification === 'sign-task') {
      findings.push(
        photo.taskAnalysis
          ? photoFinding(
              photo.taskAnalysis.lateralDistanceM <= 100 ? 'pass' : 'violation',
              'sign-route-axis-distance',
              'A2.4.4',
              photo.taskAnalysis.lateralDistanceM <= 100
                ? 'Sign task is within the route-axis limit.'
                : 'Sign task is too far from the route axis.',
              `${photo.taskAnalysis.lateralDistanceM.toFixed(1)} m`,
              'maximum 100 m'
            )
          : photoFinding(
              'warning',
              'sign-route-axis-distance-missing',
              'A2.4.4',
              'Sign distance from the route requires manual review.',
              'task position unavailable',
              'maximum 100 m'
            )
      );
    }

    if (photo.classification === 'enroute') {
      if (!photo.analysis || !cameraPositionReliable) {
        findings.push(
          photoFinding(
            'warning',
            'photo-position-missing',
            'A2.4.5',
            'Camera position cannot be checked.',
            photo.analysis ? 'GPS is marked unreliable' : 'GPS unavailable',
            'reliable camera position'
          )
        );
      } else {
        findings.push(
          photoFinding(
            photo.analysis.lateralDistanceM <= ENROUTE_CAMERA_ROUTE_AXIS_LIMIT_M ? 'pass' : 'violation',
            'route-axis-distance',
            'A2.4.5',
            photo.analysis.lateralDistanceM <= ENROUTE_CAMERA_ROUTE_AXIS_LIMIT_M
              ? 'Camera is within the configured route-axis screening limit.'
              : 'Camera exceeds the configured route-axis screening limit.',
            `${photo.analysis.lateralDistanceM.toFixed(0)} m`,
            `configured maximum ${ENROUTE_CAMERA_ROUTE_AXIS_LIMIT_M} m`
          )
        );
      }
      if (
        photo.analysis?.headingDifferenceDeg !== null &&
        photo.analysis?.headingDifferenceDeg !== undefined &&
        photo.metadata.headingDeg.reliable &&
        photo.metadata.headingReference.value === 'true' &&
        photo.metadata.headingReference.reliable &&
        cameraPositionReliable
      ) {
        findings.push(
          photoFinding(
            photo.analysis.headingDifferenceDeg <= 45 ? 'pass' : 'violation',
            'camera-angle',
            'A2.4.5',
            photo.analysis.headingDifferenceDeg <= 45
              ? 'True camera direction is within the route-leg limit.'
              : 'True camera direction exceeds the route-leg limit.',
            `${photo.analysis.headingDifferenceDeg.toFixed(1)} degrees`,
            'maximum 45 degrees from true leg direction'
          )
        );
      } else {
        findings.push(
          photoFinding(
            'warning',
            'camera-angle-missing',
            'A2.4.5',
            'Camera angle requires manual review.',
            `heading reference ${photo.metadata.headingReference.value ?? 'unavailable'}`,
            'reliable true heading within 45 degrees of leg direction'
          )
        );
      }
      const altitude = photo.metadata.altitudeAglFt;
      findings.push(
        altitude.value !== null && altitude.reliable
          ? photoFinding(
              altitude.value >= 500 && altitude.value <= 1000 ? 'pass' : 'violation',
              'capture-altitude',
              'A2.4.5',
              altitude.value >= 500 && altitude.value <= 1000
                ? 'Capture altitude is within range.'
                : 'Capture altitude is outside the permitted range.',
              `${altitude.value.toFixed(0)} ft AGL`,
              '500-1,000 ft AGL'
            )
          : photoFinding(
              'warning',
              'capture-altitude-missing',
              'A2.4.5',
              'Capture altitude requires manual review.',
              'reliable AGL unavailable',
              '500-1,000 ft AGL'
            )
      );
      const focal = photo.metadata.focalLength35Mm;
      findings.push(
        focal.value !== null && focal.reliable
          ? photoFinding(
              focal.value >= 50 && focal.value <= 70 ? 'pass' : 'violation',
              'focal-length',
              'A2.4.5',
              focal.value >= 50 && focal.value <= 70
                ? 'Equivalent focal length is within range.'
                : 'Equivalent focal length is outside the permitted range.',
              `${focal.value.toFixed(1)} mm equivalent`,
              '50-70 mm equivalent'
            )
          : photoFinding(
              'warning',
              'focal-length-missing',
              'A2.4.5',
              'Focal length requires manual review.',
              'reliable 35 mm equivalent unavailable',
              '50-70 mm equivalent'
            )
      );
    }

    if (photo.classification === 'control-false') {
      const generatedSource = photo.generatedOrthophoto?.targetSource;
      const correct =
        generatedSource?.correctObjectLatitude !== undefined &&
        generatedSource.correctObjectLongitude !== undefined
          ? {
              latitude: generatedSource.correctObjectLatitude,
              longitude: generatedSource.correctObjectLongitude,
            }
          : photo.linkedWaypoint
            ? waypointLookup.get(photo.linkedWaypoint.toUpperCase())
            : undefined;
      const task = taskCoordinates(photo, points);
      if (correct && task) {
        const distance = haversine(task[0], task[1], correct.latitude, correct.longitude);
        findings.push(
          photoFinding(
            distance >= NM_M ? 'pass' : 'violation',
            'false-control-distance',
            'A2.4.2',
            distance >= NM_M
              ? 'False object is sufficiently separated from the correct object.'
              : 'False object is too close to the correct object.',
            `${(distance / NM_M).toFixed(2)} NM`,
            'at least 1.00 NM'
          )
        );
      } else {
        findings.push(
          photoFinding(
            'warning',
            'false-control-distance-missing',
            'A2.4.2',
            'False-object separation requires manual review.',
            'task coordinate or linked waypoint missing',
            'at least 1.00 NM from correct object'
          )
        );
      }
    }

    if (photo.classification !== 'reference') {
      findings.push(
        photoFinding(
          'warning',
          'judge-content-review',
          'A2.4.1-A2.4.8',
          'A judge must confirm content, quality, identification, map marking, and presentation requirements.',
          'not safely automatable from metadata',
          'manual judge confirmation'
        )
      );
    }
  }

  const { violationCount, warningCount } = operationalFindingCounts(findings);
  return {
    status: violationCount > 0 ? 'against-rules' : warningCount > 0 ? 'manual-review' : 'ok',
    findings,
    violationCount,
    warningCount,
    enroutePhotoCount: enroute.length,
    routeTaskCount: tasks.length,
  };
}
