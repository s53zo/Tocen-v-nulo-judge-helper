import { haversine, type Waypoint } from './domain';
import type { PhotoComplianceSummary, PhotoFinding, PhotoRecord } from './photo-types';

const NM_M = 1852;

function finding(
  severity: PhotoFinding['severity'],
  code: string,
  rule: string,
  affected: string,
  message: string,
  measured: string,
  permitted: string
): PhotoFinding {
  return { severity, code, rule, affected, message, measured, permitted };
}

function isRouteTask(photo: PhotoRecord): boolean {
  if (photo.classification === 'enroute') return true;
  if (photo.classification !== 'sign-task') return false;
  return !photo.linkedWaypoint || !/^TP\d+$/i.test(photo.linkedWaypoint);
}

export function evaluatePhotoCompliance(photos: PhotoRecord[], points: Waypoint[]): PhotoComplianceSummary {
  const findings: PhotoFinding[] = [];
  const enroute = photos.filter((photo) => photo.classification === 'enroute');
  const tasks = photos.filter(isRouteTask);
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
      'maximum 15, excluding photos/signs at turning points'
    )
  );

  const waypointLookup = new Map(
    points.map(([name, latitude, longitude]) => [name.toUpperCase(), { latitude, longitude }])
  );
  for (const photo of photos) {
    const affected = `${photo.identifier || '?'} (${photo.fileName})`;
    const positionReliable = photo.metadata.latitude.reliable && photo.metadata.longitude.reliable;
    if (isRouteTask(photo) && photo.analysis && positionReliable) {
      const after = photo.analysis.distanceAfterPreviousControlPointM;
      findings.push(
        finding(
          after >= NM_M ? 'pass' : 'violation',
          'post-control-spacing',
          'A2.4.6',
          affected,
          after >= NM_M
            ? 'Task is outside the prohibited segment after the previous control point.'
            : `Task is too close after ${photo.analysis.previousControlPoint}.`,
          `${(after / NM_M).toFixed(2)} NM after ${photo.analysis.previousControlPoint}`,
          'at least 1.00 NM after a control point'
        )
      );
    } else if (isRouteTask(photo)) {
      findings.push(
        finding(
          'warning',
          'post-control-spacing-unavailable',
          'A2.4.6',
          affected,
          'Post-control task spacing requires manual review.',
          'reliable position unavailable',
          'at least 1.00 NM after a control point'
        )
      );
    }

    if (photo.classification === 'enroute') {
      if (!photo.analysis || !positionReliable) {
        findings.push(
          finding(
            'warning',
            'photo-position-missing',
            'A2.4.5',
            affected,
            'Photo position cannot be checked.',
            photo.analysis ? 'GPS is marked unreliable' : 'GPS unavailable',
            'position required'
          )
        );
      } else {
        findings.push(
          finding(
            photo.analysis.lateralDistanceM <= 300 ? 'pass' : 'violation',
            'route-axis-distance',
            'A2.4.5',
            affected,
            photo.analysis.lateralDistanceM <= 300
              ? 'Photo is within the route-axis limit.'
              : 'Photo is too far from the route axis.',
            `${photo.analysis.lateralDistanceM.toFixed(0)} m`,
            'maximum 300 m'
          )
        );
      }
      if (
        photo.analysis?.headingDifferenceDeg !== null &&
        photo.analysis?.headingDifferenceDeg !== undefined &&
        photo.metadata.headingDeg.reliable &&
        positionReliable
      ) {
        findings.push(
          finding(
            photo.analysis.headingDifferenceDeg <= 45 ? 'pass' : 'violation',
            'camera-angle',
            'A2.4.5',
            affected,
            photo.analysis.headingDifferenceDeg <= 45
              ? 'Camera direction is within the route-leg limit.'
              : 'Camera direction exceeds the route-leg limit.',
            `${photo.analysis.headingDifferenceDeg.toFixed(1)} degrees`,
            'maximum 45 degrees from leg direction'
          )
        );
      } else {
        findings.push(
          finding(
            'warning',
            'camera-angle-missing',
            'A2.4.5',
            affected,
            'Camera angle requires manual review.',
            'reliable heading unavailable',
            'maximum 45 degrees from leg direction'
          )
        );
      }
      const altitude = photo.metadata.altitudeAglFt;
      findings.push(
        altitude.value !== null && altitude.reliable
          ? finding(
              altitude.value >= 500 && altitude.value <= 1000 ? 'pass' : 'violation',
              'capture-altitude',
              'A2.4.5',
              affected,
              altitude.value >= 500 && altitude.value <= 1000
                ? 'Capture altitude is within range.'
                : 'Capture altitude is outside the permitted range.',
              `${altitude.value.toFixed(0)} ft AGL`,
              '500-1,000 ft AGL'
            )
          : finding(
              'warning',
              'capture-altitude-missing',
              'A2.4.5',
              affected,
              'Capture altitude requires manual review.',
              'reliable AGL unavailable',
              '500-1,000 ft AGL'
            )
      );
      const focal = photo.metadata.focalLength35Mm;
      findings.push(
        focal.value !== null && focal.reliable
          ? finding(
              focal.value >= 50 && focal.value <= 70 ? 'pass' : 'violation',
              'focal-length',
              'A2.4.5',
              affected,
              focal.value >= 50 && focal.value <= 70
                ? 'Equivalent focal length is within range.'
                : 'Equivalent focal length is outside the permitted range.',
              `${focal.value.toFixed(1)} mm equivalent`,
              '50-70 mm equivalent'
            )
          : finding(
              'warning',
              'focal-length-missing',
              'A2.4.5',
              affected,
              'Focal length requires manual review.',
              'reliable 35 mm equivalent unavailable',
              '50-70 mm equivalent'
            )
      );
    }

    if (photo.classification === 'control-false') {
      const correct = photo.linkedWaypoint
        ? waypointLookup.get(photo.linkedWaypoint.toUpperCase())
        : undefined;
      if (correct && photo.subjectLatitude !== null && photo.subjectLongitude !== null) {
        const distance = haversine(
          photo.subjectLatitude,
          photo.subjectLongitude,
          correct.latitude,
          correct.longitude
        );
        findings.push(
          finding(
            distance >= NM_M ? 'pass' : 'violation',
            'false-control-distance',
            'A2.4.2',
            affected,
            distance >= NM_M
              ? 'False object is sufficiently separated from the correct object.'
              : 'False object is too close to the correct object.',
            `${(distance / NM_M).toFixed(2)} NM`,
            'at least 1.00 NM'
          )
        );
      } else {
        findings.push(
          finding(
            'warning',
            'false-control-distance-missing',
            'A2.4.2',
            affected,
            'False-object separation requires manual review.',
            'subject coordinate or linked waypoint missing',
            'at least 1.00 NM from correct object'
          )
        );
      }
    }
  }

  const violationCount = findings.filter((item) => item.severity === 'violation').length;
  const warningCount = findings.filter((item) => item.severity === 'warning').length;
  return {
    status: violationCount > 0 ? 'against-rules' : warningCount > 0 ? 'manual-review' : 'ok',
    findings,
    violationCount,
    warningCount,
    enroutePhotoCount: enroute.length,
    routeTaskCount: tasks.length,
  };
}
