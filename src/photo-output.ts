import type { PhotoComplianceSummary, PhotoRecord } from './photo-types';

function csvCell(value: unknown): string {
  const raw = value === null || value === undefined ? '' : String(value);
  const text = typeof value === 'string' && /^[=+\-@\t\r]/.test(raw) ? `'${raw}` : raw;
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export const PHOTO_ANALYSIS_COLUMNS = [
  'order',
  'identifier',
  'file_name',
  'classification',
  'linked_waypoint',
  'capture_time',
  'capture_time_source',
  'capture_time_reliable',
  'original_exif_capture_time',
  'latitude',
  'latitude_source',
  'original_exif_latitude',
  'longitude',
  'longitude_source',
  'original_exif_longitude',
  'gps_altitude_msl_m',
  'altitude_agl_ft',
  'altitude_agl_source',
  'altitude_agl_reliable',
  'heading_deg',
  'heading_source',
  'heading_reliable',
  'heading_reference',
  'original_exif_heading_deg',
  'focal_length_mm',
  'focal_length_35mm_equivalent',
  'focal_length_35mm_source',
  'focal_length_35mm_reliable',
  'original_exif_focal_length_35mm_equivalent',
  'camera',
  'lens',
  'nearest_leg',
  'fraction_on_leg',
  'along_route_m',
  'lateral_m',
  'closest_latitude',
  'closest_longitude',
  'previous_control_point',
  'distance_after_control_point_nm',
  'heading_difference_deg',
  'nearest_control_point',
  'signed_distance_from_nearest_control_point_m',
  'route_position',
  'task_latitude',
  'task_latitude_source',
  'task_longitude',
  'task_longitude_source',
  'task_nearest_leg',
  'task_lateral_m',
  'task_route_position',
  'estimated_route_sequence',
  'status',
  'finding_codes',
  'finding_details',
] as const;

function photoStatus(photo: PhotoRecord): string {
  if (photo.findings.some((finding) => finding.severity === 'violation')) return 'against-rules';
  if (photo.findings.some((finding) => finding.severity === 'warning')) return 'manual-review';
  return 'ok';
}

export function photoAnalysisRows(photos: PhotoRecord[]): Record<string, unknown>[] {
  const routeSequence = new Map(
    [...photos]
      .filter((photo) => photo.analysis)
      .sort((left, right) => (left.analysis?.alongRouteM ?? 0) - (right.analysis?.alongRouteM ?? 0))
      .map((photo, index) => [photo.id, index + 1])
  );
  return photos.map((photo, index) => ({
    order: index + 1,
    identifier: photo.identifier,
    file_name: photo.fileName,
    classification: photo.classification,
    linked_waypoint: photo.linkedWaypoint,
    capture_time: photo.metadata.captureTime.value,
    capture_time_source: photo.metadata.captureTime.source,
    capture_time_reliable: photo.metadata.captureTime.reliable,
    original_exif_capture_time: photo.originalMetadata.captureTime.value,
    latitude: photo.metadata.latitude.value,
    latitude_source: photo.metadata.latitude.source,
    original_exif_latitude: photo.originalMetadata.latitude.value,
    longitude: photo.metadata.longitude.value,
    longitude_source: photo.metadata.longitude.source,
    original_exif_longitude: photo.originalMetadata.longitude.value,
    gps_altitude_msl_m: photo.metadata.gpsAltitudeMslM.value,
    altitude_agl_ft: photo.metadata.altitudeAglFt.value,
    altitude_agl_source: photo.metadata.altitudeAglFt.source,
    altitude_agl_reliable: photo.metadata.altitudeAglFt.reliable,
    heading_deg: photo.metadata.headingDeg.value,
    heading_source: photo.metadata.headingDeg.source,
    heading_reliable: photo.metadata.headingDeg.reliable,
    heading_reference: photo.metadata.headingReference.value,
    original_exif_heading_deg: photo.originalMetadata.headingDeg.value,
    focal_length_mm: photo.metadata.focalLengthMm.value,
    focal_length_35mm_equivalent: photo.metadata.focalLength35Mm.value,
    focal_length_35mm_source: photo.metadata.focalLength35Mm.source,
    focal_length_35mm_reliable: photo.metadata.focalLength35Mm.reliable,
    original_exif_focal_length_35mm_equivalent: photo.originalMetadata.focalLength35Mm.value,
    camera: [photo.metadata.cameraMake.value, photo.metadata.cameraModel.value].filter(Boolean).join(' '),
    lens: photo.metadata.lensModel.value,
    nearest_leg: photo.analysis?.legId,
    fraction_on_leg: photo.analysis?.fraction,
    along_route_m: photo.analysis?.alongRouteM,
    lateral_m: photo.analysis?.lateralDistanceM,
    closest_latitude: photo.analysis?.closestLatitude,
    closest_longitude: photo.analysis?.closestLongitude,
    previous_control_point: photo.analysis?.previousControlPoint,
    distance_after_control_point_nm:
      photo.analysis === null ? null : photo.analysis.distanceAfterPreviousControlPointM / 1852,
    heading_difference_deg: photo.analysis?.headingDifferenceDeg,
    nearest_control_point: photo.analysis?.nearestControlPoint,
    signed_distance_from_nearest_control_point_m: photo.analysis?.distanceFromNearestControlPointM,
    route_position: photo.analysis?.routePosition,
    task_latitude: photo.taskLatitude.value,
    task_latitude_source: photo.taskLatitude.source,
    task_longitude: photo.taskLongitude.value,
    task_longitude_source: photo.taskLongitude.source,
    task_nearest_leg: photo.taskAnalysis?.legId,
    task_lateral_m: photo.taskAnalysis?.lateralDistanceM,
    task_route_position: photo.taskAnalysis?.routePosition,
    estimated_route_sequence: routeSequence.get(photo.id),
    status: photoStatus(photo),
    finding_codes: photo.findings
      .filter((finding) => finding.severity !== 'pass')
      .map((finding) => finding.code)
      .join(';'),
    finding_details: JSON.stringify(
      photo.findings
        .filter((finding) => finding.severity !== 'pass')
        .map(({ rule, severity, measured, permitted, message }) => ({
          rule,
          severity,
          measured,
          permitted,
          message,
        }))
    ),
  }));
}

export function recordsToCsv(columns: readonly string[], rows: Record<string, unknown>[]): string {
  return [
    columns.map(csvCell).join(','),
    ...rows.map((row) => columns.map((column) => csvCell(row[column])).join(',')),
  ].join('\r\n');
}

export function photoAnalysisCsv(photos: PhotoRecord[]): string {
  return recordsToCsv(PHOTO_ANALYSIS_COLUMNS, photoAnalysisRows(photos));
}

export function photoOverlayKeyCsv(photos: PhotoRecord[]): string {
  const columns = [
    'identifier',
    'file_name',
    'classification',
    'linked_waypoint',
    'nearest_leg',
    'route_position',
    'along_route_m',
    'marker_latitude',
    'marker_longitude',
    'exact_latitude',
    'exact_longitude',
  ];
  const rows = photos.map((photo) => ({
    identifier: photo.identifier,
    file_name: photo.fileName,
    classification: photo.classification,
    linked_waypoint: photo.linkedWaypoint,
    nearest_leg: photo.taskAnalysis?.legId,
    route_position: photo.taskAnalysis?.routePosition,
    along_route_m: photo.taskAnalysis?.alongRouteM,
    marker_latitude: photo.taskAnalysis?.closestLatitude,
    marker_longitude: photo.taskAnalysis?.closestLongitude,
    exact_latitude: photo.metadata.latitude.value,
    exact_longitude: photo.metadata.longitude.value,
  }));
  return recordsToCsv(columns, rows);
}

export function photoSummaryJson(photos: PhotoRecord[], compliance: PhotoComplianceSummary) {
  return {
    schemaVersion: 2,
    status: compliance.status,
    counts: {
      photos: photos.length,
      enroutePhotos: compliance.enroutePhotoCount,
      routeTasks: compliance.routeTaskCount,
      violations: compliance.violationCount,
      warnings: compliance.warningCount,
    },
    photos: photos.map((photo) => ({
      id: photo.id,
      order: photo.order + 1,
      identifier: photo.identifier,
      fileName: photo.fileName,
      fileSize: photo.fileSize,
      contentHash: photo.contentHash,
      classification: photo.classification,
      linkedWaypoint: photo.linkedWaypoint,
      cameraMetadata: photo.metadata,
      originalMetadata: photo.originalMetadata,
      taskPosition: { latitude: photo.taskLatitude, longitude: photo.taskLongitude },
      cameraAnalysis: photo.analysis,
      taskAnalysis: photo.taskAnalysis,
      status: photoStatus(photo),
      findings: photo.findings,
    })),
    findings: compliance.findings,
  };
}
