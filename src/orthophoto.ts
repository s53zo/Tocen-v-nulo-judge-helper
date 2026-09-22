const EARTH_RADIUS_M = 6_371_008.8;
export const MAXIMUM_ORTHOPHOTO_COVERAGE_M = 5000;

export const GURS_DOF025_WMS_URL = 'https://ipi.eprostor.gov.si/wms-si-gurs-dts/wms';
export const GURS_DOF025_LAYER = 'SI.GURS.ZPDZ:DOF025';
export const GURS_DOF025_ATTRIBUTION =
  'Geodetska uprava Republike Slovenije, DOF025, current mosaic (2023-2025)';

export interface OrthophotoCaptureModel {
  altitudeM: number;
  focalLength35Mm: number;
  depressionDeg: number;
  aspectRatio: number;
}

export interface OrthophotoCoverage {
  widthM: number;
  heightM: number;
}

export interface OrthophotoTarget {
  label: string;
  latitude: number;
  longitude: number;
  captureModel?: OrthophotoCaptureModel;
  alongRouteM?: number;
  source?: OrthophotoTargetSource;
  control?: {
    classification: 'control-correct' | 'control-false';
    waypoint: string;
    identifier: string;
  };
}

export interface OrthophotoTargetSource {
  provider: 'OpenStreetMap';
  elementId: string;
  category: string;
  featureType: string;
  name: string | null;
  score: number;
  attribution: string;
  providerCategory?: string;
  diversityGroup?: string;
  selectionSalt?: string;
  controlRole?: 'true' | 'false';
  controlWaypoint?: string;
  correctObjectLatitude?: number;
  correctObjectLongitude?: number;
}

export function orthophotoCoverage(model: OrthophotoCaptureModel): OrthophotoCoverage {
  const { altitudeM, focalLength35Mm, depressionDeg, aspectRatio } = model;
  if (!Number.isFinite(altitudeM) || altitudeM <= 0) throw new Error('Orthophoto altitude must be positive.');
  if (!Number.isFinite(focalLength35Mm) || focalLength35Mm <= 0) {
    throw new Error('Orthophoto focal length must be positive.');
  }
  if (!Number.isFinite(aspectRatio) || aspectRatio <= 0) {
    throw new Error('Orthophoto aspect ratio must be positive.');
  }
  if (!Number.isFinite(depressionDeg) || depressionDeg <= 0 || depressionDeg > 90) {
    throw new Error('Orthophoto depression angle must be greater than 0 and at most 90 degrees.');
  }

  const depression = (depressionDeg * Math.PI) / 180;
  const horizontalTangent = 18 / focalLength35Mm;
  const verticalTangent = 18 / aspectRatio / focalLength35Mm;
  const corners: Array<{ forward: number; lateral: number }> = [];
  for (const verticalSign of [-1, 1]) {
    for (const horizontalSign of [-1, 1]) {
      const rayForward = Math.cos(depression) + verticalSign * verticalTangent * Math.sin(depression);
      const rayDown = Math.sin(depression) - verticalSign * verticalTangent * Math.cos(depression);
      if (rayDown <= 0) {
        throw new Error('The selected view angle includes the horizon; increase depression or focal length.');
      }
      const scale = altitudeM / rayDown;
      corners.push({
        forward: scale * rayForward,
        lateral: scale * horizontalSign * horizontalTangent,
      });
    }
  }
  const forward = corners.map((corner) => corner.forward);
  const lateral = corners.map((corner) => corner.lateral);
  const coverage = {
    widthM: Math.max(...lateral) - Math.min(...lateral),
    heightM: Math.max(...forward) - Math.min(...forward),
  };
  if (
    !Number.isFinite(coverage.widthM) ||
    !Number.isFinite(coverage.heightM) ||
    coverage.widthM > MAXIMUM_ORTHOPHOTO_COVERAGE_M ||
    coverage.heightM > MAXIMUM_ORTHOPHOTO_COVERAGE_M
  ) {
    throw new Error(
      `The modeled footprint exceeds ${MAXIMUM_ORTHOPHOTO_COVERAGE_M.toLocaleString()} m; increase focal length or depression angle, or reduce height.`
    );
  }
  return coverage;
}

export function orthophotoRequestUrl(
  latitude: number,
  longitude: number,
  coverage: OrthophotoCoverage,
  maximumPixelDimension = 1600
): string {
  if (!Number.isFinite(latitude) || latitude < -90 || latitude > 90) {
    throw new Error('Orthophoto latitude is invalid.');
  }
  if (!Number.isFinite(longitude) || longitude < -180 || longitude > 180) {
    throw new Error('Orthophoto longitude is invalid.');
  }
  if (
    !Number.isFinite(coverage.widthM) ||
    !Number.isFinite(coverage.heightM) ||
    coverage.widthM <= 0 ||
    coverage.heightM <= 0 ||
    coverage.widthM > MAXIMUM_ORTHOPHOTO_COVERAGE_M ||
    coverage.heightM > MAXIMUM_ORTHOPHOTO_COVERAGE_M
  ) {
    throw new Error('Orthophoto coverage is outside the supported range.');
  }
  const halfLatitudeDeg = (coverage.heightM / 2 / EARTH_RADIUS_M) * (180 / Math.PI);
  const halfLongitudeDeg =
    (coverage.widthM / 2 / (EARTH_RADIUS_M * Math.cos((latitude * Math.PI) / 180))) * (180 / Math.PI);
  const aspect = coverage.widthM / coverage.heightM;
  const width = aspect >= 1 ? maximumPixelDimension : Math.max(1, Math.round(maximumPixelDimension * aspect));
  const height =
    aspect >= 1 ? Math.max(1, Math.round(maximumPixelDimension / aspect)) : maximumPixelDimension;
  const parameters = new URLSearchParams({
    SERVICE: 'WMS',
    VERSION: '1.3.0',
    REQUEST: 'GetMap',
    LAYERS: GURS_DOF025_LAYER,
    STYLES: 'dof025',
    CRS: 'CRS:84',
    BBOX: [
      longitude - halfLongitudeDeg,
      latitude - halfLatitudeDeg,
      longitude + halfLongitudeDeg,
      latitude + halfLatitudeDeg,
    ].join(','),
    WIDTH: String(width),
    HEIGHT: String(height),
    FORMAT: 'image/jpeg',
  });
  return `${GURS_DOF025_WMS_URL}?${parameters}`;
}
