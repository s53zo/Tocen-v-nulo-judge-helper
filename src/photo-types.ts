export const PHOTO_CLASSIFICATIONS = [
  'enroute',
  'control-correct',
  'control-false',
  'sign-task',
  'reference',
] as const;

export type PhotoClassification = (typeof PHOTO_CLASSIFICATIONS)[number];
export type MetadataSource = 'exif' | 'manual' | 'example-track' | 'missing';
export type FindingSeverity = 'pass' | 'warning' | 'violation';
export type HeadingReference = 'true' | 'magnetic' | 'unknown';

export interface SourcedValue<T> {
  value: T | null;
  source: MetadataSource;
  reliable: boolean;
  note?: string;
}

export interface PhotoMetadata {
  latitude: SourcedValue<number>;
  longitude: SourcedValue<number>;
  gpsAltitudeMslM: SourcedValue<number>;
  altitudeAglFt: SourcedValue<number>;
  headingDeg: SourcedValue<number>;
  headingReference: SourcedValue<HeadingReference>;
  focalLengthMm: SourcedValue<number>;
  focalLength35Mm: SourcedValue<number>;
  captureTime: SourcedValue<string>;
  orientation: SourcedValue<number>;
  cameraMake: SourcedValue<string>;
  cameraModel: SourcedValue<string>;
  lensModel: SourcedValue<string>;
}

export type RoutePosition = 'before-route' | 'on-route' | 'after-route';

export interface PhotoRouteAnalysis {
  legIndex: number;
  legId: string;
  rawFraction: number;
  fraction: number;
  closestLatitude: number;
  closestLongitude: number;
  lateralDistanceM: number;
  lateralSignedM: number;
  alongRouteM: number;
  distanceOnLegM: number;
  routePosition: RoutePosition;
  nearestControlPoint: string;
  distanceFromNearestControlPointM: number;
  previousControlPoint: string;
  distanceAfterPreviousControlPointM: number;
  legBearingDeg: number;
  headingDifferenceDeg: number | null;
  ambiguousLegIndices: number[];
  manuallySelectedLeg: boolean;
}

export interface PhotoFinding {
  photoId: string | null;
  code: string;
  rule: string;
  severity: FindingSeverity;
  affected: string;
  message: string;
  measured: string;
  permitted: string;
}

export interface GeneratedOrthophotoInfo {
  provider: 'GURS';
  layer: 'DOF025';
  targetLabel: string;
  requestedAt: string;
  coverageWidthM: number;
  coverageHeightM: number;
  modeledAltitudeM: number;
  modeledFocalLength35Mm: number;
  modeledDepressionDeg: number;
  attribution: string;
  targetSource?: {
    provider: 'OpenStreetMap';
    elementId: string;
    category: string;
    featureType: string;
    name: string | null;
    score: number;
    attribution: string;
    selectionSalt?: string;
    controlRole?: 'true' | 'false';
    controlWaypoint?: string;
    correctObjectLatitude?: number;
    correctObjectLongitude?: number;
  };
}

export interface PhotoRecord {
  id: string;
  file: File;
  fileName: string;
  fileSize: number;
  contentHash: string;
  previewUrl: string;
  width: number;
  height: number;
  originalMetadata: PhotoMetadata;
  metadata: PhotoMetadata;
  classification: PhotoClassification;
  identifier: string;
  identifierMixSalt?: string;
  identifierMixAlgorithm?: string;
  linkedWaypoint: string | null;
  taskLatitude: SourcedValue<number>;
  taskLongitude: SourcedValue<number>;
  manualLegIndex: number | null;
  order: number;
  importError: string | null;
  analysis: PhotoRouteAnalysis | null;
  taskAnalysis: PhotoRouteAnalysis | null;
  findings: PhotoFinding[];
  exceptionAccepted: boolean;
  exceptionAcceptedAt: string | null;
  isExample: boolean;
  generatedOrthophoto?: GeneratedOrthophotoInfo;
  fieldErrors?: Record<string, string>;
  fieldDrafts?: Record<string, string>;
}

export interface PhotoComplianceSummary {
  status: 'ok' | 'against-rules' | 'manual-review';
  findings: PhotoFinding[];
  violationCount: number;
  warningCount: number;
  enroutePhotoCount: number;
  routeTaskCount: number;
}

export interface ExamplePhotoManifestItem {
  fileName: string;
  path: string;
  classification: PhotoClassification;
  identifier: string;
  linkedWaypoint: string | null;
  manualOverrides: { latitude: number; longitude: number };
  provenance: { latitude: 'example-track'; longitude: 'example-track' };
}

export interface ExamplePhotoManifest {
  schemaVersion: number;
  label: string;
  description: string;
  route: Array<[string, number, number]>;
  items: ExamplePhotoManifestItem[];
}

export function missingValue<T>(note?: string): SourcedValue<T> {
  return { value: null, source: 'missing', reliable: false, ...(note ? { note } : {}) };
}

export function sourcedValue<T>(
  value: T | null | undefined,
  source: MetadataSource,
  reliable = true,
  note?: string
): SourcedValue<T> {
  if (value === null || value === undefined) return missingValue<T>(note);
  return { value, source, reliable, ...(note ? { note } : {}) };
}
