import { haversine, type Route, type Waypoint } from './domain';
import type { OrthophotoTarget, OrthophotoTargetSource } from './orthophoto';
import { analyzePhotoPosition } from './photo-analysis';

export const OVERPASS_API_URL = 'https://overpass-api.de/api/interpreter';
export const OVERPASS_FALLBACK_URL = 'https://overpass.private.coffee/api/interpreter';
export const OVERPASS_SECOND_FALLBACK_URL = 'https://maps.mail.ru/osm/tools/overpass/api/interpreter';
export const OSM_ATTRIBUTION = '© OpenStreetMap contributors';

const ROUTE_CORRIDOR_M = 250;
const CONTROL_CLEARANCE_M = 1852;
const LEG_END_MARGIN_M = 100;
const MINIMUM_TARGET_SPACING_M = 500;
const SPATIAL_DEDUPLICATION_M = 70;
const ENRICHMENT_TIME_BUDGET_MS = 210_000;
const MAXIMUM_ROUTE_LEGS = 50;
const MAXIMUM_CORRIDOR_BOXES = 120;
const MAXIMUM_OVERPASS_RESPONSE_BYTES = 15 * 1024 * 1024;
const MAXIMUM_OVERPASS_ELEMENTS = 50_000;
const MAXIMUM_GEOMETRY_POINTS = 10_000;
const MAXIMUM_OSM_TEXT_LENGTH = 500;
const CONTROL_TRUE_SEARCH_RADIUS_M = 3000;
const CONTROL_FALSE_MINIMUM_M = 1852;
const CONTROL_FALSE_MAXIMUM_M = 10 * 1852;
const CONTROL_DISCOVERY_DEADLINE_MS = 45_000;
const PRIMARY_OVERPASS_ATTEMPTS = [[OVERPASS_SECOND_FALLBACK_URL, 18_000]] as const;
const RETRY_OVERPASS_ATTEMPTS = [
  [OVERPASS_FALLBACK_URL, 22_000],
  [OVERPASS_API_URL, 22_000],
  [OVERPASS_SECOND_FALLBACK_URL, 35_000],
] as const;

export function createSeededRandom(salt: string): () => number {
  let seed = 0x811c9dc5;
  for (const character of salt) {
    seed ^= character.charCodeAt(0);
    seed = Math.imul(seed, 0x01000193);
  }
  return () => {
    seed += 0x6d2b79f5;
    let value = seed;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

interface OsmCoordinate {
  lat: number;
  lon: number;
}

export interface OverpassElement {
  type: 'node' | 'way' | 'relation';
  id: number;
  lat?: number;
  lon?: number;
  center?: OsmCoordinate;
  nodes?: number[];
  geometry?: OsmCoordinate[];
  tags?: Record<string, string>;
}

export interface OverpassResponse {
  elements: OverpassElement[];
  remark?: string;
  warnings?: string[];
  failedRequests?: OsmFailedRequest[];
}

export type OsmDiscoveryStage = 'features' | 'roads' | 'buildings';

export interface OsmFailedRequest {
  legIndex: number;
  stage: OsmDiscoveryStage;
  query: string;
  detail: string;
}
export type OsmDiscoveryStageState =
  | 'started'
  | 'completed'
  | 'warning'
  | 'skipped'
  | 'retrying'
  | 'recovered';

export interface OsmDiscoveryProgress {
  legIndex: number;
  legCount: number;
  legName: string;
  stage: OsmDiscoveryStage;
  state: OsmDiscoveryStageState;
  completedSteps: number;
  totalSteps: number;
  detail?: string;
}

export interface FetchOverpassOptions {
  route?: Route;
  requestedCount?: number;
  splitAfterM?: number | null;
  signal?: AbortSignal;
  fetcher?: typeof fetch;
  onProgress?: (progress: OsmDiscoveryProgress) => void;
  deadlineMs?: number;
  retryDelayMs?: number;
}

export interface OsmPhotoCandidate extends OrthophotoTarget {
  id: string;
  name: string;
  category: string;
  featureType: string;
  score: number;
  confidence: 'excellent' | 'strong' | 'good' | 'fallback';
  lateralDistanceM: number;
  previousControlPoint: string;
  distanceAfterControlM: number;
  routeLegIndex: number;
  routeLegName: string;
  routeLegCount: number;
  routeLegBoundariesM: number[];
  source: OrthophotoTargetSource;
}

export interface ControlPhotoCandidate extends OrthophotoTarget {
  id: string;
  name: string;
  category: string;
  featureType: string;
  score: number;
  distanceFromWaypointM: number;
  distanceFromCorrectM: number;
  source: NonNullable<OrthophotoTarget['source']>;
}

export interface ControlPhotoProposal {
  waypoint: Waypoint;
  trueTarget: ControlPhotoCandidate;
  falseTarget: ControlPhotoCandidate | null;
}

export interface ControlDiscoveryProgress {
  waypointIndex: number;
  waypointCount: number;
  waypointName: string;
  stage: 'true' | 'false';
  state: OsmDiscoveryStageState;
  detail?: string;
}

export interface FetchControlPhotoOptions {
  signal?: AbortSignal;
  fetcher?: typeof fetch;
  retryDelayMs?: number;
  waypointIndices?: number[];
  deadlineMs?: number;
  onProgress?: (progress: ControlDiscoveryProgress) => void;
}

interface CandidateDefinition {
  category: string;
  featureType: string;
  score: number;
  name?: string;
}

function coreFeatureFilters(): string[] {
  return [
    `nwr["place"~"^(village|hamlet|isolated_dwelling|neighbourhood)$"]`,
    `nwr["bridge"]`,
    `nwr["railway"="level_crossing"]`,
    `nwr["natural"="water"]`,
    `nwr["waterway"~"^(river|canal|stream)$"]`,
    `nwr["amenity"~"^(place_of_worship|school|fire_station|townhall)$"]`,
    `nwr["leisure"~"^(pitch|sports_centre)$"]`,
    `nwr["landuse"~"^(cemetery|quarry|industrial)$"]`,
    `nwr["historic"~"^(castle|monument)$"]`,
    `nwr["power"="substation"]`,
    `nwr["man_made"~"^(silo|wastewater_plant|storage_tank)$"]`,
  ];
}

export function buildOverpassCoreQuery(points: Waypoint[]): string {
  if (points.length < 2) throw new Error('A route with at least two waypoints is required.');
  if (points.length - 1 > MAXIMUM_ROUTE_LEGS) {
    throw new Error(`OpenStreetMap discovery supports at most ${MAXIMUM_ROUTE_LEGS} route legs.`);
  }
  const boxes = corridorBoundingBoxes(points);
  if (boxes.length > MAXIMUM_CORRIDOR_BOXES) {
    throw new Error('The route is too long or complex for a bounded OpenStreetMap corridor search.');
  }
  const statements = boxes
    .flatMap((box) => coreFeatureFilters().map((filter) => `${filter}(${box});`))
    .join('');
  return `[out:json][timeout:30];(${statements});out body center;`;
}

function corridorBoundingBoxes(points: Waypoint[]): string[] {
  const boxes: string[] = [];
  for (let legIndex = 0; legIndex < points.length - 1; legIndex += 1) {
    const [, fromLatitude, fromLongitude] = points[legIndex];
    const [, toLatitude, toLongitude] = points[legIndex + 1];
    const segments = Math.max(
      1,
      Math.ceil(haversine(fromLatitude, fromLongitude, toLatitude, toLongitude) / 4000)
    );
    for (let segment = 0; segment < segments; segment += 1) {
      const start = segment / segments;
      const end = (segment + 1) / segments;
      const startLatitude = fromLatitude + (toLatitude - fromLatitude) * start;
      const startLongitude = fromLongitude + (toLongitude - fromLongitude) * start;
      const endLatitude = fromLatitude + (toLatitude - fromLatitude) * end;
      const endLongitude = fromLongitude + (toLongitude - fromLongitude) * end;
      const latitudeMargin = 300 / 111_320;
      const averageLatitude = (startLatitude + endLatitude) / 2;
      const longitudeMargin = latitudeMargin / Math.max(0.2, Math.cos((averageLatitude * Math.PI) / 180));
      boxes.push(
        [
          Math.min(startLatitude, endLatitude) - latitudeMargin,
          Math.min(startLongitude, endLongitude) - longitudeMargin,
          Math.max(startLatitude, endLatitude) + latitudeMargin,
          Math.max(startLongitude, endLongitude) + longitudeMargin,
        ]
          .map((value) => value.toFixed(6))
          .join(',')
      );
    }
  }
  return [...new Set(boxes)];
}

function straightLegCorridorPolygon(points: Waypoint[], legIndex: number): string {
  const [, fromLatitude, fromLongitude] = points[legIndex];
  const [, toLatitude, toLongitude] = points[legIndex + 1];
  const averageLatitude = (fromLatitude + toLatitude) / 2;
  const metersPerLatitudeDegree = 111_320;
  const metersPerLongitudeDegree = Math.max(
    metersPerLatitudeDegree * Math.cos((averageLatitude * Math.PI) / 180),
    metersPerLatitudeDegree * 0.2
  );
  const eastM = (toLongitude - fromLongitude) * metersPerLongitudeDegree;
  const northM = (toLatitude - fromLatitude) * metersPerLatitudeDegree;
  const lengthM = Math.hypot(eastM, northM);
  if (lengthM === 0) throw new Error('A route leg cannot have identical endpoints.');
  const directionEast = eastM / lengthM;
  const directionNorth = northM / lengthM;
  const perpendicularEast = -directionNorth;
  const perpendicularNorth = directionEast;
  const marginM = 300;
  const cornersM = [
    {
      east: -directionEast * marginM + perpendicularEast * marginM,
      north: -directionNorth * marginM + perpendicularNorth * marginM,
    },
    {
      east: -directionEast * marginM - perpendicularEast * marginM,
      north: -directionNorth * marginM - perpendicularNorth * marginM,
    },
    {
      east: eastM + directionEast * marginM - perpendicularEast * marginM,
      north: northM + directionNorth * marginM - perpendicularNorth * marginM,
    },
    {
      east: eastM + directionEast * marginM + perpendicularEast * marginM,
      north: northM + directionNorth * marginM + perpendicularNorth * marginM,
    },
  ];
  return cornersM
    .map(
      ({ east, north }) =>
        `${(fromLatitude + north / metersPerLatitudeDegree).toFixed(6)} ${(fromLongitude + east / metersPerLongitudeDegree).toFixed(6)}`
    )
    .join(' ');
}

export function buildOverpassLegQueries(
  points: Waypoint[],
  legIndex: number
): Record<OsmDiscoveryStage, string> {
  if (points.length < 2) throw new Error('A route with at least two waypoints is required.');
  if (!Number.isInteger(legIndex) || legIndex < 0 || legIndex >= points.length - 1) {
    throw new Error('The requested route leg does not exist.');
  }
  const corridor = `poly:"${straightLegCorridorPolygon(points, legIndex)}"`;
  const featureFilters = coreFeatureFilters();
  const featureStatements = featureFilters.map((filter) => `${filter}(${corridor})`);
  const roadStatements = [`way["highway"~"^(secondary|tertiary|unclassified|residential)$"](${corridor})`];
  const buildingStatements = [
    `way["building"~"^(house|apartments|terrace|farm|church|chapel|school|industrial|warehouse|public|retail|commercial)$"](${corridor})`,
  ];
  return {
    features: `[out:json][timeout:30];(${featureStatements.map((statement) => `${statement};`).join('')});out body center;`,
    roads: `[out:json][timeout:30];(${roadStatements.map((statement) => `${statement};`).join('')});out body geom;`,
    buildings: `[out:json][timeout:30];(${buildingStatements.map((statement) => `${statement};`).join('')});out body center;`,
  };
}

export function buildOverpassPhotoQueries(points: Waypoint[]): string[] {
  return points.slice(0, -1).flatMap((_, legIndex) => {
    const queries = buildOverpassLegQueries(points, legIndex);
    return [queries.features, queries.roads, queries.buildings];
  });
}

function validCoordinate(value: unknown, minimum: number, maximum: number): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= minimum && value <= maximum;
}

export function validateOverpassResponse(value: unknown): OverpassResponse {
  if (!value || typeof value !== 'object') throw new Error('invalid response object');
  const raw = value as { elements?: unknown; remark?: unknown };
  if (!Array.isArray(raw.elements)) throw new Error('invalid response elements');
  if (raw.elements.length > MAXIMUM_OVERPASS_ELEMENTS) {
    throw new Error(`response exceeds ${MAXIMUM_OVERPASS_ELEMENTS.toLocaleString()} elements`);
  }
  const elements: OverpassElement[] = raw.elements.map((item) => {
    if (!item || typeof item !== 'object') throw new Error('invalid OpenStreetMap element');
    const element = item as Record<string, unknown>;
    if (!['node', 'way', 'relation'].includes(String(element.type))) {
      throw new Error('invalid OpenStreetMap element type');
    }
    if (!Number.isSafeInteger(element.id)) throw new Error('invalid OpenStreetMap element id');
    if (element.lat !== undefined && !validCoordinate(element.lat, -90, 90)) {
      throw new Error('invalid OpenStreetMap latitude');
    }
    if (element.lon !== undefined && !validCoordinate(element.lon, -180, 180)) {
      throw new Error('invalid OpenStreetMap longitude');
    }
    const center = element.center as Record<string, unknown> | undefined;
    if (center && (!validCoordinate(center.lat, -90, 90) || !validCoordinate(center.lon, -180, 180))) {
      throw new Error('invalid OpenStreetMap centre');
    }
    const geometry = element.geometry as Array<Record<string, unknown>> | undefined;
    if (geometry) {
      if (!Array.isArray(geometry) || geometry.length > MAXIMUM_GEOMETRY_POINTS) {
        throw new Error('OpenStreetMap geometry exceeds the supported size');
      }
      for (const point of geometry) {
        if (!validCoordinate(point?.lat, -90, 90) || !validCoordinate(point?.lon, -180, 180)) {
          throw new Error('invalid OpenStreetMap geometry coordinate');
        }
      }
    }
    const tags = element.tags as Record<string, unknown> | undefined;
    if (tags) {
      if (typeof tags !== 'object' || Array.isArray(tags)) throw new Error('invalid OpenStreetMap tags');
      for (const [key, tagValue] of Object.entries(tags)) {
        if (
          key.length > MAXIMUM_OSM_TEXT_LENGTH ||
          typeof tagValue !== 'string' ||
          tagValue.length > MAXIMUM_OSM_TEXT_LENGTH
        ) {
          throw new Error('OpenStreetMap tag text exceeds the supported size');
        }
      }
    }
    const nodes = element.nodes as unknown[] | undefined;
    if (nodes && (!Array.isArray(nodes) || nodes.length > MAXIMUM_GEOMETRY_POINTS)) {
      throw new Error('OpenStreetMap node list exceeds the supported size');
    }
    return item as OverpassElement;
  });
  const remark = typeof raw.remark === 'string' ? raw.remark.slice(0, MAXIMUM_OSM_TEXT_LENGTH) : undefined;
  return { elements, ...(remark ? { remark } : {}) };
}

async function boundedOverpassJson(response: Response, signal: AbortSignal): Promise<OverpassResponse> {
  const declaredLength = Number(response.headers.get('content-length'));
  if (Number.isFinite(declaredLength) && declaredLength > MAXIMUM_OVERPASS_RESPONSE_BYTES) {
    throw new Error('OpenStreetMap response exceeds the 15 MB limit');
  }
  if (!response.body) {
    const text = await response.text();
    if (new TextEncoder().encode(text).byteLength > MAXIMUM_OVERPASS_RESPONSE_BYTES) {
      throw new Error('OpenStreetMap response exceeds the 15 MB limit');
    }
    return validateOverpassResponse(JSON.parse(text));
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let text = '';
  let total = 0;
  try {
    while (true) {
      if (signal.aborted) throw abortMessage(signal);
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > MAXIMUM_OVERPASS_RESPONSE_BYTES) {
        throw new Error('OpenStreetMap response exceeds the 15 MB limit');
      }
      text += decoder.decode(value, { stream: true });
    }
    text += decoder.decode();
  } finally {
    if (signal.aborted || total > MAXIMUM_OVERPASS_RESPONSE_BYTES) {
      await reader.cancel().catch(() => undefined);
    }
    reader.releaseLock();
  }
  return validateOverpassResponse(JSON.parse(text));
}

function abortMessage(signal: AbortSignal): Error {
  return new Error(
    signal.reason instanceof Error ? signal.reason.message : 'OpenStreetMap discovery cancelled.'
  );
}

async function requestOverpass(
  query: string,
  signal: AbortSignal | undefined,
  fetcher: typeof fetch,
  attempts: ReadonlyArray<readonly [string, number]>
): Promise<OverpassResponse> {
  const controllers: AbortController[] = [];
  const requestEndpoint = async (endpoint: string, requestTimeoutMs: number): Promise<OverpassResponse> => {
    if (signal?.aborted) throw abortMessage(signal);
    const controller = new AbortController();
    controllers.push(controller);
    const abort = () => controller.abort();
    signal?.addEventListener('abort', abort, { once: true });
    let timeout: ReturnType<typeof setTimeout> | null = null;
    try {
      timeout = globalThis.setTimeout(() => controller.abort(), requestTimeoutMs);
      const response = await fetcher(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8' },
        body: new URLSearchParams({ data: query }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await boundedOverpassJson(response, controller.signal);
      if (data.remark) throw new Error(data.remark);
      return data;
    } catch (error) {
      if (signal?.aborted) throw abortMessage(signal);
      throw controller.signal.aborted ? new Error(`${endpoint}: request timed out`) : error;
    } finally {
      if (timeout !== null) globalThis.clearTimeout(timeout);
      signal?.removeEventListener('abort', abort);
    }
  };
  const failures: string[] = [];
  try {
    // Exact-query diagnostics showed that concurrent requests make the healthy
    // public instance queue one leg until it times out. Try it alone first; only
    // contact the less reliable alternatives after a real failure.
    for (const [endpoint, requestTimeoutMs] of attempts) {
      try {
        return await requestEndpoint(endpoint, requestTimeoutMs);
      } catch (error) {
        if (signal?.aborted) throw abortMessage(signal);
        failures.push(error instanceof Error ? error.message : String(error));
      }
    }
    throw new Error(`OpenStreetMap query failed on all public endpoints: ${failures.join('; ')}`);
  } finally {
    for (const controller of controllers) controller.abort();
  }
}

export async function fetchOverpassPhotoData(
  points: Waypoint[],
  options: FetchOverpassOptions = {}
): Promise<OverpassResponse> {
  if (points.length < 2) throw new Error('A route with at least two waypoints is required.');
  if (points.length - 1 > MAXIMUM_ROUTE_LEGS) {
    throw new Error(`OpenStreetMap discovery supports at most ${MAXIMUM_ROUTE_LEGS} route legs.`);
  }
  const route = options.route;
  const fetcher = options.fetcher ?? fetch;
  const requestedCount = options.requestedCount ?? 6;
  const discoveryStartedAt = Date.now();
  const discoveryController = new AbortController();
  let deadlineReached = false;
  const abortFromCaller = () => discoveryController.abort(options.signal?.reason);
  options.signal?.addEventListener('abort', abortFromCaller, { once: true });
  const deadline = globalThis.setTimeout(() => {
    deadlineReached = true;
    discoveryController.abort(new Error('OpenStreetMap discovery reached its time limit.'));
  }, options.deadlineMs ?? 240_000);
  const discoverySignal = discoveryController.signal;
  const legCount = points.length - 1;
  let totalSteps = legCount * 3;
  let completedSteps = 0;
  const warnings: string[] = [];
  const failedRequests: OsmFailedRequest[] = [];
  const elements = new Map<string, OverpassElement>();
  const emit = (
    legIndex: number,
    stage: OsmDiscoveryStage,
    state: OsmDiscoveryStageState,
    detail?: string
  ) => {
    options.onProgress?.({
      legIndex,
      legCount,
      legName: `${points[legIndex][0]}–${points[legIndex + 1][0]}`,
      stage,
      state,
      completedSteps,
      totalSteps,
      ...(detail ? { detail } : {}),
    });
  };
  const runStage = async (
    legIndex: number,
    stage: OsmDiscoveryStage,
    query: string,
    attempts: ReadonlyArray<readonly [string, number]>,
    retry = false
  ): Promise<string | null> => {
    emit(legIndex, stage, retry ? 'retrying' : 'started');
    try {
      const result = await requestOverpass(query, discoverySignal, fetcher, attempts);
      for (const element of result.elements) elements.set(`${element.type}/${element.id}`, element);
      completedSteps += 1;
      emit(legIndex, stage, retry ? 'recovered' : 'completed');
      return null;
    } catch (error) {
      if (options.signal?.aborted) throw error;
      const detail = error instanceof Error ? error.message : String(error);
      completedSteps += 1;
      emit(legIndex, stage, 'warning', detail);
      return detail;
    }
  };

  const runLegsWithDeferredRetry = async (stage: OsmDiscoveryStage): Promise<void> => {
    let nextLeg = 0;
    const failed: Array<{ legIndex: number; query: string; detail: string }> = [];
    const worker = async () => {
      while (nextLeg < legCount && !discoverySignal.aborted) {
        const legIndex = nextLeg;
        nextLeg += 1;
        const query = buildOverpassLegQueries(points, legIndex)[stage];
        const detail = await runStage(legIndex, stage, query, PRIMARY_OVERPASS_ATTEMPTS);
        if (detail) failed.push({ legIndex, query, detail });
      }
    };
    await worker();
    if (failed.length === 0) return;
    if (discoverySignal.aborted) {
      warnings.push(
        ...failed.map(
          ({ legIndex, detail }) => `${points[legIndex][0]}–${points[legIndex + 1][0]} ${stage}: ${detail}`
        )
      );
      return;
    }
    const retryDelayMs = options.retryDelayMs ?? 900;
    if (retryDelayMs > 0) {
      await new Promise<void>((resolve) => globalThis.setTimeout(resolve, retryDelayMs));
    }
    let remaining = failed;
    for (const retryAttempt of RETRY_OVERPASS_ATTEMPTS) {
      if (remaining.length === 0 || discoverySignal.aborted) break;
      totalSteps += remaining.length;
      const nextRound: typeof remaining = [];
      for (const failedRequest of remaining) {
        if (discoverySignal.aborted) {
          nextRound.push(failedRequest);
          continue;
        }
        const retryDetail = await runStage(
          failedRequest.legIndex,
          stage,
          failedRequest.query,
          [retryAttempt],
          true
        );
        if (retryDetail) nextRound.push({ ...failedRequest, detail: retryDetail });
      }
      remaining = nextRound;
    }
    warnings.push(
      ...remaining.map(
        ({ legIndex, detail }) => `${points[legIndex][0]}–${points[legIndex + 1][0]} ${stage}: ${detail}`
      )
    );
    failedRequests.push(...remaining.map((failure) => ({ ...failure, stage })));
  };

  const skipStage = (stage: OsmDiscoveryStage, detail: string) => {
    for (let legIndex = 0; legIndex < legCount; legIndex += 1) {
      completedSteps += 1;
      emit(legIndex, stage, 'skipped', detail);
    }
  };

  try {
    // Start with bounded leg queries. A single route-wide query can monopolize the
    // complete discovery budget on a busy public Overpass instance and leave no
    // completed work to retain at the deadline.
    await runLegsWithDeferredRetry('features');

    const enoughCandidates = () => {
      if (!route) return false;
      const candidates = discoverOsmPhotoCandidates({ elements: [...elements.values()] }, route, points);
      return (
        selectOsmPhotoCandidates(candidates, requestedCount, options.splitAfterM ?? null, () => 0.5)
          .length === requestedCount
      );
    };

    if (enoughCandidates()) {
      skipStage('roads', 'Enough stronger targets were already found.');
      skipStage('buildings', 'Enough stronger targets were already found.');
    } else if (deadlineReached || Date.now() - discoveryStartedAt >= ENRICHMENT_TIME_BUDGET_MS) {
      skipStage('roads', 'Skipped after the bounded core-feature search time was exhausted.');
      skipStage('buildings', 'Skipped after the bounded enrichment search time was exhausted.');
    } else {
      await runLegsWithDeferredRetry('roads');
      if (enoughCandidates()) {
        skipStage('buildings', 'Enough feature and road targets were found.');
      } else if (deadlineReached || Date.now() - discoveryStartedAt >= ENRICHMENT_TIME_BUDGET_MS) {
        skipStage('buildings', 'Skipped after the bounded enrichment search time was exhausted.');
      } else {
        await runLegsWithDeferredRetry('buildings');
      }
    }
    if (deadlineReached) {
      warnings.push('The discovery time limit was reached; completed leg results were retained.');
    }
    return {
      elements: [...elements.values()],
      ...(warnings.length ? { warnings } : {}),
      ...(failedRequests.length ? { failedRequests } : {}),
    };
  } finally {
    globalThis.clearTimeout(deadline);
    options.signal?.removeEventListener('abort', abortFromCaller);
  }
}

export async function retryOverpassPhotoRequests(
  points: Waypoint[],
  requests: OsmFailedRequest[],
  options: FetchOverpassOptions = {}
): Promise<OverpassResponse> {
  if (points.length < 2) throw new Error('A route with at least two waypoints is required.');
  if (requests.length === 0) return { elements: [] };
  if (
    requests.length > (points.length - 1) * 3 ||
    requests.some(
      ({ legIndex, stage, query }) =>
        !Number.isInteger(legIndex) ||
        legIndex < 0 ||
        legIndex >= points.length - 1 ||
        !['features', 'roads', 'buildings'].includes(stage) ||
        !query ||
        query.length > 100_000
    )
  ) {
    throw new Error('The failed OpenStreetMap request list is invalid.');
  }
  const fetcher = options.fetcher ?? fetch;
  const controller = new AbortController();
  const abortFromCaller = () => controller.abort(options.signal?.reason);
  options.signal?.addEventListener('abort', abortFromCaller, { once: true });
  const timeout = globalThis.setTimeout(
    () => controller.abort(new Error('OpenStreetMap retry reached its time limit.')),
    options.deadlineMs ?? 180_000
  );
  const elements = new Map<string, OverpassElement>();
  const failedRequests: OsmFailedRequest[] = [];
  let completedSteps = 0;
  try {
    for (const request of requests) {
      options.onProgress?.({
        legIndex: request.legIndex,
        legCount: points.length - 1,
        legName: `${points[request.legIndex][0]}–${points[request.legIndex + 1][0]}`,
        stage: request.stage,
        state: 'retrying',
        completedSteps,
        totalSteps: requests.length,
      });
      try {
        const result = await requestOverpass(
          request.query,
          controller.signal,
          fetcher,
          RETRY_OVERPASS_ATTEMPTS
        );
        for (const element of result.elements) elements.set(`${element.type}/${element.id}`, element);
        completedSteps += 1;
        options.onProgress?.({
          legIndex: request.legIndex,
          legCount: points.length - 1,
          legName: `${points[request.legIndex][0]}–${points[request.legIndex + 1][0]}`,
          stage: request.stage,
          state: 'recovered',
          completedSteps,
          totalSteps: requests.length,
        });
      } catch (error) {
        if (options.signal?.aborted) throw error;
        const detail = error instanceof Error ? error.message : String(error);
        failedRequests.push({ ...request, detail });
        completedSteps += 1;
        options.onProgress?.({
          legIndex: request.legIndex,
          legCount: points.length - 1,
          legName: `${points[request.legIndex][0]}–${points[request.legIndex + 1][0]}`,
          stage: request.stage,
          state: 'warning',
          completedSteps,
          totalSteps: requests.length,
          detail,
        });
      }
    }
    const warnings = failedRequests.map(
      ({ legIndex, stage, detail }) => `${points[legIndex][0]}–${points[legIndex + 1][0]} ${stage}: ${detail}`
    );
    return {
      elements: [...elements.values()],
      ...(warnings.length ? { warnings } : {}),
      ...(failedRequests.length ? { failedRequests } : {}),
    };
  } finally {
    globalThis.clearTimeout(timeout);
    options.signal?.removeEventListener('abort', abortFromCaller);
  }
}

function elementCoordinate(element: OverpassElement): OsmCoordinate | null {
  if (Number.isFinite(element.lat) && Number.isFinite(element.lon)) {
    return { lat: element.lat as number, lon: element.lon as number };
  }
  if (element.center && Number.isFinite(element.center.lat) && Number.isFinite(element.center.lon)) {
    return element.center;
  }
  const geometry = element.geometry?.filter(({ lat, lon }) => Number.isFinite(lat) && Number.isFinite(lon));
  if (!geometry?.length) return null;
  return {
    lat: geometry.reduce((sum, point) => sum + point.lat, 0) / geometry.length,
    lon: geometry.reduce((sum, point) => sum + point.lon, 0) / geometry.length,
  };
}

function candidateDefinition(tags: Record<string, string>): CandidateDefinition | null {
  if (tags.railway === 'level_crossing') {
    return { category: 'rail-crossing', featureType: 'Road–rail level crossing', score: 99 };
  }
  if (tags.bridge && tags.bridge !== 'no') {
    return { category: 'bridge', featureType: 'Bridge', score: 95, name: tags.name };
  }
  if (tags.historic === 'castle') {
    return { category: 'landmark', featureType: 'Castle', score: 98, name: tags.name };
  }
  if (tags.amenity === 'place_of_worship' || ['church', 'chapel'].includes(tags.building)) {
    return { category: 'landmark', featureType: tags.building === 'chapel' ? 'Chapel' : 'Church', score: 96 };
  }
  if (tags.leisure === 'pitch') {
    return { category: 'sports', featureType: 'Sports pitch', score: 94 };
  }
  if (tags.landuse === 'cemetery') {
    return { category: 'cemetery', featureType: 'Cemetery', score: 93 };
  }
  if (tags.landuse === 'quarry') {
    return { category: 'quarry', featureType: 'Quarry', score: 92 };
  }
  if (tags.man_made === 'wastewater_plant') {
    return { category: 'infrastructure', featureType: 'Wastewater plant', score: 91 };
  }
  if (tags.power === 'substation') {
    return { category: 'infrastructure', featureType: 'Power substation', score: 90 };
  }
  if (tags.power === 'plant') {
    return { category: 'infrastructure', featureType: 'Power plant', score: 88 };
  }
  if (tags.power === 'generator') {
    return { category: 'infrastructure', featureType: 'Power generator', score: 86 };
  }
  if (tags.man_made === 'tower') {
    return { category: 'landmark', featureType: 'Tower', score: 90, name: tags.name };
  }
  if (tags.man_made === 'watermill') {
    return { category: 'landmark', featureType: 'Watermill', score: 88, name: tags.name };
  }
  if (tags.man_made === 'windmill') {
    return { category: 'landmark', featureType: 'Windmill', score: 88, name: tags.name };
  }
  if (tags.man_made === 'lighthouse') {
    return { category: 'landmark', featureType: 'Lighthouse', score: 90, name: tags.name };
  }
  if (['breakwater', 'pier'].includes(tags.man_made)) {
    return {
      category: 'infrastructure',
      featureType: tags.man_made === 'pier' ? 'Pier' : 'Breakwater',
      score: 82,
    };
  }
  if (tags.man_made === 'mine') {
    return { category: 'landmark', featureType: 'Mine', score: 84, name: tags.name };
  }
  if (tags.man_made === 'silo' || tags.man_made === 'storage_tank') {
    return {
      category: 'infrastructure',
      featureType: tags.man_made === 'silo' ? 'Silo' : 'Storage tank',
      score: 89,
    };
  }
  if (tags.amenity === 'school' || tags.building === 'school') {
    return { category: 'public-building', featureType: 'School', score: 88 };
  }
  if (['community_centre', 'hospital'].includes(tags.amenity)) {
    return {
      category: 'public-building',
      featureType: tags.amenity === 'hospital' ? 'Hospital' : 'Community centre',
      score: tags.amenity === 'hospital' ? 84 : 78,
    };
  }
  if (tags.amenity === 'fire_station') {
    return { category: 'public-building', featureType: 'Fire station', score: 87 };
  }
  if (tags.natural === 'water') {
    return { category: 'water', featureType: 'Lake or pond', score: 86 };
  }
  if (tags.leisure === 'sports_centre') {
    return { category: 'sports', featureType: 'Sports centre', score: 85 };
  }
  if (tags.tourism === 'museum') {
    return { category: 'landmark', featureType: 'Museum', score: 84, name: tags.name };
  }
  if (tags.aeroway === 'aerodrome') {
    return { category: 'aviation', featureType: 'Airfield', score: 94, name: tags.name };
  }
  if (tags.waterway === 'river') {
    return { category: 'water', featureType: 'River', score: tags.name ? 82 : 76 };
  }
  if (tags.historic === 'monument') {
    return { category: 'landmark', featureType: 'Monument', score: 80 };
  }
  if (['monastery', 'fort', 'manor', 'ruins'].includes(tags.historic)) {
    return {
      category: 'landmark',
      featureType: tags.historic[0].toUpperCase() + tags.historic.slice(1),
      score: tags.historic === 'monastery' || tags.historic === 'fort' ? 94 : 84,
      name: tags.name,
    };
  }
  if (tags.amenity === 'townhall') {
    return { category: 'public-building', featureType: 'Town hall', score: 80 };
  }
  if (tags.landuse === 'industrial' || ['industrial', 'warehouse'].includes(tags.building)) {
    return { category: 'industrial', featureType: 'Industrial building', score: 78 };
  }
  if (tags.waterway === 'canal') {
    return { category: 'water', featureType: 'Canal', score: 74 };
  }
  if (['village', 'hamlet', 'town', 'city'].includes(tags.place)) {
    return {
      category: 'settlement',
      featureType:
        tags.place === 'village'
          ? 'Village centre'
          : tags.place === 'hamlet'
            ? 'Hamlet centre'
            : `${tags.place[0].toUpperCase()}${tags.place.slice(1)} centre`,
      score: tags.place === 'village' ? 72 : tags.place === 'hamlet' ? 68 : 70,
    };
  }
  if (tags.place === 'isolated_dwelling' || tags.place === 'neighbourhood') {
    return {
      category: 'settlement',
      featureType: tags.place === 'isolated_dwelling' ? 'Isolated dwelling' : 'Neighbourhood centre',
      score: tags.place === 'isolated_dwelling' ? 66 : 62,
    };
  }
  if (tags.natural === 'forest') {
    return { category: 'terrain', featureType: 'Forest', score: 70, name: tags.name };
  }
  if (['public', 'retail', 'commercial'].includes(tags.building)) {
    return { category: 'building', featureType: 'Distinctive building', score: tags.name ? 72 : 66 };
  }
  if (['house', 'apartments', 'terrace', 'farm'].includes(tags.building)) {
    return {
      category: 'building',
      featureType: tags.building === 'farm' ? 'Farmhouse' : 'Building',
      score: 62,
    };
  }
  if (tags.waterway === 'stream' && tags.name) {
    return { category: 'water', featureType: 'Named stream', score: 58 };
  }
  return null;
}

function confidence(score: number): OsmPhotoCandidate['confidence'] {
  if (score >= 92) return 'excellent';
  if (score >= 82) return 'strong';
  if (score >= 70) return 'good';
  return 'fallback';
}

function normalizedOsmText(value: string | undefined): string | undefined {
  if (!value) return undefined;
  const withoutControls = Array.from(value.normalize('NFC'), (character) => {
    const code = character.charCodeAt(0);
    return code < 32 || code === 127 ? ' ' : character;
  }).join('');
  const normalized = withoutControls.replace(/\s+/g, ' ').trim().slice(0, MAXIMUM_OSM_TEXT_LENGTH);
  return normalized || undefined;
}

function toCandidate(
  id: string,
  coordinate: OsmCoordinate,
  definition: CandidateDefinition,
  tags: Record<string, string>,
  route: Route,
  points: Waypoint[]
): OsmPhotoCandidate | null {
  const analysis = analyzePhotoPosition(coordinate.lat, coordinate.lon, null, route, points);
  const leg = route.legs[analysis.legIndex];
  if (
    analysis.routePosition !== 'on-route' ||
    analysis.ambiguousLegIndices.length > 1 ||
    analysis.lateralDistanceM > ROUTE_CORRIDOR_M ||
    analysis.distanceAfterPreviousControlPointM < CONTROL_CLEARANCE_M ||
    leg.length - analysis.distanceOnLegM < LEG_END_MARGIN_M
  ) {
    return null;
  }
  const source: OrthophotoTargetSource = {
    provider: 'OpenStreetMap',
    elementId: id,
    category: definition.category,
    featureType: definition.featureType,
    name: normalizedOsmText(definition.name ?? tags.name) ?? null,
    score: definition.score,
    attribution: OSM_ATTRIBUTION,
  };
  return {
    id,
    label: '',
    latitude: Number(coordinate.lat.toFixed(7)),
    longitude: Number(coordinate.lon.toFixed(7)),
    alongRouteM: analysis.alongRouteM,
    name: source.name ?? definition.featureType,
    category: definition.category,
    featureType: definition.featureType,
    score: definition.score,
    confidence: confidence(definition.score),
    lateralDistanceM: analysis.lateralDistanceM,
    previousControlPoint: analysis.previousControlPoint,
    distanceAfterControlM: analysis.distanceAfterPreviousControlPointM,
    routeLegIndex: analysis.legIndex,
    routeLegName: `${points[analysis.legIndex][0]}–${points[analysis.legIndex + 1][0]}`,
    routeLegCount: route.legs.length,
    routeLegBoundariesM: [...route.legs.map((routeLeg) => routeLeg.cumulativeStart), route.totalDistance],
    source,
  };
}

function derivedRoadJunctions(elements: OverpassElement[], route: Route, points: Waypoint[]) {
  const nodes = new Map<
    number,
    { coordinate: OsmCoordinate; neighbours: Set<number>; roadClasses: Set<string>; names: Set<string> }
  >();
  for (const way of elements.filter((element) => element.type === 'way' && element.tags?.highway)) {
    if (!way.nodes || !way.geometry || way.nodes.length !== way.geometry.length) continue;
    for (let index = 0; index < way.nodes.length; index += 1) {
      const nodeId = way.nodes[index];
      const entry = nodes.get(nodeId) ?? {
        coordinate: way.geometry[index],
        neighbours: new Set<number>(),
        roadClasses: new Set<string>(),
        names: new Set<string>(),
      };
      if (index > 0) entry.neighbours.add(way.nodes[index - 1]);
      if (index < way.nodes.length - 1) entry.neighbours.add(way.nodes[index + 1]);
      entry.roadClasses.add(way.tags?.highway ?? 'road');
      if (way.tags?.name) entry.names.add(way.tags.name);
      nodes.set(nodeId, entry);
    }
  }
  const classScore: Record<string, number> = {
    secondary: 90,
    tertiary: 86,
    unclassified: 81,
    residential: 75,
  };
  return Array.from(nodes.entries()).flatMap(([nodeId, entry]) => {
    if (entry.neighbours.size < 3) return [];
    const roadClass = [...entry.roadClasses].sort(
      (left, right) => (classScore[right] ?? 0) - (classScore[left] ?? 0)
    )[0];
    const definition: CandidateDefinition = {
      category: 'road-junction',
      featureType: `${entry.neighbours.size}-arm road junction`,
      score: classScore[roadClass] ?? 74,
      name: [...entry.names].join(' / ') || undefined,
    };
    const candidate = toCandidate(
      `node/${nodeId}`,
      entry.coordinate,
      definition,
      { highway: roadClass },
      route,
      points
    );
    return candidate ? [candidate] : [];
  });
}

function deduplicateCandidates(candidates: OsmPhotoCandidate[]): OsmPhotoCandidate[] {
  const result: OsmPhotoCandidate[] = [];
  for (const candidate of [...candidates].sort((left, right) => right.score - left.score)) {
    const duplicate = result.some(
      (existing) =>
        haversine(existing.latitude, existing.longitude, candidate.latitude, candidate.longitude) <
        SPATIAL_DEDUPLICATION_M
    );
    if (!duplicate) result.push(candidate);
  }
  return result;
}

export function discoverOsmPhotoCandidates(
  data: OverpassResponse,
  route: Route,
  points: Waypoint[]
): OsmPhotoCandidate[] {
  const direct = data.elements.flatMap((element) => {
    if (element.tags?.highway && !element.tags.bridge) return [];
    const coordinate = elementCoordinate(element);
    const definition = candidateDefinition(element.tags ?? {});
    if (!coordinate || !definition) return [];
    const candidate = toCandidate(
      `${element.type}/${element.id}`,
      coordinate,
      definition,
      element.tags ?? {},
      route,
      points
    );
    return candidate ? [candidate] : [];
  });
  return deduplicateCandidates([...direct, ...derivedRoadJunctions(data.elements, route, points)]).sort(
    (left, right) => right.score - left.score || (left.alongRouteM ?? 0) - (right.alongRouteM ?? 0)
  );
}

interface RankedControlCandidate extends ControlPhotoCandidate {
  tags: Record<string, string>;
}

function controlFeatureFilters(): string[] {
  return [
    ...coreFeatureFilters(),
    `nwr["building"~"^(house|apartments|terrace|farm|church|chapel|school|industrial|warehouse|public|retail|commercial)$"]`,
  ];
}

export function buildOverpassControlTrueQuery([, latitude, longitude]: Waypoint): string {
  const around = `(around:${CONTROL_TRUE_SEARCH_RADIUS_M},${latitude.toFixed(7)},${longitude.toFixed(7)})`;
  return `[out:json][timeout:30];(${controlFeatureFilters()
    .map((filter) => `${filter}${around};`)
    .join('')});out body center;`;
}

function buildOverpassControlTrueBatchQuery(points: Waypoint[], indices: number[]): string {
  const statements = indices.flatMap((index) => {
    const [, latitude, longitude] = points[index];
    const around = `(around:${CONTROL_TRUE_SEARCH_RADIUS_M},${latitude.toFixed(7)},${longitude.toFixed(7)})`;
    return controlFeatureFilters().map((filter) => `${filter}${around};`);
  });
  return `[out:json][timeout:30];(${statements.join('')});out body center;`;
}

function specificControlFilter(tags: Record<string, string>): string | null {
  const choices: Array<[string, string | undefined]> = [
    ['railway', tags.railway === 'level_crossing' ? tags.railway : undefined],
    ['bridge', tags.bridge && tags.bridge !== 'no' ? tags.bridge : undefined],
    ['historic', tags.historic],
    ['amenity', tags.amenity],
    ['leisure', tags.leisure],
    ['landuse', tags.landuse],
    ['man_made', tags.man_made],
    ['power', tags.power],
    ['natural', tags.natural],
    ['waterway', tags.waterway],
    ['place', tags.place],
    ['building', tags.building],
  ];
  const selected = choices.find(([, value]) => value);
  if (!selected) return null;
  const [key, value] = selected;
  if (!/^[a-z_]+$/.test(key) || !/^[a-z0-9_:.-]+$/i.test(value ?? '')) return null;
  return `nwr["${key}"="${value}"]`;
}

export function buildOverpassControlFalseQuery(
  correct: ControlPhotoCandidate,
  tags: Record<string, string>
): string {
  const filter = specificControlFilter(tags);
  if (!filter) throw new Error(`No bounded similarity query is available for ${correct.featureType}.`);
  // Fetch the complete 10 NM candidate area once. The 1 NM minimum and exact
  // 10 NM maximum remain enforced by rankedControlCandidates below. Repeated
  // 8-direction probes at five radii made a single false lookup issue up to
  // 40 expensive Overpass subqueries.
  return `[out:json][timeout:30];${filter}(around:${CONTROL_FALSE_MAXIMUM_M},${correct.latitude.toFixed(7)},${correct.longitude.toFixed(7)});out body center;`;
}

function buildOverpassControlFalseBatchQuery(
  correctTargets: Array<[number, RankedControlCandidate]>
): string {
  const statements = correctTargets.flatMap(([, correct]) => {
    const filter = specificControlFilter(correct.tags);
    if (!filter) return [];
    return `${filter}(around:${CONTROL_FALSE_MAXIMUM_M},${correct.latitude.toFixed(7)},${correct.longitude.toFixed(7)});`;
  });
  return `[out:json][timeout:30];(${statements.join('')});out body center;`;
}

function rankedControlCandidates(
  data: OverpassResponse,
  waypoint: Waypoint,
  correct?: ControlPhotoCandidate
): RankedControlCandidate[] {
  const [, waypointLatitude, waypointLongitude] = waypoint;
  return data.elements
    .flatMap((element) => {
      const coordinate = elementCoordinate(element);
      const tags = element.tags ?? {};
      const definition = candidateDefinition(tags);
      if (!coordinate || !definition) return [];
      const distanceFromWaypointM = haversine(
        waypointLatitude,
        waypointLongitude,
        coordinate.lat,
        coordinate.lon
      );
      const distanceFromCorrectM = correct
        ? haversine(correct.latitude, correct.longitude, coordinate.lat, coordinate.lon)
        : 0;
      if (
        correct &&
        (distanceFromCorrectM < CONTROL_FALSE_MINIMUM_M ||
          distanceFromCorrectM > CONTROL_FALSE_MAXIMUM_M ||
          definition.featureType !== correct.featureType)
      ) {
        return [];
      }
      const source: NonNullable<OrthophotoTarget['source']> = {
        provider: 'OpenStreetMap',
        elementId: `${element.type}/${element.id}`,
        category: definition.category,
        featureType: definition.featureType,
        name: normalizedOsmText(definition.name ?? tags.name) ?? null,
        score: definition.score,
        attribution: OSM_ATTRIBUTION,
        controlRole: correct ? 'false' : 'true',
        controlWaypoint: waypoint[0],
        ...(correct
          ? {
              correctObjectLatitude: correct.latitude,
              correctObjectLongitude: correct.longitude,
            }
          : {}),
      };
      return [
        {
          id: source.elementId,
          label: '',
          latitude: Number(coordinate.lat.toFixed(7)),
          longitude: Number(coordinate.lon.toFixed(7)),
          name: source.name ?? definition.featureType,
          category: definition.category,
          featureType: definition.featureType,
          score: definition.score,
          distanceFromWaypointM,
          distanceFromCorrectM,
          source,
          tags,
        },
      ];
    })
    .sort((left, right) => {
      const leftDistance = correct ? left.distanceFromCorrectM : left.distanceFromWaypointM;
      const rightDistance = correct ? right.distanceFromCorrectM : right.distanceFromWaypointM;
      return leftDistance - rightDistance || right.score - left.score;
    });
}

function centerTrueControlPhotoOnWaypoint(
  candidate: RankedControlCandidate,
  waypoint: Waypoint
): RankedControlCandidate {
  return {
    ...candidate,
    latitude: waypoint[1],
    longitude: waypoint[2],
    distanceFromCorrectM: 0,
    source: {
      ...candidate.source,
      correctObjectLatitude: waypoint[1],
      correctObjectLongitude: waypoint[2],
    },
  };
}

export async function fetchOsmControlPhotoProposals(
  points: Waypoint[],
  options: FetchControlPhotoOptions = {}
): Promise<{ proposals: ControlPhotoProposal[]; warnings: string[] }> {
  const fetcher = options.fetcher ?? fetch;
  const waypointIndices = options.waypointIndices ?? points.map((_, index) => index);
  if (
    waypointIndices.length === 0 ||
    waypointIndices.some(
      (index, position) =>
        !Number.isInteger(index) ||
        index < 0 ||
        index >= points.length ||
        waypointIndices.indexOf(index) !== position
    )
  ) {
    throw new Error('Control-photo discovery requires unique, valid waypoint indices.');
  }
  const warnings: string[] = [];
  const discoveryController = new AbortController();
  let deadlineReached = false;
  const abortFromCaller = () => discoveryController.abort(options.signal?.reason);
  options.signal?.addEventListener('abort', abortFromCaller, { once: true });
  const deadline = globalThis.setTimeout(() => {
    deadlineReached = true;
    discoveryController.abort(new Error('OpenStreetMap control-photo discovery reached its time limit.'));
  }, options.deadlineMs ?? CONTROL_DISCOVERY_DEADLINE_MS);
  const discoverySignal = discoveryController.signal;
  const trueTargets = new Map<number, RankedControlCandidate>();
  const emit = (
    waypointIndex: number,
    stage: 'true' | 'false',
    state: OsmDiscoveryStageState,
    detail?: string
  ) =>
    options.onProgress?.({
      waypointIndex,
      waypointCount: points.length,
      waypointName: points[waypointIndex][0],
      stage,
      state,
      ...(detail ? { detail } : {}),
    });
  const requestBatch = async (
    indices: number[],
    stage: 'true' | 'false',
    query: string,
    attempts: ReadonlyArray<readonly [string, number]>,
    retry = false
  ): Promise<OverpassResponse | string> => {
    for (const index of indices) emit(index, stage, retry ? 'retrying' : 'started');
    try {
      const result = await requestOverpass(query, discoverySignal, fetcher, attempts);
      for (const index of indices) emit(index, stage, retry ? 'recovered' : 'completed');
      return result;
    } catch (error) {
      if (options.signal?.aborted) throw error;
      const detail = error instanceof Error ? error.message : String(error);
      for (const index of indices) emit(index, stage, 'warning', detail);
      return detail;
    }
  };
  try {
    const trueQuery = buildOverpassControlTrueBatchQuery(points, waypointIndices);
    let trueResult = await requestBatch(waypointIndices, 'true', trueQuery, PRIMARY_OVERPASS_ATTEMPTS);
    if (typeof trueResult === 'string' && !discoverySignal.aborted) {
      const delay = options.retryDelayMs ?? 900;
      if (delay > 0) await new Promise<void>((resolve) => globalThis.setTimeout(resolve, delay));
      trueResult = await requestBatch(waypointIndices, 'true', trueQuery, RETRY_OVERPASS_ATTEMPTS, true);
    }
    if (typeof trueResult === 'string') {
      for (const index of waypointIndices) warnings.push(`${points[index][0]} true target: ${trueResult}`);
    } else {
      for (const index of waypointIndices) {
        const candidate = rankedControlCandidates(trueResult, points[index])[0];
        if (candidate) trueTargets.set(index, centerTrueControlPhotoOnWaypoint(candidate, points[index]));
        else warnings.push(`${points[index][0]}: no identifiable OSM object was found within 3 km.`);
      }
    }

    const falseTargets = new Map<number, RankedControlCandidate>();
    const falseTargetsToFind = [...trueTargets.entries()].filter(
      ([index]) => index !== 0 && index !== points.length - 1
    );
    if (falseTargetsToFind.length > 0 && !discoverySignal.aborted) {
      const falseQuery = buildOverpassControlFalseBatchQuery(falseTargetsToFind);
      let falseResult = await requestBatch(
        falseTargetsToFind.map(([index]) => index),
        'false',
        falseQuery,
        PRIMARY_OVERPASS_ATTEMPTS
      );
      if (typeof falseResult === 'string' && !discoverySignal.aborted) {
        const delay = options.retryDelayMs ?? 900;
        if (delay > 0) await new Promise<void>((resolve) => globalThis.setTimeout(resolve, delay));
        falseResult = await requestBatch(
          falseTargetsToFind.map(([index]) => index),
          'false',
          falseQuery,
          RETRY_OVERPASS_ATTEMPTS,
          true
        );
      }
      for (const [index, correct] of falseTargetsToFind) {
        if (typeof falseResult === 'string') {
          warnings.push(`${points[index][0]} false target: ${falseResult}`);
          continue;
        }
        const candidate = rankedControlCandidates(falseResult, points[index], correct)[0];
        if (candidate) falseTargets.set(index, candidate);
        else warnings.push(`${points[index][0]}: no similar false object was found from 1 to 10 NM away.`);
      }
    }

    if (deadlineReached) {
      warnings.push('The control-photo discovery time limit was reached; completed results were retained.');
    }
    const proposals = [...trueTargets.entries()]
      .sort(([leftIndex], [rightIndex]) => leftIndex - rightIndex)
      .map(([index, trueTarget]) => ({
        waypoint: points[index],
        trueTarget,
        falseTarget: falseTargets.get(index) ?? null,
      }));
    return { proposals, warnings };
  } finally {
    globalThis.clearTimeout(deadline);
    options.signal?.removeEventListener('abort', abortFromCaller);
  }
}

const CATEGORY_LIMIT = 2;

function selectGroup(
  pool: OsmPhotoCandidate[],
  count: number,
  alreadySelected: OsmPhotoCandidate[],
  random: () => number
): OsmPhotoCandidate[] {
  const ranked = pool
    .map((candidate) => ({ candidate, rank: candidate.score + random() * 14 }))
    .sort((left, right) => right.rank - left.rank)
    .map(({ candidate }) => candidate);
  const selected: OsmPhotoCandidate[] = [];
  const canSelect = (candidate: OsmPhotoCandidate, enforceCategoryLimit: boolean): boolean => {
    const combined = [...alreadySelected, ...selected];
    if (
      combined.some(
        (existing) =>
          Math.abs((existing.alongRouteM ?? 0) - (candidate.alongRouteM ?? 0)) < MINIMUM_TARGET_SPACING_M
      )
    ) {
      return false;
    }
    return (
      !enforceCategoryLimit ||
      combined.filter((existing) => existing.category === candidate.category).length < CATEGORY_LIMIT
    );
  };

  const legIndices = [...new Set(ranked.map((candidate) => candidate.routeLegIndex))].sort(
    (left, right) => left - right
  );
  const fillEvenly = (enforceCategoryLimit: boolean) => {
    let madeProgress = true;
    while (selected.length < count && madeProgress) {
      madeProgress = false;
      for (const routeLegIndex of legIndices) {
        if (selected.length >= count) break;
        const candidate = ranked.find(
          (item) =>
            item.routeLegIndex === routeLegIndex &&
            !selected.includes(item) &&
            !alreadySelected.includes(item) &&
            canSelect(item, enforceCategoryLimit)
        );
        if (!candidate) continue;
        selected.push(candidate);
        madeProgress = true;
      }
    }
  };
  fillEvenly(true);
  if (selected.length < count) {
    fillEvenly(false);
  }
  return selected;
}

export function selectOsmPhotoCandidates(
  candidates: OsmPhotoCandidate[],
  count: number,
  splitAfterM: number | null,
  random: () => number = Math.random
): OsmPhotoCandidate[] {
  if (!Number.isInteger(count) || count < 1 || count > 12) {
    throw new Error('OSM target count must be an integer from 1 to 12.');
  }
  let selected: OsmPhotoCandidate[] = [];
  if (splitAfterM !== null && count >= 2) {
    const firstCount = Math.ceil(count / 2);
    selected = selectGroup(
      candidates.filter((candidate) => (candidate.alongRouteM ?? 0) <= splitAfterM),
      firstCount,
      selected,
      random
    );
    selected.push(
      ...selectGroup(
        candidates.filter((candidate) => (candidate.alongRouteM ?? 0) > splitAfterM),
        count - firstCount,
        selected,
        random
      )
    );
  } else {
    selected = selectGroup(candidates, count, [], random);
  }
  return selected.sort((left, right) => (left.alongRouteM ?? 0) - (right.alongRouteM ?? 0));
}

export function targetSelectionIssue(
  selected: OsmPhotoCandidate[],
  _requestedCount: number,
  splitAfterM: number | null
): string | null {
  if (selected.length === 0) return 'Select at least one proposed target.';
  const boundaries = selected[0]?.routeLegBoundariesM;
  const routeLegCount = Math.max(1, ...selected.map((candidate) => candidate.routeLegCount));
  const sectionCoverage = (beforeSplit: boolean) => {
    const sectionCount =
      splitAfterM === null
        ? selected.length
        : beforeSplit
          ? selected.filter((candidate) => (candidate.alongRouteM ?? 0) <= splitAfterM).length
          : selected.filter((candidate) => (candidate.alongRouteM ?? 0) > splitAfterM).length;
    const expectedLegIndices = boundaries
      ? boundaries.slice(0, -1).flatMap((start, routeLegIndex) => {
          const end = boundaries[routeLegIndex + 1];
          const belongs = splitAfterM === null || (beforeSplit ? start < splitAfterM : end > splitAfterM);
          return belongs ? [routeLegIndex] : [];
        })
      : Array.from({ length: routeLegCount }, (_, routeLegIndex) => routeLegIndex);
    const selectedLegIndices = new Set(
      selected
        .filter((candidate) =>
          splitAfterM === null
            ? true
            : beforeSplit
              ? (candidate.alongRouteM ?? 0) <= splitAfterM
              : (candidate.alongRouteM ?? 0) > splitAfterM
        )
        .map((candidate) => candidate.routeLegIndex)
    );
    return {
      sectionCount,
      expectedLegIndices,
      missingLegIndices: expectedLegIndices.filter((routeLegIndex) => !selectedLegIndices.has(routeLegIndex)),
    };
  };
  const sections =
    splitAfterM === null ? [sectionCoverage(true)] : [sectionCoverage(true), sectionCoverage(false)];
  for (const section of sections) {
    if (section.sectionCount >= section.expectedLegIndices.length && section.missingLegIndices.length > 0) {
      return `Select at least one target on every route leg; missing route leg${section.missingLegIndices.length === 1 ? '' : 's'} ${section.missingLegIndices.map((index) => index + 1).join(', ')}.`;
    }
  }
  for (let first = 0; first < selected.length; first += 1) {
    for (let second = first + 1; second < selected.length; second += 1) {
      if (
        Math.abs((selected[first].alongRouteM ?? 0) - (selected[second].alongRouteM ?? 0)) <
        MINIMUM_TARGET_SPACING_M
      ) {
        return `${selected[first].name} and ${selected[second].name} are less than 500 m apart along the route.`;
      }
    }
  }
  if (splitAfterM !== null && selected.length >= 2) {
    const firstHalfCount = selected.filter((candidate) => (candidate.alongRouteM ?? 0) <= splitAfterM).length;
    const requiredFirstHalf = Math.ceil(selected.length / 2);
    if (firstHalfCount !== requiredFirstHalf) {
      return `Selection balance warning: choose ${requiredFirstHalf} targets before the handout split and ${Math.floor(selected.length / 2)} after it for an even split.`;
    }
  }
  return null;
}
