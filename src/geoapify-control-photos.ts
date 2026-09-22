import {
  catalogEntryForGeoapifyCategories,
  GEOAPIFY_CANDIDATE_CATALOG,
  GEOAPIFY_SEARCH_CATEGORIES,
  type GeoapifyCandidateCatalogEntry,
} from './candidate-catalog';
import { haversine, type Route, type Waypoint } from './domain';
import type {
  ControlDiscoveryProgress,
  ControlPhotoCandidate,
  ControlPhotoProposal,
  OsmDiscoveryProgress,
  OverpassResponse,
} from './osm-photo-candidates';

export const GEOAPIFY_PLACES_URL = 'https://api.geoapify.com/v2/places';
const TRUE_RADIUS_M = 3000;
const FALSE_MINIMUM_M = 1852;
const FALSE_MAXIMUM_M = 10 * 1852;
const DEFAULT_DEADLINE_MS = 45_000;
const ROUTE_CORRIDOR_MARGIN_M = 300;

interface GeoapifyProperties {
  place_id?: string;
  name?: string;
  categories?: string[];
  lat?: number;
  lon?: number;
  datasource?: { raw?: Record<string, unknown> };
}

interface GeoapifyFeature {
  type?: string;
  geometry?: { type?: string; coordinates?: unknown };
  properties?: GeoapifyProperties;
}

interface GeoapifyResponse {
  features?: GeoapifyFeature[];
}

interface ProviderCandidate {
  candidate: ControlPhotoCandidate;
  featureType: string;
}

export interface FetchGeoapifyControlPhotoOptions {
  apiKey: string;
  signal?: AbortSignal;
  fetcher?: typeof fetch;
  waypointIndices?: number[];
  deadlineMs?: number;
  onProgress?: (progress: ControlDiscoveryProgress) => void;
}

export interface FetchGeoapifyPhotoOptions {
  apiKey: string;
  route?: Route;
  signal?: AbortSignal;
  fetcher?: typeof fetch;
  deadlineMs?: number;
  onProgress?: (progress: OsmDiscoveryProgress) => void;
}

function catalogEntryForFeature(feature: GeoapifyFeature): GeoapifyCandidateCatalogEntry | null {
  return catalogEntryForGeoapifyCategories(feature.properties?.categories ?? []);
}

const CATEGORY_DEFINITIONS = [
  ['man_made.bridge', 'Bridge', 'bridge', 95],
  ['religion.place_of_worship', 'Church or place of worship', 'landmark', 96],
  ['building.place_of_worship', 'Church or place of worship', 'landmark', 96],
  ['sport.pitch', 'Sports pitch', 'sports', 94],
  ['sport.sports_centre', 'Sports centre', 'sports', 85],
  ['sport.stadium', 'Stadium', 'sports', 86],
  ['sport.golf_course', 'Golf course', 'sports', 80],
  ['sport.track', 'Sports track', 'sports', 82],
  ['education.school', 'School', 'public-building', 88],
  ['building.school', 'School', 'public-building', 88],
  ['building.public_and_civil', 'Public building', 'public-building', 78],
  ['building.industrial', 'Industrial building', 'industrial', 78],
  ['building.transportation', 'Transport building', 'building', 76],
  ['entertainment.museum', 'Museum', 'landmark', 84],
  ['healthcare.hospital', 'Hospital', 'public-building', 84],
  ['man_made.tower', 'Tower', 'landmark', 90],
  ['man_made.water_tower', 'Water tower', 'landmark', 89],
  ['man_made.watermill', 'Watermill', 'landmark', 88],
  ['man_made.windmill', 'Windmill', 'landmark', 88],
  ['man_made.lighthouse', 'Lighthouse', 'landmark', 90],
  ['man_made.breakwater', 'Breakwater', 'infrastructure', 82],
  ['man_made.pier', 'Pier', 'infrastructure', 80],
  ['power.substation', 'Power substation', 'infrastructure', 90],
  ['power.plant', 'Power plant', 'infrastructure', 88],
  ['power.generator', 'Power generator', 'infrastructure', 86],
  ['production.factory', 'Factory', 'industrial', 78],
  ['production.brewery', 'Brewery', 'industrial', 76],
  ['production.pottery', 'Pottery works', 'industrial', 76],
  ['production.winery', 'Winery', 'industrial', 76],
  ['activity.community_center', 'Community centre', 'public-building', 78],
  ['airport.airfield', 'Airfield', 'aviation', 94],
  ['airport.gliding', 'Gliding airfield', 'aviation', 94],
  ['memorial.cemetery', 'Cemetery', 'cemetery', 93],
  ['memorial.graveyard', 'Graveyard', 'cemetery', 93],
  ['memorial', 'Memorial', 'landmark', 80],
  ['populated_place.village', 'Village', 'settlement', 72],
  ['populated_place.hamlet', 'Hamlet', 'settlement', 68],
  ['populated_place.town', 'Town', 'settlement', 70],
  ['populated_place.city', 'City', 'settlement', 70],
  ['tourism.sights.castle', 'Castle', 'landmark', 98],
  ['tourism.sights.tower', 'Tower', 'landmark', 90],
  ['tourism.sights.windmill', 'Windmill', 'landmark', 88],
  ['tourism.sights.monastery', 'Monastery', 'landmark', 94],
  ['tourism.sights.fort', 'Fort', 'landmark', 94],
  ['tourism.sights.manor', 'Manor', 'landmark', 90],
  ['tourism.sights.mine', 'Mine', 'landmark', 84],
  ['tourism.sights.ruines', 'Ruins', 'landmark', 84],
  ['tourism.sights', 'Landmark', 'landmark', 80],
  ['natural.forest', 'Forest', 'terrain', 70],
  ['natural.water', 'Lake or pond', 'water', 86],
  ['waterway', 'Waterway', 'water', 82],
] as const;

const SEARCH_CATEGORIES = GEOAPIFY_SEARCH_CATEGORIES;

function categoryTags(feature: GeoapifyFeature): Record<string, string> | null {
  const catalogEntry = catalogEntryForFeature(feature);
  if (catalogEntry) return catalogEntry.osmTags;
  const categories = feature.properties?.categories ?? [];
  if (
    categories.some((category) => category === 'man_made.bridge' || category.startsWith('man_made.bridge.'))
  ) {
    return { bridge: 'yes' };
  }
  if (categories.some((category) => category.startsWith('religion.place_of_worship'))) {
    return { amenity: 'place_of_worship' };
  }
  if (categories.some((category) => category.startsWith('building.place_of_worship'))) {
    return { building: 'church' };
  }
  if (categories.some((category) => category.startsWith('sport.pitch'))) return { leisure: 'pitch' };
  if (categories.some((category) => category.startsWith('sport.sports_centre'))) {
    return { leisure: 'sports_centre' };
  }
  if (categories.some((category) => category.startsWith('sport.stadium')))
    return { leisure: 'sports_centre' };
  if (categories.some((category) => category.startsWith('sport.golf_course'))) return { leisure: 'pitch' };
  if (categories.some((category) => category.startsWith('sport.track'))) return { leisure: 'pitch' };
  if (categories.some((category) => category.startsWith('education.school'))) return { amenity: 'school' };
  if (categories.some((category) => category.startsWith('building.school'))) return { building: 'school' };
  if (categories.some((category) => category.startsWith('building.public_and_civil')))
    return { building: 'public' };
  if (categories.some((category) => category.startsWith('building.industrial')))
    return { building: 'industrial' };
  if (categories.some((category) => category.startsWith('building.transportation'))) {
    return { building: 'public' };
  }
  if (categories.some((category) => category.startsWith('entertainment.museum')))
    return { tourism: 'museum' };
  if (categories.some((category) => category.startsWith('healthcare.hospital')))
    return { amenity: 'hospital' };
  if (categories.some((category) => category.startsWith('man_made.tower'))) return { man_made: 'tower' };
  if (categories.some((category) => category.startsWith('man_made.water_tower'))) {
    return { man_made: 'storage_tank' };
  }
  if (categories.some((category) => category.startsWith('man_made.watermill')))
    return { man_made: 'watermill' };
  if (categories.some((category) => category.startsWith('man_made.windmill')))
    return { man_made: 'windmill' };
  if (categories.some((category) => category.startsWith('man_made.lighthouse')))
    return { man_made: 'lighthouse' };
  if (categories.some((category) => category.startsWith('man_made.breakwater')))
    return { man_made: 'breakwater' };
  if (categories.some((category) => category.startsWith('man_made.pier'))) return { man_made: 'pier' };
  if (categories.some((category) => category.startsWith('power.substation'))) return { power: 'substation' };
  if (categories.some((category) => category.startsWith('power.plant'))) return { landuse: 'industrial' };
  if (categories.some((category) => category.startsWith('power.generator'))) return { power: 'generator' };
  if (categories.some((category) => category.startsWith('production.factory')))
    return { landuse: 'industrial' };
  if (categories.some((category) => category.startsWith('production.'))) return { landuse: 'industrial' };
  if (categories.some((category) => category.startsWith('activity.community_center'))) {
    return { amenity: 'community_centre' };
  }
  if (categories.some((category) => category.startsWith('airport.'))) return { aeroway: 'aerodrome' };
  if (categories.some((category) => category.startsWith('memorial.cemetery'))) return { landuse: 'cemetery' };
  if (categories.some((category) => category.startsWith('memorial.graveyard')))
    return { landuse: 'cemetery' };
  if (categories.some((category) => category.startsWith('populated_place.village')))
    return { place: 'village' };
  if (categories.some((category) => category.startsWith('populated_place.hamlet')))
    return { place: 'hamlet' };
  if (categories.some((category) => category.startsWith('populated_place.town'))) return { place: 'town' };
  if (categories.some((category) => category.startsWith('populated_place.city'))) return { place: 'city' };
  if (categories.some((category) => category === 'memorial' || category.startsWith('memorial.'))) {
    return { historic: 'monument' };
  }
  if (categories.some((category) => category.startsWith('tourism.sights.castle')))
    return { historic: 'castle' };
  if (categories.some((category) => category.startsWith('tourism.sights.tower')))
    return { man_made: 'tower' };
  if (categories.some((category) => category.startsWith('tourism.sights.windmill')))
    return { man_made: 'windmill' };
  if (categories.some((category) => category.startsWith('tourism.sights.monastery'))) {
    return { historic: 'monastery' };
  }
  if (categories.some((category) => category.startsWith('tourism.sights.fort'))) return { historic: 'fort' };
  if (categories.some((category) => category.startsWith('tourism.sights.manor')))
    return { historic: 'manor' };
  if (categories.some((category) => category.startsWith('tourism.sights.mine'))) return { man_made: 'mine' };
  if (categories.some((category) => category.startsWith('tourism.sights.ruines'))) {
    return { historic: 'ruins' };
  }
  if (categories.some((category) => category.startsWith('tourism.sights'))) return { historic: 'monument' };
  if (categories.some((category) => category.startsWith('natural.forest'))) return { natural: 'forest' };
  if (categories.some((category) => category.startsWith('natural.water'))) return { natural: 'water' };
  if (categories.some((category) => category.startsWith('waterway'))) return { waterway: 'river' };
  return null;
}

function coordinateFromFeature(feature: GeoapifyFeature): { latitude: number; longitude: number } | null {
  const properties = feature.properties;
  if (Number.isFinite(properties?.lat) && Number.isFinite(properties?.lon)) {
    const latitude = properties?.lat;
    const longitude = properties?.lon;
    if (latitude !== undefined && longitude !== undefined) return { latitude, longitude };
  }
  const coordinates = feature.geometry?.coordinates;
  if (
    Array.isArray(coordinates) &&
    coordinates.length >= 2 &&
    typeof coordinates[0] === 'number' &&
    typeof coordinates[1] === 'number'
  ) {
    return { latitude: coordinates[1], longitude: coordinates[0] };
  }
  return null;
}

function definitionForFeature(feature: GeoapifyFeature): GeoapifyCandidateCatalogEntry | null {
  const categories = feature.properties?.categories ?? [];
  const catalogEntry = catalogEntryForFeature(feature);
  if (catalogEntry) return catalogEntry;
  const legacyEntry = CATEGORY_DEFINITIONS.find(([category]) =>
    categories.some((candidate) => candidate === category || candidate.startsWith(`${category}.`))
  );
  return legacyEntry
    ? {
        geoapifyCategory: legacyEntry[0],
        featureType: legacyEntry[1],
        category: legacyEntry[2],
        diversityGroup: legacyEntry[2],
        score: legacyEntry[3],
        osmTags: {},
      }
    : null;
}

function sourceForFeature(
  feature: GeoapifyFeature,
  definition: GeoapifyCandidateCatalogEntry,
  waypoint: Waypoint,
  role: 'true' | 'false',
  correct?: { latitude: number; longitude: number }
): NonNullable<ControlPhotoCandidate['source']> {
  const properties = feature.properties ?? {};
  const raw = properties.datasource?.raw ?? {};
  const osmType = typeof raw.osm_type === 'string' ? raw.osm_type : 'feature';
  const osmId = raw.osm_id ?? properties.place_id ?? 'unknown';
  return {
    provider: 'OpenStreetMap',
    elementId: `geoapify:${osmType}/${osmId}`,
    category: definition.category,
    featureType: definition.featureType,
    name: properties.name ?? null,
    score: definition.score,
    attribution: '© OpenStreetMap contributors; accessed through Geoapify',
    providerCategory: definition.geoapifyCategory,
    diversityGroup: definition.diversityGroup,
    controlRole: role,
    controlWaypoint: waypoint[0],
    ...(correct
      ? { correctObjectLatitude: correct.latitude, correctObjectLongitude: correct.longitude }
      : {}),
  };
}

function toCandidate(
  feature: GeoapifyFeature,
  waypoint: Waypoint,
  role: 'true' | 'false',
  correct?: { latitude: number; longitude: number }
): ProviderCandidate | null {
  const coordinate = coordinateFromFeature(feature);
  const definition = definitionForFeature(feature);
  if (!coordinate || !definition) return null;
  const source = sourceForFeature(feature, definition, waypoint, role, correct);
  const distanceFromWaypointM = haversine(
    waypoint[1],
    waypoint[2],
    coordinate.latitude,
    coordinate.longitude
  );
  const distanceFromCorrectM = correct
    ? haversine(correct.latitude, correct.longitude, coordinate.latitude, coordinate.longitude)
    : 0;
  return {
    featureType: definition.featureType,
    candidate: {
      id: source.elementId,
      label: '',
      latitude: coordinate.latitude,
      longitude: coordinate.longitude,
      name: source.name ?? definition.featureType,
      category: definition.category,
      featureType: definition.featureType,
      score: definition.score,
      distanceFromWaypointM,
      distanceFromCorrectM,
      source,
    },
  };
}

async function fetchPlaces(
  latitude: number,
  longitude: number,
  radiusM: number,
  categories: string,
  options: FetchGeoapifyControlPhotoOptions,
  signal: AbortSignal
): Promise<GeoapifyFeature[]> {
  const url = new URL(GEOAPIFY_PLACES_URL);
  url.searchParams.set('categories', categories);
  url.searchParams.set('filter', `circle:${longitude},${latitude},${radiusM}`);
  url.searchParams.set('bias', `proximity:${longitude},${latitude}`);
  url.searchParams.set('limit', '100');
  url.searchParams.set('apiKey', options.apiKey.trim());
  const response = await (options.fetcher ?? fetch)(url, { signal });
  if (!response.ok) throw new Error(`Geoapify Places returned HTTP ${response.status}`);
  const data = (await response.json()) as GeoapifyResponse;
  return Array.isArray(data.features) ? data.features : [];
}

async function fetchPlacesInRect(
  route: Route,
  legIndex: number,
  options: FetchGeoapifyPhotoOptions,
  signal: AbortSignal
): Promise<GeoapifyFeature[]> {
  const leg = route.legs[legIndex];
  const latitudeMargin = ROUTE_CORRIDOR_MARGIN_M / 111_320;
  const averageLatitude = ((leg.fromLat + leg.toLat) / 2) * (Math.PI / 180);
  const longitudeMargin = ROUTE_CORRIDOR_MARGIN_M / Math.max(111_320 * Math.cos(averageLatitude), 22_264);
  const url = new URL(GEOAPIFY_PLACES_URL);
  url.searchParams.set('categories', SEARCH_CATEGORIES);
  url.searchParams.set(
    'filter',
    `rect:${Math.min(leg.fromLon, leg.toLon) - longitudeMargin},${Math.min(leg.fromLat, leg.toLat) - latitudeMargin},${Math.max(leg.fromLon, leg.toLon) + longitudeMargin},${Math.max(leg.fromLat, leg.toLat) + latitudeMargin}`
  );
  url.searchParams.set('bias', `proximity:${(leg.fromLon + leg.toLon) / 2},${(leg.fromLat + leg.toLat) / 2}`);
  url.searchParams.set('limit', '100');
  url.searchParams.set('apiKey', options.apiKey.trim());
  const response = await (options.fetcher ?? fetch)(url, { signal });
  if (!response.ok) throw new Error(`Geoapify Places returned HTTP ${response.status}`);
  const data = (await response.json()) as GeoapifyResponse;
  return Array.isArray(data.features) ? data.features : [];
}

function geoFeaturesToElements(features: GeoapifyFeature[], legIndex: number) {
  return features.flatMap((feature, featureIndex) => {
    const coordinate = coordinateFromFeature(feature);
    const definition = definitionForFeature(feature);
    const tags = categoryTags(feature);
    if (!coordinate || !definition || !tags) return [];
    const rawId = feature.properties?.place_id ?? `${legIndex}-${featureIndex}`;
    let id = 0;
    for (const character of rawId) id = (Math.imul(id, 31) + character.charCodeAt(0)) >>> 0;
    return [
      {
        type: 'node' as const,
        id: Math.max(1, id),
        lat: coordinate.latitude,
        lon: coordinate.longitude,
        tags: {
          ...tags,
          __candidate_category: definition.category,
          __candidate_feature_type: definition.featureType,
          __candidate_score: String(definition.score),
          __diversity_group: definition.diversityGroup,
          __provider_category: definition.geoapifyCategory,
          ...(feature.properties?.name ? { name: feature.properties.name } : {}),
        },
      },
    ];
  });
}

export async function fetchGeoapifyPhotoData(
  points: Waypoint[],
  options: FetchGeoapifyPhotoOptions
): Promise<OverpassResponse> {
  if (!options.apiKey.trim()) throw new Error('Geoapify API key is required.');
  const route = options.route;
  if (!route) throw new Error('Geoapify route-photo discovery requires a built route.');
  const controller = new AbortController();
  let deadlineReached = false;
  const abortFromCaller = () => controller.abort(options.signal?.reason);
  options.signal?.addEventListener('abort', abortFromCaller, { once: true });
  const deadline = globalThis.setTimeout(() => {
    deadlineReached = true;
    controller.abort(new Error('Geoapify competition-photo discovery reached its time limit.'));
  }, options.deadlineMs ?? 90_000);
  const elements = new Map<string, ReturnType<typeof geoFeaturesToElements>[number]>();
  const warnings: string[] = [];
  const emit = (
    legIndex: number,
    stage: OsmDiscoveryProgress['stage'],
    state: OsmDiscoveryProgress['state'],
    detail?: string
  ) =>
    options.onProgress?.({
      legIndex,
      legCount: route.legs.length,
      legName: `${points[legIndex][0]}–${points[legIndex + 1][0]}`,
      stage,
      state,
      completedSteps: 0,
      totalSteps: route.legs.length * 3,
      ...(detail ? { detail } : {}),
    });
  try {
    for (let legIndex = 0; legIndex < route.legs.length; legIndex += 1) {
      if (controller.signal.aborted) break;
      emit(legIndex, 'features', 'started');
      try {
        const features = await fetchPlacesInRect(route, legIndex, options, controller.signal);
        for (const element of geoFeaturesToElements(features, legIndex)) {
          elements.set(`${element.type}/${element.id}`, element);
        }
        emit(legIndex, 'features', 'completed');
        emit(legIndex, 'roads', 'skipped', 'Geoapify Places does not provide route-road geometry.');
        emit(
          legIndex,
          'buildings',
          'skipped',
          'Geoapify Places does not provide a separate building fallback.'
        );
      } catch (error) {
        if (options.signal?.aborted) throw error;
        const detail = error instanceof Error ? error.message : String(error);
        warnings.push(`${points[legIndex][0]}–${points[legIndex + 1][0]} features: ${detail}`);
        emit(legIndex, 'features', 'warning', detail);
        emit(legIndex, 'roads', 'skipped', 'Skipped after Geoapify feature lookup failed.');
        emit(legIndex, 'buildings', 'skipped', 'Skipped after Geoapify feature lookup failed.');
      }
    }
    if (deadlineReached)
      warnings.push(
        'The Geoapify competition-photo discovery time limit was reached; completed leg results were retained.'
      );
    return {
      elements: [...elements.values()],
      ...(warnings.length ? { warnings } : {}),
    };
  } finally {
    globalThis.clearTimeout(deadline);
    options.signal?.removeEventListener('abort', abortFromCaller);
  }
}

function nearestCandidate(
  features: GeoapifyFeature[],
  waypoint: Waypoint,
  role: 'true' | 'false',
  correct?: ControlPhotoCandidate
): ControlPhotoCandidate | null {
  return (
    features
      .map((feature) => toCandidate(feature, waypoint, role, correct))
      .filter((entry): entry is ProviderCandidate => entry !== null)
      .filter(({ candidate }) => {
        if (!correct) return true;
        return (
          candidate.featureType === correct.featureType &&
          candidate.distanceFromCorrectM >= FALSE_MINIMUM_M &&
          candidate.distanceFromCorrectM <= FALSE_MAXIMUM_M
        );
      })
      .sort((left, right) => {
        const leftDistance = correct
          ? left.candidate.distanceFromCorrectM
          : left.candidate.distanceFromWaypointM;
        const rightDistance = correct
          ? right.candidate.distanceFromCorrectM
          : right.candidate.distanceFromWaypointM;
        return leftDistance - rightDistance || right.candidate.score - left.candidate.score;
      })[0]?.candidate ?? null
  );
}

export async function fetchGeoapifyControlPhotoProposals(
  points: Waypoint[],
  options: FetchGeoapifyControlPhotoOptions
): Promise<{ proposals: ControlPhotoProposal[]; warnings: string[] }> {
  if (!options.apiKey.trim()) throw new Error('Geoapify API key is required.');
  const indices = options.waypointIndices ?? points.map((_, index) => index);
  const controller = new AbortController();
  let deadlineReached = false;
  const abortFromCaller = () => controller.abort(options.signal?.reason);
  options.signal?.addEventListener('abort', abortFromCaller, { once: true });
  const deadline = globalThis.setTimeout(() => {
    deadlineReached = true;
    controller.abort(new Error('Geoapify control-photo discovery reached its time limit.'));
  }, options.deadlineMs ?? DEFAULT_DEADLINE_MS);
  const warnings: string[] = [];
  const trueTargets = new Map<number, ControlPhotoCandidate>();
  const falseTargets = new Map<number, ControlPhotoCandidate>();
  const emit = (
    index: number,
    stage: 'true' | 'false',
    state: ControlDiscoveryProgress['state'],
    detail?: string
  ) =>
    options.onProgress?.({
      waypointIndex: index,
      waypointCount: points.length,
      waypointName: points[index][0],
      stage,
      state,
      ...(detail ? { detail } : {}),
    });
  try {
    for (const index of indices) {
      if (controller.signal.aborted) break;
      const point = points[index];
      emit(index, 'true', 'started');
      try {
        const features = await fetchPlaces(
          point[1],
          point[2],
          TRUE_RADIUS_M,
          SEARCH_CATEGORIES,
          options,
          controller.signal
        );
        const candidate = nearestCandidate(features, point, 'true');
        if (candidate) {
          trueTargets.set(index, {
            ...candidate,
            latitude: point[1],
            longitude: point[2],
            distanceFromCorrectM: 0,
            source: {
              ...candidate.source,
              correctObjectLatitude: point[1],
              correctObjectLongitude: point[2],
            },
          });
          emit(index, 'true', 'completed');
        } else {
          const message = `${point[0]}: Geoapify found no identifiable object within 3 km.`;
          warnings.push(message);
          emit(index, 'true', 'warning', message);
        }
      } catch (error) {
        if (options.signal?.aborted) throw error;
        const message = error instanceof Error ? error.message : String(error);
        warnings.push(`${point[0]} true target: ${message}`);
        emit(index, 'true', 'warning', message);
      }
    }
    for (const [index, correct] of trueTargets) {
      if (controller.signal.aborted || index === 0 || index === points.length - 1) continue;
      const point = points[index];
      emit(index, 'false', 'started');
      try {
        const category = GEOAPIFY_CANDIDATE_CATALOG.find(
          (entry) => entry.featureType === correct.featureType
        )?.geoapifyCategory;
        if (!category) throw new Error(`Geoapify has no category mapping for ${correct.featureType}.`);
        const features = await fetchPlaces(
          correct.latitude,
          correct.longitude,
          FALSE_MAXIMUM_M,
          category,
          options,
          controller.signal
        );
        const candidate = nearestCandidate(features, point, 'false', correct);
        if (candidate) {
          falseTargets.set(index, candidate);
          emit(index, 'false', 'completed');
        } else {
          const message = `${point[0]}: no similar false object was found from 1 to 10 NM away.`;
          warnings.push(message);
          emit(index, 'false', 'warning', message);
        }
      } catch (error) {
        if (options.signal?.aborted) throw error;
        const message = error instanceof Error ? error.message : String(error);
        warnings.push(`${point[0]} false target: ${message}`);
        emit(index, 'false', 'warning', message);
      }
    }
    if (deadlineReached)
      warnings.push(
        'The Geoapify control-photo discovery time limit was reached; completed results were retained.'
      );
    return {
      proposals: [...trueTargets.entries()].map(([index, trueTarget]) => ({
        waypoint: points[index],
        trueTarget,
        falseTarget: falseTargets.get(index) ?? null,
      })),
      warnings,
    };
  } finally {
    globalThis.clearTimeout(deadline);
    options.signal?.removeEventListener('abort', abortFromCaller);
  }
}
