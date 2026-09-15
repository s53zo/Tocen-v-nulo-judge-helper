import { describe, expect, it, vi } from 'vitest';
import { buildRoute, type Waypoint } from '../src/domain';
import {
  buildOverpassCoreQuery,
  buildOverpassLegQueries,
  buildOverpassPhotoQueries,
  createSeededRandom,
  discoverOsmPhotoCandidates,
  fetchOsmControlPhotoProposals,
  fetchOverpassPhotoData,
  type OverpassResponse,
  selectOsmPhotoCandidates,
  targetSelectionIssue,
  validateOverpassResponse,
} from '../src/osm-photo-candidates';

const points: Waypoint[] = [
  ['SP', 46.6, 16.0],
  ['TP1', 46.6, 16.1],
  ['FP', 46.6, 16.2],
];
const route = buildRoute(points);

const data: OverpassResponse = {
  elements: [
    {
      type: 'way',
      id: 10,
      center: { lat: 46.6, lon: 16.03 },
      tags: { bridge: 'yes', highway: 'tertiary', name: 'River bridge' },
    },
    {
      type: 'way',
      id: 11,
      center: { lat: 46.6002, lon: 16.055 },
      tags: { leisure: 'pitch', name: 'Football field' },
    },
    {
      type: 'node',
      id: 12,
      lat: 46.6,
      lon: 16.004,
      tags: { historic: 'castle', name: 'Too close to SP' },
    },
    {
      type: 'way',
      id: 20,
      nodes: [201, 202, 203],
      geometry: [
        { lat: 46.599, lon: 16.069 },
        { lat: 46.6, lon: 16.07 },
        { lat: 46.601, lon: 16.071 },
      ],
      tags: { highway: 'tertiary', name: 'Main road' },
    },
    {
      type: 'way',
      id: 21,
      nodes: [204, 202, 205],
      geometry: [
        { lat: 46.601, lon: 16.069 },
        { lat: 46.6, lon: 16.07 },
        { lat: 46.599, lon: 16.071 },
      ],
      tags: { highway: 'residential', name: 'Side road' },
    },
    {
      type: 'node',
      id: 30,
      lat: 46.6,
      lon: 16.13,
      tags: { railway: 'level_crossing', name: 'Rail crossing' },
    },
    {
      type: 'way',
      id: 31,
      center: { lat: 46.6002, lon: 16.16 },
      tags: { amenity: 'school', building: 'school', name: 'Route school' },
    },
    {
      type: 'way',
      id: 32,
      center: { lat: 46.6, lon: 16.18 },
      tags: { natural: 'scrub' },
    },
  ],
};

describe('Overpass photo query', () => {
  it('creates repeatable but salt-specific selection randomness', () => {
    const first = createSeededRandom('route-mix-a');
    const repeated = createSeededRandom('route-mix-a');
    const different = createSeededRandom('route-mix-b');
    const firstSequence = Array.from({ length: 6 }, () => first());
    expect(Array.from({ length: 6 }, () => repeated())).toEqual(firstSequence);
    expect(Array.from({ length: 6 }, () => different())).not.toEqual(firstSequence);
  });

  it('requests identifiable features and road geometry without vegetation candidates', () => {
    const coreQuery = buildOverpassCoreQuery(points);
    const legQueries = buildOverpassLegQueries(points, 0);
    const query = [coreQuery, ...buildOverpassPhotoQueries(points)].join('\n');
    expect(query).toContain('["bridge"]');
    expect(query).toContain('["railway"="level_crossing"]');
    expect(query).toContain('["leisure"~"^(pitch|sports_centre)$"]');
    expect(query).toContain('["waterway"~"^(river|canal|stream)$"]');
    expect(query).toContain('["highway"~"^(secondary|tertiary|unclassified|residential)$"]');
    expect(query).not.toContain('farmland');
    expect(query).not.toContain('scrub');
    expect(query).not.toContain('highway"="crossing');
    expect(query).toContain('out body center');
    expect(query).toContain('out body geom');
    expect(coreQuery.match(/\["bridge"\]/g)?.length).toBeGreaterThan(1);
    expect(legQueries.features).toContain('(poly:"');
    expect(legQueries.features.match(/\["bridge"\]/g)).toHaveLength(1);
    expect(legQueries.features).not.toMatch(/\(-?\d+\.\d+,-?\d+\.\d+,-?\d+\.\d+,-?\d+\.\d+\)/);
  });

  it('rejects malformed unbounded OSM tag values at runtime', () => {
    expect(() =>
      validateOverpassResponse({
        elements: [{ type: 'node', id: 1, lat: 46.6, lon: 16, tags: { name: 5 } }],
      })
    ).toThrow(/tag text/);
  });

  it('posts the query and validates the Overpass response', async () => {
    const fetcher = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(JSON.stringify({ elements: [] }), { status: 200 })
    );
    const progress: string[] = [];
    await expect(
      fetchOverpassPhotoData(points, {
        fetcher: fetcher as typeof fetch,
        onProgress: ({ legIndex, stage, state }) => progress.push(`${legIndex}:${stage}:${state}`),
      })
    ).resolves.toEqual({ elements: [] });
    expect(fetcher).toHaveBeenCalledTimes(6);
    expect(progress).toContain('0:features:started');
    expect(progress).toContain('1:buildings:completed');
    const init = fetcher.mock.calls[0][1] as RequestInit;
    expect(init.method).toBe('POST');
    expect(String(init.body)).toContain('data=');
  });

  it('runs route-leg requests sequentially against the preferred healthy endpoint', async () => {
    let activeRequests = 0;
    let maximumActiveRequests = 0;
    const requestedHosts: string[] = [];
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      activeRequests += 1;
      maximumActiveRequests = Math.max(maximumActiveRequests, activeRequests);
      requestedHosts.push(new URL(String(input)).host);
      await new Promise((resolve) => setTimeout(resolve, 2));
      activeRequests -= 1;
      return new Response(JSON.stringify({ elements: [] }), { status: 200 });
    });
    await fetchOverpassPhotoData(points, { fetcher: fetcher as typeof fetch });
    expect(maximumActiveRequests).toBe(1);
    expect(new Set(requestedHosts)).toEqual(new Set(['maps.mail.ru']));
  });

  it('defers a failed leg until the other legs run, then retries it on an alternative', async () => {
    const requestedHosts: string[] = [];
    let firstRequest = true;
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      requestedHosts.push(new URL(String(input)).host);
      if (firstRequest) {
        firstRequest = false;
        return new Response('', { status: 504 });
      }
      return new Response(JSON.stringify({ elements: [] }), { status: 200 });
    });
    const progress: string[] = [];
    await fetchOverpassPhotoData(points, {
      fetcher: fetcher as typeof fetch,
      retryDelayMs: 0,
      onProgress: ({ legIndex, stage, state }) => progress.push(`${legIndex}:${stage}:${state}`),
    });
    expect(requestedHosts.slice(0, 2)).toEqual(['maps.mail.ru', 'maps.mail.ru']);
    expect(requestedHosts[2]).toBe('overpass.private.coffee');
    expect(progress).toContain('0:features:retrying');
    expect(progress).toContain('0:features:recovered');
  });

  it('retries failed legs fairly in endpoint rounds', async () => {
    const requestedHosts: string[] = [];
    let requestCount = 0;
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      requestedHosts.push(new URL(String(input)).host);
      requestCount += 1;
      if (requestCount <= 2) return new Response('', { status: 504 });
      return new Response(JSON.stringify({ elements: [] }), { status: 200 });
    });
    await fetchOverpassPhotoData(points, {
      fetcher: fetcher as typeof fetch,
      retryDelayMs: 0,
    });
    expect(requestedHosts.slice(0, 4)).toEqual([
      'maps.mail.ru',
      'maps.mail.ru',
      'overpass.private.coffee',
      'overpass.private.coffee',
    ]);
  });

  it('skips per-leg building fallback after enough stronger targets are found', async () => {
    const fetcher = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(JSON.stringify(data), { status: 200 })
    );
    const progress: string[] = [];
    await fetchOverpassPhotoData(points, {
      route,
      requestedCount: 4,
      splitAfterM: route.legs[0].length,
      fetcher: fetcher as typeof fetch,
      onProgress: ({ legIndex, stage, state }) => progress.push(`${legIndex}:${stage}:${state}`),
    });
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(progress).toContain('0:roads:skipped');
    expect(progress).toContain('1:buildings:skipped');
  });

  it('returns completed partial results when the discovery deadline is reached', async () => {
    const fetcher = vi.fn(
      async (_input: RequestInfo | URL, init?: RequestInit) =>
        await new Promise<Response>((_resolve, reject) => {
          const signal = init?.signal;
          signal?.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
        })
    );
    const result = await fetchOverpassPhotoData(points, {
      route,
      fetcher: fetcher as typeof fetch,
      deadlineMs: 5,
    });
    expect(result.elements).toEqual([]);
    expect(result.warnings?.join(' ')).toContain('time limit');
  });

  it('distinguishes explicit cancellation from a graceful deadline cutoff', async () => {
    const controller = new AbortController();
    const fetcher = vi.fn(
      async (_input: RequestInfo | URL, init?: RequestInit) =>
        await new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
        })
    );
    const result = fetchOverpassPhotoData(points, {
      route,
      fetcher: fetcher as typeof fetch,
      signal: controller.signal,
    });
    controller.abort(new Error('cancelled by test'));
    await expect(result).rejects.toThrow(/cancelled by test/);
  });

  it('proposes true SP/FP photos and a rule-separated false option for a TP', async () => {
    const controlData = {
      elements: [
        {
          type: 'way',
          id: 100,
          center: { lat: 46.6002, lon: 16.001 },
          tags: { bridge: 'yes', name: 'SP bridge' },
        },
        {
          type: 'way',
          id: 101,
          center: { lat: 46.6002, lon: 16.101 },
          tags: { bridge: 'yes', name: 'TP bridge' },
        },
        {
          type: 'way',
          id: 102,
          center: { lat: 46.6002, lon: 16.199 },
          tags: { bridge: 'yes', name: 'FP bridge' },
        },
        {
          type: 'way',
          id: 103,
          center: { lat: 46.6, lon: 16.13 },
          tags: { bridge: 'yes', name: 'False bridge' },
        },
      ],
    };
    const fetcher = vi.fn(async () => new Response(JSON.stringify(controlData), { status: 200 }));
    const result = await fetchOsmControlPhotoProposals(points, {
      fetcher: fetcher as typeof fetch,
      retryDelayMs: 0,
    });
    expect(result.warnings).toEqual([]);
    expect(result.proposals).toHaveLength(3);
    expect(result.proposals[0].trueTarget.name).toBe('SP bridge');
    expect(result.proposals[0].trueTarget.latitude).toBe(points[0][1]);
    expect(result.proposals[0].trueTarget.longitude).toBe(points[0][2]);
    expect(result.proposals[0].trueTarget.distanceFromWaypointM).toBeGreaterThan(0);
    expect(result.proposals[0].falseTarget).toBeNull();
    expect(result.proposals[1].trueTarget.name).toBe('TP bridge');
    expect(result.proposals[1].trueTarget.latitude).toBe(points[1][1]);
    expect(result.proposals[1].trueTarget.longitude).toBe(points[1][2]);
    expect(result.proposals[1].falseTarget?.name).toBe('False bridge');
    expect(result.proposals[1].falseTarget?.source.correctObjectLatitude).toBe(points[1][1]);
    expect(result.proposals[1].falseTarget?.source.correctObjectLongitude).toBe(points[1][2]);
    expect(result.proposals[1].falseTarget?.distanceFromCorrectM).toBeGreaterThanOrEqual(1852);
    expect(result.proposals[1].falseTarget?.distanceFromCorrectM).toBeLessThanOrEqual(10 * 1852);
    expect(result.proposals[2].trueTarget.name).toBe('FP bridge');
    expect(result.proposals[2].trueTarget.latitude).toBe(points[2][1]);
    expect(result.proposals[2].trueTarget.longitude).toBe(points[2][2]);
    expect(result.proposals[2].falseTarget).toBeNull();
  });
});

describe('OSM candidate discovery and selection', () => {
  it('derives junctions, removes control-clearance and vegetation entries, and scores targets', () => {
    const candidates = discoverOsmPhotoCandidates(data, route, points);
    expect(candidates.some((candidate) => candidate.featureType === '4-arm road junction')).toBe(true);
    expect(candidates.some((candidate) => candidate.name === 'Too close to SP')).toBe(false);
    expect(candidates.some((candidate) => candidate.featureType.includes('scrub'))).toBe(false);
    expect(candidates[0].confidence).toBe('excellent');
    expect(candidates.every((candidate) => candidate.lateralDistanceM <= 250)).toBe(true);
  });

  it('selects both route halves with spacing and validates manual review choices', () => {
    const candidates = discoverOsmPhotoCandidates(data, route, points);
    const selected = selectOsmPhotoCandidates(candidates, 4, route.legs[0].length, () => 0.5);
    expect(selected).toHaveLength(4);
    expect(selected.filter((candidate) => (candidate.alongRouteM ?? 0) <= route.legs[0].length)).toHaveLength(
      2
    );
    expect(selected.filter((candidate) => (candidate.alongRouteM ?? 0) > route.legs[0].length)).toHaveLength(
      2
    );
    expect(targetSelectionIssue(selected, 4, route.legs[0].length)).toBeNull();
    expect(targetSelectionIssue(selected.slice(0, 2), 4, route.legs[0].length)).toMatch(
      /selection balance warning/i
    );
  });

  it('has no per-leg maximum but requires at least one target on every leg', () => {
    const base = discoverOsmPhotoCandidates(data, route, points)[0];
    const firstLegOnly = [2000, 4000, 6000, 8000].map((alongRouteM, index) => ({
      ...base,
      id: `same-leg/${index}`,
      alongRouteM,
      category: `category-${index}`,
      routeLegIndex: 0,
      routeLegName: 'SP–TP1',
      routeLegCount: 2,
      routeLegBoundariesM: [0, route.legs[1].cumulativeStart, route.totalDistance],
    }));
    const selected = selectOsmPhotoCandidates(firstLegOnly, 4, null, () => 0.5);
    expect(selected).toHaveLength(4);
    expect(targetSelectionIssue(selected, 4, null)).toMatch(/at least one target.*route leg/i);
  });

  it('calculates balancing quotas separately on each side of the handout split', () => {
    const base = discoverOsmPhotoCandidates(data, route, points)[0];
    const boundaries = [0, 2000, 4000, 6000, 8000, 10_000, 12_000, 14_000];
    const positions = [1000, 3000, 5000, 7000, 8300, 9100, 10_100, 10_700, 11_300, 12_100, 12_700, 13_300];
    const candidates = positions.map((alongRouteM, index) => {
      const routeLegIndex = Math.min(6, Math.floor(alongRouteM / 2000));
      return {
        ...base,
        id: `balanced/${index}`,
        alongRouteM,
        category: `category-${index}`,
        routeLegIndex,
        routeLegName: `L${routeLegIndex}`,
        routeLegCount: 7,
        routeLegBoundariesM: boundaries,
      };
    });
    const selected = selectOsmPhotoCandidates(candidates, 12, 10_000, () => 0.5);
    expect(selected).toHaveLength(12);
    expect(selected.map((candidate) => candidate.alongRouteM)).toEqual(
      [...selected]
        .map((candidate) => candidate.alongRouteM)
        .sort((left, right) => (left ?? 0) - (right ?? 0))
    );
    const beforeLegCounts = [0, 1, 2, 3, 4].map(
      (legIndex) => selected.filter((candidate) => candidate.routeLegIndex === legIndex).length
    );
    expect(Math.max(...beforeLegCounts) - Math.min(...beforeLegCounts)).toBeLessThanOrEqual(1);
    expect(selected.filter((candidate) => candidate.routeLegIndex === 5)).toHaveLength(3);
    expect(selected.filter((candidate) => candidate.routeLegIndex === 6)).toHaveLength(3);
    expect(targetSelectionIssue(selected, 12, 10_000)).toBeNull();
  });
});
