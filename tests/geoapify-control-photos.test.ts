import { describe, expect, it, vi } from 'vitest';
import type { Waypoint } from '../src/domain';
import { fetchGeoapifyControlPhotoProposals } from '../src/geoapify-control-photos';

const points: Waypoint[] = [
  ['SP', 46.6, 16.0],
  ['TP1', 46.6, 16.1],
  ['FP', 46.6, 16.2],
];

function response(latitude: number, longitude: number, name: string) {
  return new Response(
    JSON.stringify({
      features: [
        {
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [longitude, latitude] },
          properties: {
            place_id: name,
            name,
            categories: ['man_made.bridge'],
          },
        },
      ],
    }),
    { status: 200, headers: { 'content-type': 'application/json' } }
  );
}

describe('Geoapify control-photo lookup', () => {
  it('uses nearby places for true controls and a 1–10 NM false candidate', async () => {
    const requestedUrls: string[] = [];
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const url = new URL(String(input));
      requestedUrls.push(url.toString());
      const isFalseLookup =
        url.searchParams.get('filter')?.includes('16.1') && url.searchParams.get('filter')?.includes('18520');
      return isFalseLookup
        ? response(46.6, 16.125, 'False bridge')
        : response(46.6002, 16.1002, 'True bridge');
    });

    const result = await fetchGeoapifyControlPhotoProposals(points, {
      apiKey: 'test-key',
      fetcher: fetcher as typeof fetch,
    });

    expect(result.warnings).toEqual([]);
    expect(result.proposals).toHaveLength(3);
    expect(result.proposals[1].trueTarget.latitude).toBe(points[1][1]);
    expect(result.proposals[1].trueTarget.longitude).toBe(points[1][2]);
    expect(result.proposals[1].trueTarget.source.correctObjectLatitude).toBe(points[1][1]);
    expect(result.proposals[1].trueTarget.source.correctObjectLongitude).toBe(points[1][2]);
    expect(result.proposals[1].falseTarget?.name).toBe('False bridge');
    expect(result.proposals[1].falseTarget?.distanceFromCorrectM).toBeGreaterThanOrEqual(1852);
    expect(result.proposals[1].falseTarget?.distanceFromCorrectM).toBeLessThanOrEqual(10 * 1852);
    expect(requestedUrls).toHaveLength(4);
    expect(requestedUrls.every((url) => url.includes('apiKey=test-key'))).toBe(true);
  });

  it('requires a key before making a request', async () => {
    await expect(fetchGeoapifyControlPhotoProposals(points, { apiKey: ' ' })).rejects.toThrow(
      /API key is required/
    );
  });
});
