import { describe, expect, it } from 'vitest';
import { GURS_DOF025_LAYER, orthophotoCoverage, orthophotoRequestUrl } from '../src/orthophoto';

const model = {
  altitudeM: 100,
  focalLength35Mm: 60,
  depressionDeg: 45,
  aspectRatio: 16 / 9,
};

describe('DOF025 coverage', () => {
  it('models the angled 100 m AGL camera footprint', () => {
    const coverage = orthophotoCoverage(model);
    expect(coverage.widthM).toBeCloseTo(102.08, 1);
    expect(coverage.heightM).toBeCloseTo(69.48, 1);
  });

  it('creates a high-resolution GURS WMS request with the matching aspect ratio', () => {
    const coverage = orthophotoCoverage(model);
    const url = new URL(orthophotoRequestUrl(46.63, 16.18, coverage));
    expect(url.origin).toBe('https://ipi.eprostor.gov.si');
    expect(url.searchParams.get('LAYERS')).toBe(GURS_DOF025_LAYER);
    expect(url.searchParams.get('CRS')).toBe('CRS:84');
    expect(url.searchParams.get('FORMAT')).toBe('image/jpeg');
    expect(url.searchParams.get('WIDTH')).toBe('1600');
    expect(Number(url.searchParams.get('HEIGHT'))).toBeGreaterThan(1000);
  });

  it('rejects near-horizon models with impractically large footprints', () => {
    expect(() =>
      orthophotoCoverage({
        altitudeM: 1000,
        focalLength35Mm: 10,
        depressionDeg: 46,
        aspectRatio: 16 / 9,
      })
    ).toThrow(/footprint exceeds/);
  });
});
