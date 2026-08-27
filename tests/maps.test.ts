import { describe, expect, it } from 'vitest';
import rawPresets from '../src/map-presets.json';
import { loadMapPresets } from '../src/maps';

describe('map preset validation', () => {
  it('loads repository presets using local URLs', () => {
    const presets = loadMapPresets(rawPresets, new URL('https://example.test/app/'));
    expect(presets.vfr.type).toBe('pdf');
    if (presets.vfr.type === 'pdf') {
      expect(presets.vfr.url).toBe('https://example.test/app/maps/00_VFRspredaj_25_SC_WEB_flat.pdf');
      expect(presets.vfr.previewUrl).toBe('https://example.test/app/maps/previews/vfr.webp');
    }
  });

  it('rejects affine presets with unaccounted control points', () => {
    expect(() =>
      loadMapPresets(
        {
          vfr: {
            label: 'Bad calibration',
            type: 'pdf',
            edition: 'test',
            fileName: 'map.pdf',
            assetPath: 'map.pdf',
            previewAssetPath: 'map.webp',
            baseWidth: 100,
            baseHeight: 100,
            scaleDenominator: 250000,
            controlPoints: [],
          },
        },
        new URL('https://example.test/')
      )
    ).toThrow(/exactly three/);
  });
});
