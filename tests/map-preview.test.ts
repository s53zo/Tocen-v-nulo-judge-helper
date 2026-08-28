import { describe, expect, it } from 'vitest';
import { computePreviewGeometry, tilesForGeometry, waitForAbortSignal } from '../src/map-preview';

describe('bounded map preview geometry', () => {
  it('maps a PDF crop into the top-left raster coordinate system', () => {
    const geometry = computePreviewGeometry(2000, 1000, 1000, 500, {
      minX: 250,
      minY: 100,
      maxX: 750,
      maxY: 400,
    });
    expect(geometry).toEqual({
      sourceX: 500,
      sourceY: 200,
      sourceWidth: 1000,
      sourceHeight: 600,
      outputWidth: 1000,
      outputHeight: 600,
    });
  });

  it('enforces output edge and pixel limits', () => {
    const geometry = computePreviewGeometry(
      20_000,
      10_000,
      1000,
      500,
      { minX: 0, minY: 0, maxX: 1000, maxY: 500 },
      1800,
      3_000_000
    );
    expect(Math.max(geometry.outputWidth, geometry.outputHeight)).toBeLessThanOrEqual(1800);
    expect(geometry.outputWidth * geometry.outputHeight).toBeLessThanOrEqual(3_000_000);
  });

  it('rejects crops outside the page', () => {
    expect(() =>
      computePreviewGeometry(2000, 1000, 1000, 500, { minX: -1, minY: 0, maxX: 100, maxY: 100 })
    ).toThrow('within');
  });

  it('loads only tiles intersecting the high-resolution crop', () => {
    const geometry = computePreviewGeometry(4096, 3072, 1000, 750, {
      minX: 250,
      minY: 250,
      maxX: 750,
      maxY: 500,
    });
    expect(
      tilesForGeometry(geometry, {
        baseUrl: 'https://example.test/tiles/',
        width: 4096,
        height: 3072,
        tileSize: 1024,
        columns: 4,
        rows: 3,
      })
    ).toEqual([
      { column: 1, row: 1 },
      { column: 2, row: 1 },
    ]);
  });

  it('rejects a pending preview stage when it is cancelled', async () => {
    const controller = new AbortController();
    const pending = new Promise<void>(() => undefined);
    const result = waitForAbortSignal(pending, controller.signal);
    controller.abort('cancelled');
    await expect(result).rejects.toBe('cancelled');
  });
});
