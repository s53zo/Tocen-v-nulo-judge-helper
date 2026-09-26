import { describe, expect, it } from 'vitest';
import { choosePreviewScale, chooseTrueScaleCropPage } from '../src/crop';

const A4 = [595.28, 841.89] as const;
const A3 = [841.89, 1190.55] as const;

describe('true-scale crop page selection', () => {
  it('rotates a tall crop to retain A3 scale with header and footer space', () => {
    const page = chooseTrueScaleCropPage(250, 400, [210, 297], [297, 420], 36);
    expect(page?.format).toBe('A3 landscape');
    expect(page?.rotateContent).toBe(true);
  });
  it('uses an A4 orientation when the content fits', () => {
    expect(chooseTrueScaleCropPage(500, 700, A4, A3)?.format).toBe('A4 portrait');
    expect(chooseTrueScaleCropPage(700, 500, A4, A3)?.format).toBe('A4 landscape');
  });

  it('uses A3 when true-scale content cannot fit A4', () => {
    expect(chooseTrueScaleCropPage(1000, 700, A4, A3)?.format).toBe('A3 landscape');
  });

  it('rejects content that cannot fit A3 without scaling', () => {
    expect(chooseTrueScaleCropPage(1500, 600, A4, A3)).toBeNull();
  });

  it('rejects invalid dimensions', () => {
    expect(() => chooseTrueScaleCropPage(Number.NaN, 100, A4, A3)).toThrow('finite');
    expect(() => chooseTrueScaleCropPage(0, 100, A4, A3)).toThrow('positive');
  });

  it('bounds previews of very large chart pages', () => {
    const scale = choosePreviewScale(11890, 8410, 1.2, 1800, 3_000_000);
    expect(11890 * scale).toBeLessThanOrEqual(1800);
    expect(11890 * scale * (8410 * scale)).toBeLessThanOrEqual(3_000_000);
  });

  it('keeps hard caps for extreme dimensions', () => {
    const scale = choosePreviewScale(1_000_000, 1_000_000, 1.2, 1800, 3_000_000);
    expect(1_000_000 * scale).toBeLessThanOrEqual(1800);
    expect(1_000_000 * scale * (1_000_000 * scale)).toBeLessThanOrEqual(3_000_000.001);
  });
});
