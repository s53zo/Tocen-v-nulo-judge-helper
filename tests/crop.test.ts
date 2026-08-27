import { describe, expect, it } from 'vitest';
import { chooseTrueScaleCropPage } from '../src/crop';

const A4 = [595.28, 841.89] as const;

describe('true-scale crop page selection', () => {
  it('uses an A4 orientation when the content fits', () => {
    expect(chooseTrueScaleCropPage(500, 700, A4).format).toBe('A4 portrait');
    expect(chooseTrueScaleCropPage(700, 500, A4).format).toBe('A4 landscape');
  });

  it('creates a custom true-scale page when a route cannot fit A4', () => {
    expect(chooseTrueScaleCropPage(1500, 600, A4)).toEqual({
      width: 1500,
      height: 600,
      format: 'custom true-scale',
    });
  });

  it('rejects invalid dimensions', () => {
    expect(() => chooseTrueScaleCropPage(Number.NaN, 100, A4)).toThrow('finite');
    expect(() => chooseTrueScaleCropPage(0, 100, A4)).toThrow('positive');
  });
});
