import { describe, expect, it } from 'vitest';
import type { PhotoRecord } from '../src/photo-types';
import {
  alphabeticIdentifier,
  createIdentifierMixRandom,
  jpegDimensions,
  lettersRevealRouteOrder,
  mixedRouteIndices,
  validateManualNumber,
} from '../src/photo-workflow';

describe('photo workflow validation', () => {
  it('reads JPEG dimensions before browser decoding', () => {
    const jpeg = Uint8Array.from([
      0xff, 0xd8, 0xff, 0xc0, 0x00, 0x0b, 0x08, 0x04, 0x00, 0x08, 0x00, 0x01, 0x01, 0x11, 0x00, 0xff, 0xd9,
    ]).buffer;
    expect(jpegDimensions(jpeg)).toEqual([2048, 1024]);
    expect(jpegDimensions(Uint8Array.from([1, 2, 3]).buffer)).toBeNull();
  });
  it('validates coordinate and metadata domains', () => {
    expect(validateManualNumber('latitude', '90')).toEqual({ value: 90, error: null });
    expect(validateManualNumber('latitude', '90.1').error).toContain('between');
    expect(validateManualNumber('longitude', '-180')).toEqual({ value: -180, error: null });
    expect(validateManualNumber('longitude', '-180.1').error).toContain('between');
    expect(validateManualNumber('headingDeg', '359.9')).toEqual({ value: 359.9, error: null });
    expect(validateManualNumber('headingDeg', '360').error).toContain('below');
    expect(validateManualNumber('altitudeAglFt', '0').error).toContain('greater');
    expect(validateManualNumber('focalLength35Mm', 'NaN').error).toContain('finite');
  });

  it('generates unique identifiers beyond Z', () => {
    expect(alphabeticIdentifier(0)).toBe('A');
    expect(alphabeticIdentifier(25)).toBe('Z');
    expect(alphabeticIdentifier(26)).toBe('AA');
    expect(alphabeticIdentifier(51)).toBe('AZ');
  });

  it('forces a mixed route order when a shuffle returns forward or reverse order', () => {
    expect(mixedRouteIndices(4, () => 0.999)).toEqual([2, 3, 0, 1]);
    const reverseRandom = [0, 0.999];
    expect(mixedRouteIndices(3, () => reverseRandom.shift() ?? 0)).toEqual([2, 0, 1]);
    expect(mixedRouteIndices(2, () => 0.999)).toEqual([1, 0]);
  });

  it('replays identifier mixing from the exported salt', () => {
    const first = mixedRouteIndices(8, createIdentifierMixRandom('mix-salt-1'));
    expect(mixedRouteIndices(8, createIdentifierMixRandom('mix-salt-1'))).toEqual(first);
    expect(mixedRouteIndices(8, createIdentifierMixRandom('mix-salt-2'))).not.toEqual(first);
  });

  it('detects when identifiers disclose the route sequence', () => {
    const record = (identifier: string, alongRouteM: number) =>
      ({ identifier, analysis: { alongRouteM }, taskAnalysis: null }) as PhotoRecord;
    expect(lettersRevealRouteOrder([record('A', 100), record('B', 200), record('C', 300)])).toBe(true);
    expect(lettersRevealRouteOrder([record('C', 100), record('A', 200), record('B', 300)])).toBe(false);
  });
});
