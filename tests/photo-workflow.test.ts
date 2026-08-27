import { describe, expect, it } from 'vitest';
import { alphabeticIdentifier, validateManualNumber } from '../src/photo-workflow';

describe('photo workflow validation', () => {
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
});
