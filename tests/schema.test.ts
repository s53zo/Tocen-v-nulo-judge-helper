import Ajv2020 from 'ajv/dist/2020.js';
import { describe, expect, it } from 'vitest';
import photoSchema from '../schemas/photo-summary.schema.json';
import routeSchema from '../schemas/route-summary.schema.json';
import { photoSummaryJson } from '../src/photo-output';

describe('export schemas', () => {
  const ajv = new Ajv2020({ strict: true, formats: { 'date-time': true } });

  it('compiles the published schemas', () => {
    expect(() => ajv.compile(photoSchema)).not.toThrow();
    expect(() => ajv.compile(routeSchema)).not.toThrow();
  });

  it('validates a generated empty photo summary', () => {
    const validate = ajv.compile(photoSchema);
    const summary = photoSummaryJson([], {
      status: 'ok',
      findings: [],
      violationCount: 0,
      warningCount: 0,
      enroutePhotoCount: 0,
      routeTaskCount: 0,
    });
    expect(validate(summary), JSON.stringify(validate.errors)).toBe(true);
  });
});
