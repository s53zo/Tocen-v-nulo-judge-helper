import { describe, expect, it } from 'vitest';
import {
  CANDIDATE_CATALOG_VERSION,
  catalogEntryForGeoapifyCategories,
  GEOAPIFY_CANDIDATE_CATALOG,
} from '../src/candidate-catalog';

describe('Geoapify candidate catalog', () => {
  it('contains unique provider categories and keeps specific matches ahead of parents', () => {
    const categories = GEOAPIFY_CANDIDATE_CATALOG.map((entry) => entry.geoapifyCategory);
    expect(new Set(categories).size).toBe(categories.length);
    expect(CANDIDATE_CATALOG_VERSION).toMatch(/^\d{4}-\d{2}-\d{2}/);
    expect(catalogEntryForGeoapifyCategories(['tourism.sights.castle'])?.featureType).toBe('Castle');
    expect(catalogEntryForGeoapifyCategories(['power.substation'])?.diversityGroup).toBe('power');
  });
});
