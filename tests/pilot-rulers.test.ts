import { PDFDocument, PrintScaling } from 'pdf-lib';
import { describe, expect, it } from 'vitest';
import { buildPilotRulers, rulerSpacing } from '../src/pilot-rulers';

describe('printable pilot rulers', () => {
  it('matches nautical miles and minutes to physical map scale', () => {
    expect(rulerSpacing(200000, 60)).toEqual({ nauticalMileMm: 9.26, minuteMm: 9.26 });
    expect(rulerSpacing(250000, 90).nauticalMileMm).toBeCloseTo(7.408);
    expect(rulerSpacing(250000, 90).minuteMm).toBeCloseTo(11.112);
    expect(rulerSpacing(250000, 140 / 1.852).minuteMm).toBeCloseTo(9.333333);
  });
  it('paginates every speed at the original paper dimensions and disables print scaling', async () => {
    const bytes = await buildPilotRulers({
      widthMm: 297,
      heightMm: 210,
      scaleDenominator: 250000,
      speeds: Array.from({ length: 15 }, (_, index) => ({
        label: `${50 + index * 5} kt`,
        knots: 50 + index * 5,
      })),
    });
    const document = await PDFDocument.load(bytes);
    expect(document.getPageCount()).toBe(2);
    expect(document.getPages()[0].getWidth()).toBeCloseTo((297 * 72) / 25.4);
    expect(document.catalog.getViewerPreferences()?.getPrintScaling()).toBe(PrintScaling.None);
  });
});
