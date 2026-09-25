import { PDFDocument, PrintScaling, rgb, StandardFonts } from 'pdf-lib';

const MM = 72 / 25.4;

export function rulerSpacing(scaleDenominator: number, knots: number) {
  if (!Number.isFinite(scaleDenominator) || scaleDenominator <= 0 || !Number.isFinite(knots) || knots <= 0) {
    throw new Error('Rulers require a positive map scale and groundspeed.');
  }
  const nauticalMileMm = (1852 * 1000) / scaleDenominator;
  return { nauticalMileMm, minuteMm: (nauticalMileMm * knots) / 60 };
}

export async function buildPilotRulers(options: {
  widthMm: number;
  heightMm: number;
  scaleDenominator: number;
  speeds: Array<{ label: string; knots: number }>;
}): Promise<Uint8Array> {
  const { widthMm, heightMm, scaleDenominator, speeds } = options;
  if (![widthMm, heightMm].every((value) => Number.isFinite(value) && value >= 100) || !speeds.length) {
    throw new Error('Rulers require a valid paper size and at least one speed.');
  }
  const doc = await PDFDocument.create();
  const font = await doc.embedFont(StandardFonts.Helvetica);
  const bold = await doc.embedFont(StandardFonts.HelveticaBold);
  const preferences = doc.catalog.getOrCreateViewerPreferences();
  preferences.setPrintScaling(PrintScaling.None);
  preferences.setPickTrayByPDFSize(true);
  doc.setTitle('Pilot navigation rulers');
  const width = widthMm * MM;
  const height = heightMm * MM;
  const left = 10 * MM;
  const length = width - 2 * left;
  let page = doc.addPage([width, height]);
  let top = 0;
  const header = () => {
    page.drawText(`Pilot rulers - 1:${scaleDenominator.toLocaleString('en-US')}`, {
      x: left,
      y: height - 13 * MM,
      size: 12,
      font: bold,
    });
    page.drawText(
      'Print at Actual size / 100%. Cut on the dotted outlines. Top: minutes. Bottom: nautical miles.',
      {
        x: left,
        y: height - 19 * MM,
        size: 8,
        font,
      }
    );
    page.drawText('100 mm print check', { x: left, y: 14 * MM, size: 7, font });
    page.drawLine({
      start: { x: left, y: 10 * MM },
      end: { x: left + 100 * MM, y: 10 * MM },
      thickness: 0.7,
    });
    for (const x of [left, left + 100 * MM]) {
      page.drawLine({ start: { x, y: 8.5 * MM }, end: { x, y: 11.5 * MM }, thickness: 0.7 });
    }
    top = height - 30 * MM;
  };
  header();
  for (const speed of speeds) {
    if (top - 10 * MM < 24 * MM) {
      page = doc.addPage([width, height]);
      header();
    }
    const spacing = rulerSpacing(scaleDenominator, speed.knots);
    page.drawText(`${speed.label} - 1:${scaleDenominator.toLocaleString('en-US')}`, {
      x: left,
      y: top + 2 * MM,
      size: 8,
      font: bold,
    });
    page.drawRectangle({
      x: left,
      y: top - 10 * MM,
      width: length,
      height: 10 * MM,
      borderWidth: 0.4,
      borderColor: rgb(0.45, 0.45, 0.45),
      borderDashArray: [2, 2],
    });
    // A common zero on both edges permits direct distance/time comparisons.
    for (const [isMinutes, unitMm] of [
      [true, spacing.minuteMm],
      [false, spacing.nauticalMileMm],
    ] as const) {
      const major = unitMm * MM;
      const subdivisions = major >= 4 * MM ? 4 : 1;
      const labelEvery = Math.max(1, Math.ceil((5 * MM) / major));
      const count = Math.floor(length / (major / subdivisions));
      for (let tick = 0; tick <= count; tick += 1) {
        const x = left + (tick * major) / subdivisions;
        const whole = tick % subdivisions === 0;
        const tickLength = (whole ? 2 : tick % 2 === 0 ? 1.4 : 0.9) * MM;
        const edge = isMinutes ? top : top - 10 * MM;
        page.drawLine({
          start: { x, y: edge },
          end: { x, y: edge + (isMinutes ? -tickLength : tickLength) },
          thickness: whole ? 0.6 : 0.35,
        });
        if (whole && (tick / subdivisions) % labelEvery === 0) {
          const text = String(tick / subdivisions);
          const textWidth = font.widthOfTextAtSize(text, 6);
          const labelX = Math.max(left + 0.5, Math.min(x - textWidth / 2, left + length - textWidth - 0.5));
          page.drawText(text, { x: labelX, y: isMinutes ? top - 3.8 * MM : top - 7.8 * MM, size: 6, font });
        }
      }
    }
    const stripLabel = `${speed.label} | 1:${scaleDenominator} | MIN above / NM below`;
    page.drawText(stripLabel, {
      x: left + (length - bold.widthOfTextAtSize(stripLabel, 4.5)) / 2,
      y: top - 5.7 * MM,
      size: 4.5,
      font: bold,
    });
    top -= 19 * MM;
  }
  return doc.save();
}
