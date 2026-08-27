import fontkit from '@pdf-lib/fontkit';
import { PDFDocument, type PDFFont, rgb } from 'pdf-lib';
import type { PhotoComplianceSummary, PhotoRecord } from './photo-types';

export interface HandoutPhoto {
  record: PhotoRecord;
  jpeg: Uint8Array;
}

export interface HandoutOptions {
  splitWaypoint: string;
  includeSummary: boolean;
}

const A4: [number, number] = [595.28, 841.89];

function safeText(value: string): string {
  return Array.from(value.normalize('NFC'), (character) =>
    character.charCodeAt(0) < 32 ? ' ' : character
  ).join('');
}

function splitIndex(photo: PhotoRecord, split: string): number {
  const match = photo.analysis?.legId.match(/-(?:TP)?(\d+|FP)$/i);
  const splitMatch = split.match(/TP(\d+)/i);
  if (!match || !splitMatch) return 0;
  const destination = match[1].toUpperCase() === 'FP' ? Number.POSITIVE_INFINITY : Number(match[1]);
  return destination > Number(splitMatch[1]) ? 1 : 0;
}

function drawWrapped(
  page,
  font: PDFFont,
  text: string,
  x: number,
  y: number,
  maxWidth: number,
  size: number,
  color = rgb(0.12, 0.15, 0.18)
) {
  const words = safeText(text).split(/\s+/);
  let line = '';
  let row = 0;
  for (const word of words) {
    const candidate = line ? `${line} ${word}` : word;
    if (font.widthOfTextAtSize(candidate, size) > maxWidth && line) {
      page.drawText(line, { x, y: y - row * (size + 2), font, size, color });
      row += 1;
      line = word;
    } else line = candidate;
  }
  if (line) page.drawText(line, { x, y: y - row * (size + 2), font, size, color });
}

export async function buildPhotoHandout(
  photos: HandoutPhoto[],
  compliance: PhotoComplianceSummary,
  fontBytes: Uint8Array,
  options: HandoutOptions
): Promise<Uint8Array> {
  const document = await PDFDocument.create();
  document.registerFontkit(fontkit);
  const font = await document.embedFont(fontBytes, { subset: true });
  const sections = [
    {
      title: `Photos before ${options.splitWaypoint}`,
      items: photos.filter(({ record }) => splitIndex(record, options.splitWaypoint) === 0),
    },
    {
      title: `Photos after ${options.splitWaypoint}`,
      items: photos.filter(({ record }) => splitIndex(record, options.splitWaypoint) === 1),
    },
  ].filter((section) => section.items.length > 0);
  const margin = 20;
  const headerHeight = 38;
  const gap = 14;
  const slotHeight = (A4[1] - margin * 2 - headerHeight - gap) / 2;
  for (const section of sections) {
    for (let index = 0; index < section.items.length; index += 2) {
      const page = document.addPage(A4);
      page.drawText(section.title, {
        x: margin,
        y: A4[1] - margin - 15,
        size: 15,
        font,
        color: rgb(0.08, 0.26, 0.2),
      });
      for (let slot = 0; slot < 2; slot += 1) {
        const item = section.items[index + slot];
        if (!item) continue;
        const top = A4[1] - margin - headerHeight - slot * (slotHeight + gap);
        const metaHeight = 34;
        const imageBoxHeight = slotHeight - metaHeight;
        const image = await document.embedJpg(item.jpeg);
        const fit = Math.min((A4[0] - margin * 2) / image.width, imageBoxHeight / image.height);
        const width = image.width * fit;
        const height = image.height * fit;
        const x = (A4[0] - width) / 2;
        const y = top - height;
        page.drawImage(image, { x, y, width, height });
        page.drawRectangle({ x, y, width, height, borderWidth: 0.8, borderColor: rgb(0.72, 0.75, 0.76) });
        const record = item.record;
        const badge = safeText(record.identifier || '?');
        const badgeSize = 20;
        const badgeWidth = Math.max(34, font.widthOfTextAtSize(badge, badgeSize) + 14);
        page.drawRectangle({
          x: x + 8,
          y: y + height - 37,
          width: badgeWidth,
          height: 29,
          color: rgb(1, 1, 1),
          opacity: 0.9,
          borderColor: rgb(0.08, 0.26, 0.2),
          borderWidth: 1,
        });
        page.drawText(badge, {
          x: x + 15,
          y: y + height - 31,
          size: badgeSize,
          font,
          color: rgb(0.08, 0.26, 0.2),
        });
        const status = record.findings.some((finding) => finding.severity === 'violation')
          ? 'AGAINST THE RULES'
          : record.findings.some((finding) => finding.severity === 'warning')
            ? 'MANUAL REVIEW'
            : 'OK';
        drawWrapped(
          page,
          font,
          `${record.identifier || '?'} · ${record.classification}${record.linkedWaypoint ? ` · ${record.linkedWaypoint}` : ''} · ${status} · ${record.fileName}`,
          margin,
          y - 13,
          A4[0] - margin * 2,
          8
        );
      }
      page.drawText(`Page ${document.getPageCount()}`, {
        x: A4[0] - 70,
        y: 8,
        size: 7,
        font,
        color: rgb(0.4, 0.4, 0.4),
      });
    }
  }
  if (options.includeSummary) {
    const page = document.addPage(A4);
    page.drawText('Photo compliance summary', {
      x: margin,
      y: A4[1] - 42,
      size: 18,
      font,
      color: rgb(0.08, 0.26, 0.2),
    });
    page.drawText(
      `${compliance.status.toUpperCase()} · ${photos.length} photos · ${compliance.violationCount} violations · ${compliance.warningCount} manual checks`,
      { x: margin, y: A4[1] - 68, size: 10, font }
    );
    let y = A4[1] - 94;
    for (const finding of compliance.findings.filter((item) => item.severity !== 'pass')) {
      drawWrapped(
        page,
        font,
        `${finding.rule} · ${finding.affected}: ${finding.measured}; permitted ${finding.permitted}.`,
        margin,
        y,
        A4[0] - margin * 2,
        8
      );
      y -= 30;
      if (y < 30) break;
    }
  }
  if (document.getPageCount() === 0) {
    const page = document.addPage(A4);
    page.drawText('No photos were selected.', { x: margin, y: A4[1] - 42, size: 14, font });
  }
  return document.save();
}
