import fs from 'node:fs';
import { PDFDocument } from 'pdf-lib';
import { describe, expect, it } from 'vitest';
import { buildPhotoHandout } from '../src/photo-handout';
import { orientationTransform } from '../src/photo-image';
import { photoAnalysisCsv, photoSummaryJson } from '../src/photo-output';
import type { PhotoComplianceSummary, PhotoRecord } from '../src/photo-types';
import { applyPhotoClassification, hashFileSha256, isDuplicatePhoto } from '../src/photo-workflow';

const compliance: PhotoComplianceSummary = {
  status: 'ok',
  findings: [],
  violationCount: 0,
  warningCount: 0,
  enroutePhotoCount: 0,
  routeTaskCount: 0,
};

describe('photo outputs', () => {
  it('quotes CSV values and exposes a stable JSON schema', () => {
    const metadata = {
      captureTime: { value: null },
      latitude: { value: null, source: 'missing' },
      longitude: { value: null, source: 'missing' },
      gpsAltitudeMslM: { value: null },
      altitudeAglFt: { value: null },
      headingDeg: { value: null },
      focalLengthMm: { value: null },
      focalLength35Mm: { value: null },
      cameraMake: { value: null },
      cameraModel: { value: null },
      lensModel: { value: null },
    };
    const records = [
      {
        fileName: 'one,"two".jpg',
        identifier: 'A',
        classification: 'reference',
        linkedWaypoint: null,
        originalMetadata: metadata,
        metadata,
        analysis: null,
        findings: [],
      },
    ] as unknown as PhotoRecord[];
    expect(photoAnalysisCsv(records)).toContain('"one,""two"".jpg"');
    expect(photoSummaryJson(records, compliance).schemaVersion).toBe(1);
  });

  it('defines transforms for all eight EXIF orientations', () => {
    expect(Array.from({ length: 8 }, (_, index) => orientationTransform(index + 1, 300, 200))).toEqual([
      [1, 0, 0, 1, 0, 0],
      [-1, 0, 0, 1, 300, 0],
      [-1, 0, 0, -1, 300, 200],
      [1, 0, 0, -1, 0, 200],
      [0, 1, 1, 0, 0, 0],
      [0, 1, -1, 0, 200, 0],
      [0, -1, -1, 0, 200, 300],
      [0, -1, 1, 0, 0, 300],
    ]);
  });

  it('detects duplicate content and duplicate name/size metadata', async () => {
    const file = new File(['same bytes'], 'photo.jpg', { type: 'image/jpeg' });
    const hash = await hashFileSha256(file);
    const records = [{ contentHash: hash, fileName: 'different.jpg', fileSize: 999 }] as PhotoRecord[];
    expect(isDuplicatePhoto(file, hash, records)).toBe(true);
    expect(
      isDuplicatePhoto(file, 'another hash', [
        { contentHash: 'different', fileName: file.name, fileSize: file.size } as PhotoRecord,
      ])
    ).toBe(true);
  });

  it('applies explicit classification and waypoint overrides', () => {
    const record = { classification: 'enroute', linkedWaypoint: null } as PhotoRecord;
    const updated = applyPhotoClassification(record, 'control-false', 'TP3');
    expect(updated.classification).toBe('control-false');
    expect(updated.linkedWaypoint).toBe('TP3');
  });

  it('creates a valid empty handout PDF', async () => {
    const font = fs.readFileSync('node_modules/notosans-fontface/fonts/NotoSans-Bold.ttf');
    const bytes = await buildPhotoHandout([], compliance, new Uint8Array(font), {
      splitWaypoint: 'TP5',
      includeSummary: true,
    });
    const document = await PDFDocument.load(bytes);
    expect(document.getPageCount()).toBe(1);
  });
});
