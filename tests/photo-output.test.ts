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
      headingReference: { value: null },
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
        taskAnalysis: null,
        taskLatitude: { value: null, source: 'missing' },
        taskLongitude: { value: null, source: 'missing' },
        findings: [],
      },
    ] as unknown as PhotoRecord[];
    expect(photoAnalysisCsv(records)).toContain('"one,""two"".jpg"');
    expect(photoSummaryJson(records, compliance).schemaVersion).toBe(3);
  });

  it('neutralizes spreadsheet formulas while leaving numeric values numeric', () => {
    const metadata = {
      captureTime: { value: null },
      latitude: { value: -46, source: 'manual' },
      longitude: { value: null, source: 'missing' },
      gpsAltitudeMslM: { value: null },
      altitudeAglFt: { value: null },
      headingDeg: { value: null },
      headingReference: { value: null },
      focalLengthMm: { value: null },
      focalLength35Mm: { value: null },
      cameraMake: { value: null },
      cameraModel: { value: null },
      lensModel: { value: null },
    };
    const record = {
      fileName: '=HYPERLINK("bad")',
      identifier: '+SUM(1,1)',
      classification: 'reference',
      linkedWaypoint: null,
      originalMetadata: metadata,
      metadata,
      analysis: null,
      taskAnalysis: null,
      taskLatitude: { value: null, source: 'missing' },
      taskLongitude: { value: null, source: 'missing' },
      findings: [],
    } as unknown as PhotoRecord;
    const csv = photoAnalysisCsv([record]);
    expect(csv).toContain("'+SUM(1,1)");
    expect(csv).toContain("'=HYPERLINK");
    expect(csv).toContain('-46');
  });

  it('exports accepted exceptions without hiding violations', () => {
    const record = {
      id: 'photo-1',
      order: 0,
      fileName: 'accepted.jpg',
      fileSize: 10,
      contentHash: 'hash',
      identifier: 'A',
      classification: 'enroute',
      linkedWaypoint: null,
      originalMetadata: {},
      metadata: {},
      analysis: null,
      taskAnalysis: null,
      taskLatitude: { value: null, source: 'missing' },
      taskLongitude: { value: null, source: 'missing' },
      exceptionAccepted: true,
      exceptionAcceptedAt: '2026-08-28T08:00:00.000Z',
      findings: [{ severity: 'violation', code: 'test' }],
    } as unknown as PhotoRecord;
    const summary = photoSummaryJson([record], { ...compliance, status: 'against-rules', violationCount: 1 });
    expect(summary.counts.acceptedExceptions).toBe(1);
    expect(summary.photos[0]).toMatchObject({
      status: 'against-rules',
      exceptionAccepted: true,
    });
    expect(summary.photos[0].findings).toHaveLength(1);
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
      splitAfterM: 10_000,
      includeSummary: true,
    });
    const document = await PDFDocument.load(bytes);
    expect(document.getPageCount()).toBe(1);
  });

  it('paginates every handout compliance finding', async () => {
    const font = fs.readFileSync('node_modules/notosans-fontface/fonts/NotoSans-Bold.ttf');
    const manyFindings: PhotoComplianceSummary = {
      ...compliance,
      status: 'manual-review',
      warningCount: 60,
      findings: Array.from({ length: 60 }, (_, index) => ({
        photoId: null,
        severity: 'warning' as const,
        code: `finding-${index}`,
        rule: 'A2.4.5',
        affected: `photo-${index}`,
        message: `Manual finding ${index}`,
        measured: `measurement ${index}`,
        permitted: 'manual confirmation',
      })),
    };
    const bytes = await buildPhotoHandout([], manyFindings, new Uint8Array(font), {
      splitWaypoint: 'TP5',
      splitAfterM: 10_000,
      includeSummary: true,
    });
    const document = await PDFDocument.load(bytes);
    expect(document.getPageCount()).toBeGreaterThan(2);
  });
});
