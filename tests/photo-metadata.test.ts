import fs from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  inferClassification,
  markRepeatedGps,
  normalizeExifDate,
  normalizeExifTags,
  orientationOutputSize,
} from '../src/photo-metadata';

describe('photo metadata', () => {
  it('normalizes expanded EXIF and distinguishes MSL from AGL', () => {
    const metadata = normalizeExifTags({
      exif: {
        DateTimeOriginal: { description: '2025:09:19 15:51:39' },
        OffsetTimeOriginal: { description: '+02:00' },
        GPSImgDirection: { value: 370 },
        FocalLength: { value: [12, 10] },
        FocalLengthIn35mmFilm: { value: 55 },
        Orientation: { value: 6 },
        Make: { description: 'Insta360' },
      },
      gps: { Latitude: 46.5, Longitude: 16.2, Altitude: 491.5 },
    });
    expect(metadata.captureTime.value).toBe('2025-09-19T15:51:39+02:00');
    expect(metadata.headingDeg.value).toBe(10);
    expect(metadata.focalLengthMm.value).toBe(1.2);
    expect(metadata.focalLength35Mm.value).toBe(55);
    expect(metadata.gpsAltitudeMslM.note).toContain('MSL');
    expect(metadata.altitudeAglFt.value).toBeNull();
    expect(metadata.orientation.value).toBe(6);
  });

  it('marks absent timezone and absent fields as unreliable', () => {
    expect(normalizeExifDate('2025:01:02 03:04:05', null).reliable).toBe(false);
    const metadata = normalizeExifTags({});
    expect(metadata.latitude.source).toBe('missing');
    expect(metadata.headingDeg.reliable).toBe(false);
  });

  it('flags repeated GPS without changing non-repeated coordinates', () => {
    const repeated = [1, 2, 3, 4].map((index) =>
      normalizeExifTags({ gps: { Latitude: index === 4 ? 47 : 46, Longitude: 16 } })
    );
    const result = markRepeatedGps(repeated);
    expect(result[0].latitude.reliable).toBe(false);
    expect(result[2].latitude.note).toContain('3 photos');
    expect(result[3].latitude.reliable).toBe(true);
  });

  it('infers control classifications and orientation dimensions', () => {
    expect(inferClassification('IMG 001 TP4.jpg')).toEqual({
      classification: 'control-correct',
      linkedWaypoint: 'TP4',
    });
    expect(inferClassification('village.jpg').classification).toBe('enroute');
    expect(orientationOutputSize(3000, 2000, 6)).toEqual([2000, 3000]);
  });

  it('extracts the expected camera facts from a shipped historical JPEG', async () => {
    const { extractPhotoMetadata } = await import('../src/photo-metadata');
    const bytes = fs.readFileSync('examples/photos/IMG__160450_00_111.jpg');
    const file = new File([bytes], 'IMG__160450_00_111.jpg', { type: 'image/jpeg' });
    const metadata = await extractPhotoMetadata(file);
    expect(metadata.cameraMake.value).toBe('Arashi Vision');
    expect(metadata.cameraModel.value).toBe('insta360 x4');
    expect(metadata.focalLength35Mm.value).toBe(6);
    expect(metadata.gpsAltitudeMslM.value).not.toBeNull();
    expect(metadata.altitudeAglFt.value).toBeNull();
  });
});
