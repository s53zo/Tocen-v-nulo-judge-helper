import { readFile } from 'node:fs/promises';
import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';
import Ajv2020 from 'ajv/dist/2020.js';

async function expectPdfBlob(page, selector: string): Promise<void> {
  const [download] = await Promise.all([page.waitForEvent('download'), page.locator(selector).click()]);
  const path = await download.path();
  expect(path).not.toBeNull();
  const bytes = await readFile(path as string);
  expect(bytes.subarray(0, 5).toString()).toBe('%PDF-');
}

async function downloadBytes(page, selector: string): Promise<Buffer> {
  const [download] = await Promise.all([page.waitForEvent('download'), page.locator(selector).click()]);
  const path = await download.path();
  expect(path).not.toBeNull();
  return readFile(path as string);
}

async function expectSummarySchemas(summary): Promise<void> {
  const [routeSchema, photoSchema] = await Promise.all(
    ['route-summary.schema.json', 'photo-summary.schema.json'].map(async (name) =>
      JSON.parse(await readFile(new URL(`../schemas/${name}`, import.meta.url), 'utf8'))
    )
  );
  const ajv = new Ajv2020({ strict: true, formats: { 'date-time': true } });
  const validateRoute = ajv.compile(routeSchema);
  const validatePhotos = ajv.compile(photoSchema);
  expect(validateRoute(summary), JSON.stringify(validateRoute.errors)).toBe(true);
  expect(validatePhotos(summary.photos), JSON.stringify(validatePhotos.errors)).toBe(true);
}

async function expectStructuredDownloads(page): Promise<void> {
  const summary = JSON.parse((await downloadBytes(page, '#downloadSummary')).toString());
  await expectSummarySchemas(summary);
  const header = (await downloadBytes(page, '#downloadPhotoAnalysis')).toString().split(/\r?\n/, 1)[0];
  expect(header).toContain('identifier,file_name,classification');
  expect(header).toContain('task_latitude');
}

function stripApp1Segments(jpeg: Buffer): Buffer {
  const chunks = [jpeg.subarray(0, 2)];
  let offset = 2;
  while (offset + 4 <= jpeg.length && jpeg[offset] === 0xff) {
    const marker = jpeg[offset + 1];
    if (marker === 0xda || marker === 0xd9) {
      chunks.push(jpeg.subarray(offset));
      return Buffer.concat(chunks);
    }
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      chunks.push(jpeg.subarray(offset, offset + 2));
      offset += 2;
      continue;
    }
    const segmentLength = jpeg.readUInt16BE(offset + 2);
    const end = offset + 2 + segmentLength;
    if (end > jpeg.length) break;
    if (marker !== 0xe1) chunks.push(jpeg.subarray(offset, end));
    offset = end;
  }
  chunks.push(jpeg.subarray(offset));
  return Buffer.concat(chunks);
}

function withExifOrientation(jpeg: Buffer, orientation: number): Buffer {
  const cleanJpeg = stripApp1Segments(jpeg);
  const exif = Buffer.from([
    0x45,
    0x78,
    0x69,
    0x66,
    0,
    0,
    0x49,
    0x49,
    0x2a,
    0,
    8,
    0,
    0,
    0,
    1,
    0,
    0x12,
    0x01,
    3,
    0,
    1,
    0,
    0,
    0,
    orientation,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
  ]);
  const length = Buffer.alloc(2);
  length.writeUInt16BE(exif.length + 2);
  return Buffer.concat([
    cleanJpeg.subarray(0, 2),
    Buffer.from([0xff, 0xe1]),
    length,
    exif,
    cleanJpeg.subarray(2),
  ]);
}

async function generateAndVerify(page, mapKey: 'vfr' | 'p250', browserName: string): Promise<void> {
  await page.goto('/');
  if (mapKey !== 'vfr') await page.locator(`[data-map-key="${mapKey}"]`).click();
  const started = Date.now();
  await page.locator('#generate').click();
  await expect(page.locator('#status')).toHaveAttribute('aria-busy', 'false', { timeout: 60_000 });
  await expect(page.locator('#status')).toContainText('Generated');
  expect(Date.now() - started).toBeLessThan(60_000);
  await expect(page.locator('[data-artifact="map"]')).toHaveAttribute('data-state', 'ok');
  await expect(page.locator('[data-artifact="overlay"]')).toHaveAttribute('data-state', 'ok');
  await expect(page.locator('[data-artifact="crop"]')).toHaveAttribute('data-state', 'ok');
  await expect(page.locator('[data-artifact="preview"]')).toHaveAttribute('data-state', 'ok');
  await expect(page.locator('#croppedPreviewImage')).toHaveJSProperty('complete', true);
  expect(
    await page
      .locator('#croppedPreviewImage')
      .evaluate((image: HTMLImageElement) => Math.max(image.naturalWidth, image.naturalHeight))
  ).toBeGreaterThan(1800);
  await expectPdfBlob(page, '#downloadPdf');
  await expectPdfBlob(page, '#downloadOverlay');
  await expectPdfBlob(page, '#downloadCropped');
  await expectStructuredDownloads(page);
  if (browserName === 'chromium') {
    const heapBytes = await page.evaluate(() =>
      'memory' in performance
        ? (performance as Performance & { memory: { usedJSHeapSize: number } }).memory.usedJSHeapSize
        : null
    );
    expect(heapBytes).not.toBeNull();
    expect(heapBytes as number).toBeLessThan(512 * 1024 * 1024);
  }
}

test('VFR generation completes with bounded preview and valid PDFs', async ({ page, browserName }) => {
  await generateAndVerify(page, 'vfr', browserName);
});

test('P250 generation completes with bounded preview and valid PDFs', async ({ page, browserName }) => {
  await generateAndVerify(page, 'p250', browserName);
});

test('corrupt JPEG remains a manual-review item instead of crashing generation', async ({ page }) => {
  await page.goto('/');
  await page.locator('#photoFiles').setInputFiles({
    name: 'broken.jpg',
    mimeType: 'image/jpeg',
    buffer: Buffer.from('not a jpeg'),
  });
  await expect(page.locator('#photoProgressText')).toContainText('1 of 1 photos imported');
  await expect(page.locator('.photo-card')).toContainText('Metadata error');
  await page.locator('#generate').click();
  await expect(page.locator('#status')).toHaveAttribute('aria-busy', 'false', { timeout: 60_000 });
  await expect(page.locator('#status')).toContainText('Generated');
  await expect(page.locator('.photo-card .photo-status')).toContainText('Manual review');
  await expect(page.locator('[data-artifact="map"]')).toHaveAttribute('data-state', 'ok');
  await expect(page.locator('[data-artifact="handout"]')).toHaveAttribute('data-state', 'failed');
  await expectPdfBlob(page, '#downloadPdf');
});

test('OSM requires explicit third-party tile consent', async ({ page }) => {
  await page.goto('/');
  await page.locator('[data-map-key="osm"]').click();
  await expect(page.locator('[data-map-key="osm"]')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#osmThirdPartyConsent')).not.toBeChecked();
  await page.locator('#generate').click();
  await expect(page.locator('#status')).toContainText('Consent to third-party OpenStreetMap');
});

test('preview failure preserves successful map downloads', async ({ page }) => {
  await page.route('**/maps/previews/**/*.webp', (route) => route.abort());
  await page.goto('/');
  await page.locator('#generate').click();
  await expect(page.locator('#status')).toHaveAttribute('aria-busy', 'false', { timeout: 60_000 });
  await expect(page.locator('[data-artifact="map"]')).toHaveAttribute('data-state', 'ok');
  await expect(page.locator('[data-artifact="preview"]')).toHaveAttribute('data-state', 'failed');
  await expectPdfBlob(page, '#downloadPdf');
});

test('generation is rejected while an import transaction is active', async ({ page }) => {
  await page.route('**/examples/photo-manifest.json', async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    await route.continue();
  });
  await page.goto('/');
  await page.locator('#loadPhotoExample').click({ noWaitAfter: true });
  await expect(page.locator('#photoFiles')).toBeDisabled();
  await expect(page.locator('#loadPhotoExample')).toBeDisabled();
  await page.locator('#generate').click();
  await expect(page.locator('#status')).toContainText('Wait for the current photo import');
});

test('a completed route can be generated repeatedly', async ({ page }) => {
  await page.goto('/');
  for (let run = 0; run < 2; run += 1) {
    await page.locator('#generate').click();
    await expect(page.locator('#status')).toHaveAttribute('aria-busy', 'false', { timeout: 60_000 });
    await expect(page.locator('[data-artifact="map"]')).toHaveAttribute('data-state', 'ok');
    await expect(page.locator('#downloadPdf')).toBeVisible();
  }
});

test('generation can be cancelled without starting a second operation', async ({ page }) => {
  await page.goto('/');
  await page.locator('[data-map-key="p250"]').click();
  await page.locator('#generate').click();
  await expect(page.locator('#cancelGeneration')).toBeVisible();
  await page.locator('#cancelGeneration').click();
  await expect(page.locator('#status')).toHaveAttribute('aria-busy', 'false');
  await expect(page.locator('#status')).toContainText('cancelled');
  await expect(page.locator('#generate')).toBeEnabled();
});

test('initial UI has no serious accessibility violations at mobile width', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 800 });
  await page.goto('/');
  const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa']).analyze();
  expect(
    results.violations.filter((violation) => ['serious', 'critical'].includes(violation.impact ?? ''))
  ).toEqual([]);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(1);
});

test('tablet layout does not overflow horizontally', async ({ page }) => {
  await page.setViewportSize({ width: 768, height: 1024 });
  await page.goto('/');
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(1);
  await expect(page.locator('#generate')).toBeVisible();
  await expect(page.locator('#photoDropzone')).toBeVisible();
});

test('oversized photo is rejected before decoding', async ({ page, browserName }) => {
  test.skip(browserName !== 'chromium', 'The shared import-limit path only needs one large allocation.');
  await page.goto('/');
  await page.locator('#photoFiles').setInputFiles({
    name: 'oversized.jpg',
    mimeType: 'image/jpeg',
    buffer: Buffer.alloc(25 * 1024 * 1024 + 1),
  });
  await expect(page.locator('#photoProgressText')).toContainText('exceeds the 25 MB limit');
  await expect(page.locator('#status')).toContainText('1 failed');
  await expect(page.locator('.photo-card')).toHaveCount(0);
});

test('all eight EXIF orientations produce correctly shaped thumbnails', async ({ page }) => {
  await page.goto('/');
  const base64 = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 40;
    canvas.height = 20;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Canvas unavailable');
    context.fillStyle = '#f00';
    context.fillRect(0, 0, 20, 20);
    context.fillStyle = '#00f';
    context.fillRect(20, 0, 20, 20);
    return canvas.toDataURL('image/jpeg').split(',')[1];
  });
  const base = Buffer.from(base64, 'base64');
  await page.locator('#photoFiles').setInputFiles(
    Array.from({ length: 8 }, (_, index) => ({
      name: `orientation-${index + 1}.jpg`,
      mimeType: 'image/jpeg',
      buffer: withExifOrientation(base, index + 1),
    }))
  );
  await expect(page.locator('.photo-card')).toHaveCount(8);
  const dimensions = await page
    .locator('.photo-thumbnail')
    .evaluateAll((images: HTMLImageElement[]) =>
      images.map((image) => [image.naturalWidth, image.naturalHeight])
    );
  dimensions.forEach(([width, height], index) => {
    if (index < 4) expect(width).toBeGreaterThan(height);
    else expect(height).toBeGreaterThan(width);
  });
});

test('full historical example generates all artifacts', async ({ page, browserName }) => {
  test.skip(browserName !== 'chromium', 'The full 49 MB fixture runs once; map flows are cross-browser.');
  test.setTimeout(180_000);
  await page.goto('/');
  await page.locator('#loadPhotoExample').click();
  await expect(page.locator('.photo-card')).toHaveCount(29, { timeout: 120_000 });
  const enrouteCard = page
    .locator('.photo-card')
    .filter({
      has: page.locator('[data-field="classification"] option:checked', { hasText: 'En-route photo' }),
    })
    .first();
  await expect(enrouteCard).toBeVisible();
  const enrouteId = await enrouteCard.getAttribute('data-photo-id');
  expect(enrouteId).not.toBeNull();
  const liveEnrouteCard = () => page.locator(`.photo-card[data-photo-id="${enrouteId}"]`);
  const exceptionButton = liveEnrouteCard().locator('.photo-exception-button:visible');
  await expect(exceptionButton).toBeVisible();
  await exceptionButton.click();
  await expect(page.locator('.photo-status[data-tone="accepted"]')).toHaveCount(1);
  await page.locator('[data-map-key="p250"]').click();
  await page.locator('#generate').click();
  await expect(page.locator('#status')).toHaveAttribute('aria-busy', 'false', { timeout: 120_000 });
  await expect(page.locator('[data-artifact="map"]')).toHaveAttribute('data-state', 'ok');
  await expect(page.locator('[data-artifact="preview"]')).toHaveAttribute('data-state', 'ok');
  await expect(page.locator('[data-artifact="handout"]')).toHaveAttribute('data-state', 'ok');
  const summary = JSON.parse((await downloadBytes(page, '#downloadSummary')).toString());
  await expectSummarySchemas(summary);
  expect(summary.photos.counts.acceptedExceptions).toBe(1);
  expect(
    summary.photos.photos.some(
      (photo: { exceptionAccepted: boolean; findings: Array<{ severity: string }> }) =>
        photo.exceptionAccepted && photo.findings.some((finding) => finding.severity === 'violation')
    )
  ).toBe(true);
  const judgeMap = await downloadBytes(page, '#downloadPdf');
  const competitorMap = await downloadBytes(page, '#downloadOverlay');
  const emptyMap = await downloadBytes(page, '#downloadCropped');
  for (const pdf of [judgeMap, competitorMap, emptyMap]) {
    expect(pdf.subarray(0, 5).toString()).toBe('%PDF-');
  }
  expect(judgeMap.equals(competitorMap)).toBe(false);
  expect(competitorMap.equals(emptyMap)).toBe(false);
  await expectPdfBlob(page, '#downloadPhotoHandout');
});
