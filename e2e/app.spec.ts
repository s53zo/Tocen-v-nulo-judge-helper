import { readFile } from 'node:fs/promises';
import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';
import Ajv2020 from 'ajv/dist/2020.js';
import { strFromU8, unzipSync } from 'fflate';
import { PDFDocument, PrintScaling } from 'pdf-lib';

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

async function expectTrueScaleMapPdf(page, selector: string): Promise<void> {
  const bytes = await downloadBytes(page, selector);
  expect(bytes.subarray(0, 5).toString()).toBe('%PDF-');
  const document = await PDFDocument.load(bytes);
  const pages = document.getPages();
  expect(pages).toHaveLength(1);
  const { width, height } = pages[0].getSize();
  const millimetersToPoints = 72 / 25.4;
  const supportedSizes = [
    [210 * millimetersToPoints, 297 * millimetersToPoints],
    [297 * millimetersToPoints, 210 * millimetersToPoints],
    [297 * millimetersToPoints, 420 * millimetersToPoints],
    [420 * millimetersToPoints, 297 * millimetersToPoints],
  ];
  expect(
    supportedSizes.some(
      ([expectedWidth, expectedHeight]) =>
        Math.abs(width - expectedWidth) < 0.1 && Math.abs(height - expectedHeight) < 0.1
    )
  ).toBe(true);
  const viewerPreferences = document.catalog.getViewerPreferences();
  expect(viewerPreferences?.getPrintScaling()).toBe(PrintScaling.None);
  expect(viewerPreferences?.getPickTrayByPDFSize()).toBe(true);
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

async function expectStructuredDownloads(page, expectPhotoCsv = true): Promise<void> {
  const summary = JSON.parse((await downloadBytes(page, '#downloadSummary')).toString());
  await expectSummarySchemas(summary);
  if (expectPhotoCsv) {
    const header = (await downloadBytes(page, '#downloadPhotoAnalysis')).toString().split(/\r?\n/, 1)[0];
    expect(header).toContain('identifier,file_name,classification');
    expect(header).toContain('task_latitude');
  }
}

async function goToStage(page, stage: 1 | 2 | 3 | 4): Promise<void> {
  await page.locator(`[data-workflow-step="${stage}"]`).click();
  await expect(page.locator(`[data-workflow-stage="${stage}"]`)).toBeVisible();
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
  await goToStage(page, 3);
  const started = Date.now();
  await page.locator('#generate').click();
  await expect(page.locator('#status')).toHaveAttribute('aria-busy', 'false', { timeout: 60_000 });
  await expect(page.locator('#status')).toContainText('Generated');
  expect(Date.now() - started).toBeLessThan(60_000);
  await expect(page.locator('#workflowStage4')).toBeVisible();
  await expect(page.locator('#outputs').getByRole('heading', { name: 'Judge' })).toBeVisible();
  await expect(page.locator('#outputs').getByRole('heading', { name: 'Competitor' })).toBeVisible();
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
  await expectTrueScaleMapPdf(page, '#downloadPdf');
  await expectTrueScaleMapPdf(page, '#downloadOverlay');
  await expectTrueScaleMapPdf(page, '#downloadCropped');
  await expectStructuredDownloads(page, false);
  await expect(page.locator('#downloadPhotoAnalysis')).toBeHidden();
  await expect(page.locator('[data-artifact="handout"]')).toContainText('omitted');
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

test('guided workflow shows one stage, keeps state, and puts control photos first', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#saveProject')).toHaveText('Save');
  await expect(page.locator('#loadProject')).toHaveText('Load');
  const generateStepBox = await page.locator('[data-workflow-step="4"]').boundingBox();
  const saveBox = await page.locator('#saveProject').boundingBox();
  expect(generateStepBox).not.toBeNull();
  expect(saveBox).not.toBeNull();
  expect((saveBox?.x ?? 0) > (generateStepBox?.x ?? 0)).toBe(true);
  await expect(page.locator('.workflow-stage:visible')).toHaveCount(1);
  await expect(page.locator('#workflowStage1')).toBeVisible();
  await page.locator('#speed').selectOption('80kt');
  await page.locator('#waypoints').fill('SP,46.6,16.0\nTP1,46.55,16.1\nFP,46.5,16.2');
  await page.locator('#routeContinue').click();
  await expect(page.locator('.workflow-stage:visible')).toHaveCount(1);
  await expect(page.locator('#workflowStage2')).toBeVisible();
  await expect(page.locator('[data-workflow-step="1"]')).toHaveAttribute('data-state', 'complete');
  const controlComesFirst = await page.evaluate(() => {
    const control = document.querySelector('#controlPreparationTitle');
    const competition = document.querySelector('#competitionPreparationTitle');
    return Boolean(control?.compareDocumentPosition(competition) & Node.DOCUMENT_POSITION_FOLLOWING);
  });
  expect(controlComesFirst).toBe(true);
  await goToStage(page, 1);
  await expect(page.locator('#speed')).toHaveValue('80kt');
  await expect(page.locator('#waypoints')).toHaveValue(/TP1,46\.55,16\.1/);
});

test('invalid route blocks forward progress and identifies the route field', async ({ page }) => {
  await page.goto('/');
  await page.locator('#waypoints').fill('SP,46.6,16.0');
  await page.locator('#routeContinue').click();
  await expect(page.locator('#workflowStage1')).toBeVisible();
  await expect(page.locator('#waypoints')).toHaveAttribute('aria-invalid', 'true');
  await expect(page.locator('#routeSetupError')).toContainText('At least two waypoints');
});

test('corrupt JPEG remains an actionable item instead of crashing generation', async ({ page }) => {
  await page.goto('/');
  await goToStage(page, 2);
  await page.locator('#photoFiles').setInputFiles({
    name: 'broken.jpg',
    mimeType: 'image/jpeg',
    buffer: Buffer.from('not a jpeg'),
  });
  await expect(page.locator('#photoProgressText')).toContainText('1 of 1 photos imported');
  await goToStage(page, 3);
  await expect(page.locator('.photo-card')).toContainText('Metadata error');
  await page.locator('#generate').click();
  await expect(page.locator('#status')).toHaveAttribute('aria-busy', 'false', { timeout: 60_000 });
  await expect(page.locator('#status')).toContainText('Generated');
  await expect(page.locator('.photo-card .photo-status')).toContainText('Action required');
  await expect(page.locator('[data-artifact="map"]')).toHaveAttribute('data-state', 'ok');
  await expect(page.locator('[data-artifact="handout"]')).toHaveAttribute('data-state', 'failed');
  await expectPdfBlob(page, '#downloadPdf');
});

test('removing a competition photo closes identifier gaps', async ({ page }) => {
  const jpegs = await Promise.all([
    readFile(new URL('../examples/photos/IMG__160111_00_092 TP2.jpg', import.meta.url)),
    readFile(new URL('../examples/photos/IMG__164053_00_298.jpg', import.meta.url)),
    readFile(new URL('../examples/photos/IMG__162634_00_224 TP5.jpg', import.meta.url)),
  ]);
  await page.goto('/');
  await goToStage(page, 2);
  await page.locator('#photoFiles').setInputFiles(
    jpegs.map((jpeg, index) => ({
      name: `IMG_remove_${index + 1}.jpg`,
      mimeType: 'image/jpeg',
      buffer: jpeg,
    }))
  );
  await expect(page.locator('#photoProgressText')).toContainText('3 of 3 photos imported');
  await goToStage(page, 3);
  await page.locator('.photo-card').nth(1).locator('[data-action="remove"]').click();

  const identifiers = await page
    .locator('input[data-field="identifier"]')
    .evaluateAll((inputs) => inputs.map((input) => (input as HTMLInputElement).value));
  expect(identifiers).toEqual(['A', 'B']);
  await expect(page.locator('#status')).toContainText('without letter gaps');
});

test('a saved project restores route settings and embedded photos', async ({ page }) => {
  const jpeg = await readFile(new URL('../examples/photos/IMG__164053_00_298.jpg', import.meta.url));
  await page.goto('/');
  await page.locator('.custom-speed-panel summary').click();
  await page.locator('[data-custom-speed-value="1"]').fill('154');
  await page.locator('[data-custom-speed-unit="1"]').selectOption('kmh');
  await page.locator('#speed').selectOption('custom-1');
  await page.locator('[data-map-key="p250"]').click();
  await goToStage(page, 2);
  await page.locator('#photoFiles').setInputFiles({
    name: 'IMG_saved_project.jpg',
    mimeType: 'image/jpeg',
    buffer: jpeg,
  });
  await expect(page.locator('#photoProgressText')).toContainText('1 of 1 photos imported');
  await goToStage(page, 3);
  await page.locator('.photo-editor-details summary').click();
  const identifier = page.locator('input[data-field="identifier"]');
  await identifier.fill('Z');
  await identifier.press('Tab');
  await page.locator('#workflowStage3 > .panel-content > .advanced-panel > summary').click();
  await page.locator('#photoCropBounds').uncheck();

  const [download] = await Promise.all([page.waitForEvent('download'), page.locator('#saveProject').click()]);
  const projectPath = await download.path();
  expect(projectPath).not.toBeNull();
  expect(download.suggestedFilename()).toMatch(/^route_project_v2_\d{4}-\d{2}-\d{2}\.tvn-project$/);
  const savedProject = JSON.parse(await readFile(projectPath as string, 'utf8'));
  expect(savedProject.format).toBe('tocen-v-nulo-route-project');
  expect(savedProject.schemaVersion).toBe(2);
  expect(savedProject.settings.customSpeeds[0]).toEqual({ value: '154', unit: 'kmh' });

  await page.locator('.photo-card [data-action="remove"]').click();
  await goToStage(page, 1);
  await page.locator('#speed').selectOption('55kt');
  await page.locator('#projectFile').setInputFiles(projectPath as string);
  await expect(page.locator('#status')).toContainText('Loaded project');
  await expect(page.locator('#speed')).toHaveValue('custom-1');
  await expect(page.locator('[data-custom-speed-value="1"]')).toHaveValue('154');
  await expect(page.locator('[data-custom-speed-unit="1"]')).toHaveValue('kmh');
  await expect(page.locator('[data-map-key="p250"]')).toHaveAttribute('aria-pressed', 'true');
  await goToStage(page, 3);
  await expect(page.locator('.photo-card')).toHaveCount(1);
  await expect(page.locator('input[data-field="identifier"]')).toHaveValue('Z');
  await expect(page.locator('#photoCropBounds')).not.toBeChecked();
});

test('speed-edition ZIP contains default, custom, and shared competition files', async ({
  page,
  browserName,
}) => {
  test.skip(browserName !== 'chromium', 'The full archive structure only needs one browser engine.');
  test.setTimeout(120_000);
  const document = await PDFDocument.create();
  const pageTemplate = document.addPage([2862, 1985]);
  pageTemplate.drawLine({ start: { x: 0, y: 0 }, end: { x: 1, y: 1 } });
  const blankMap = Buffer.from(await document.save());
  await page.route('**/maps/00_VFRspredaj_25_SC_WEB_flat.pdf', (route) =>
    route.fulfill({ status: 200, contentType: 'application/pdf', body: blankMap })
  );
  await page.goto('/');
  await page.locator('.custom-speed-panel summary').click();
  await page.locator('[data-custom-speed-value="1"]').fill('140');
  await page.locator('[data-custom-speed-unit="1"]').selectOption('kmh');
  const jpeg = await readFile(new URL('../examples/photos/IMG__164053_00_298.jpg', import.meta.url));
  await goToStage(page, 2);
  await page.locator('#photoFiles').setInputFiles({
    name: 'IMG_speed_archive.jpg',
    mimeType: 'image/jpeg',
    buffer: jpeg,
  });
  await expect(page.locator('#photoProgressText')).toContainText('1 of 1 photos imported');
  await goToStage(page, 3);
  await page.locator('#generate').click();
  await expect(page.locator('#status')).toHaveAttribute('aria-busy', 'false', { timeout: 30_000 });
  await expect(page.locator('#status')).toContainText('Generated');
  await page.locator('#generateSpeedSet').click();
  await expect
    .poll(
      async () => {
        const state = await page.locator('#speedSetProgress').getAttribute('data-state');
        if (state === 'error')
          throw new Error((await page.locator('#status').textContent()) ?? 'Unknown error');
        return state;
      },
      { timeout: 120_000 }
    )
    .toBe('complete');
  await expect(page.locator('#speedSetProgressCount')).toHaveText('12 of 12');
  const archive = unzipSync(await downloadBytes(page, '#downloadSpeedSet'));
  const speedPdfNames = Object.keys(archive).filter(
    (name) => !name.startsWith('shared/') && name.endsWith('.pdf')
  );
  expect(speedPdfNames).toHaveLength(24);
  for (const speed of [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]) {
    const judge = archive[`${speed}kt/judge_solution_map_${speed}kt.pdf`];
    const competitor = archive[`${speed}kt/competitor_route_map_${speed}kt.pdf`];
    expect(Buffer.from(judge.subarray(0, 5)).toString()).toBe('%PDF-');
    expect(Buffer.from(competitor.subarray(0, 5)).toString()).toBe('%PDF-');
  }
  expect(archive['custom1_140kmh/judge_solution_map_custom1_140kmh.pdf']).toBeDefined();
  expect(archive['custom1_140kmh/competitor_route_map_custom1_140kmh.pdf']).toBeDefined();

  for (const name of [
    'shared/empty_map.pdf',
    'shared/judge_photo_handout.pdf',
    'shared/competitor_photo_handout.pdf',
  ]) {
    expect(Buffer.from(archive[name].subarray(0, 5)).toString()).toBe('%PDF-');
  }
  expect(strFromU8(archive['shared/photo_analysis.csv'])).toContain('file_name');
  expect(strFromU8(archive['shared/photo_overlay_key.csv'])).toContain('file_name');
  expect(strFromU8(archive['README.txt'])).toContain('shared folder');

  const projectName = Object.keys(archive).find((name) =>
    /^shared\/route_project_v2_\d{4}-\d{2}-\d{2}\.tvn-project$/.test(name)
  );
  expect(projectName).toBeDefined();
  const savedProject = JSON.parse(strFromU8(archive[projectName as string]));
  expect(savedProject.schemaVersion).toBe(2);
  expect(savedProject.photos).toHaveLength(1);
  expect(savedProject.settings.customSpeeds[0]).toEqual({ value: '140', unit: 'kmh' });
});

test('OSM requires explicit third-party tile consent', async ({ page }) => {
  await page.goto('/');
  await page.locator('[data-map-key="osm"]').click();
  await expect(page.locator('[data-map-key="osm"]')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#osmThirdPartyConsent')).not.toBeChecked();
  await goToStage(page, 3);
  await page.locator('#generate').click();
  await expect(page.locator('#status')).toContainText('Consent to third-party OpenStreetMap');
});

test('OSM-selected DOF025 route photo is reviewed, fetched, and retained as a PHOTO_ target', async ({
  page,
}) => {
  const jpeg = await readFile(new URL('../examples/photos/IMG__160111_00_092 TP2.jpg', import.meta.url));
  const secondJpeg = await readFile(new URL('../examples/photos/IMG__164053_00_298.jpg', import.meta.url));
  let requestedUrl = '';
  let orthophotoRequestCount = 0;
  const overpassQueries: string[] = [];
  await page.route('https://maps.mail.ru/osm/tools/overpass/**', async (route) => {
    overpassQueries.push(route.request().postData() ?? '');
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        elements: [
          {
            type: 'way',
            id: 123,
            center: { lat: 46.5508, lon: 16.1676 },
            tags: { bridge: 'yes', highway: 'tertiary', name: 'Test bridge' },
          },
          {
            type: 'way',
            id: 124,
            center: { lat: 46.5608, lon: 16.1701 },
            tags: { bridge: 'yes', highway: 'tertiary', name: 'Second test bridge' },
          },
        ],
      }),
    });
  });
  await page.route('https://ipi.eprostor.gov.si/**', async (route) => {
    requestedUrl = route.request().url();
    const body = orthophotoRequestCount % 2 === 0 ? jpeg : secondJpeg;
    orthophotoRequestCount += 1;
    await route.fulfill({ status: 200, contentType: 'image/jpeg', body });
  });
  await page.goto('/');
  await goToStage(page, 2);
  await page.locator('[data-photo-source-tab="osm"]').click();
  await page.locator('#orthophotoRandomCount').fill('1');
  await page.locator('#addRandomOrthophotos').click();
  await expect(page.locator('#osmDiscoveryProgress')).toBeVisible();
  await expect(page.locator('#osmCandidateReview')).toBeVisible();
  await expect(page.locator('#osmDiscoveryProgressPercent')).toHaveText('100%');
  await expect(page.locator('.osm-leg-progress[data-state="done"]')).toHaveCount(7);
  await expect(page.getByText('Test bridge', { exact: true })).toBeVisible();
  const initialQueryCount = overpassQueries.length;
  expect(initialQueryCount).toBeGreaterThan(1);
  const firstSelectionMix = await page.locator('#osmCandidateSummary').textContent();
  await page.locator('#refreshOsmCandidates').click();
  await expect(page.locator('#osmCandidateSummary')).not.toHaveText(firstSelectionMix ?? '');
  await expect(page.locator('#osmCandidateSummary')).toContainText('Selection mix');
  const replacedSelectionMix = await page.locator('#osmCandidateSummary').textContent();
  await page.locator('#addRandomOrthophotos').click();
  await expect(page.locator('#osmCandidateSummary')).not.toHaveText(replacedSelectionMix ?? '');
  expect(overpassQueries).toHaveLength(initialQueryCount);
  await page.locator('#orthophotoRandomCount').fill('2');
  await page.locator('#addRandomOrthophotos').click();
  await expect(page.locator('#osmCandidateReview')).toBeVisible();
  expect(overpassQueries.length).toBeGreaterThan(1);
  await page.locator('#orthophotoRandomCount').fill('1');
  await expect(page.locator('.osm-candidate')).toHaveCount(2);
  await expect(page.locator('.osm-candidate input:checked')).toHaveCount(1);
  await page.locator('#selectAllOsmCandidates').click();
  await expect(page.locator('.osm-candidate input:checked')).toHaveCount(2);
  await expect(page.locator('#importOsmCandidates')).toBeEnabled();
  await expect(page.locator('#osmCandidateSelectionStatus')).toContainText('2 selected');
  await page.locator('#importOsmCandidates').click();

  await expect(page.locator('#photoProgressText')).toContainText('2 of 2 DOF025 crops imported');
  await goToStage(page, 3);
  await expect(page.locator('.photo-card')).toHaveCount(2);
  await expect(page.locator('.photo-card').first()).toContainText('GURS DOF025 crop');
  await expect(page.locator('.photo-card').filter({ hasText: 'OSM target: Test bridge' })).toHaveCount(1);
  await expect(page.locator('.photo-card').first()).toContainText('selection mix');
  await expect(page.locator('.photo-card .photo-status').first()).toContainText('OK for automated checks');
  await expect(page.locator('[data-action="verify-generated-photo"]')).toHaveCount(0);
  await expect(page.locator('#waypoints')).toHaveValue(/PHOTO_OSM_BRIDGE_01,/);
  expect(overpassQueries.some((query) => query.includes('bridge'))).toBe(true);
  expect(requestedUrl).toContain('LAYERS=SI.GURS.ZPDZ%3ADOF025');
  expect(requestedUrl).toContain('WIDTH=1600');
});

test('control-photo discovery fixes SP/FP to true and lets each TP choose a false object', async ({
  page,
}) => {
  const importJpegs = await Promise.all([
    readFile(new URL('../examples/photos/IMG__160111_00_092 TP2.jpg', import.meta.url)),
    readFile(new URL('../examples/photos/IMG__164053_00_298.jpg', import.meta.url)),
    readFile(new URL('../examples/photos/IMG__162634_00_224 TP5.jpg', import.meta.url)),
  ]);
  const controlData = {
    elements: [
      {
        type: 'way',
        id: 100,
        center: { lat: 46.6002, lon: 16.001 },
        tags: { bridge: 'yes', name: 'SP bridge' },
      },
      {
        type: 'way',
        id: 101,
        center: { lat: 46.6002, lon: 16.101 },
        tags: { bridge: 'yes', name: 'TP bridge' },
      },
      {
        type: 'way',
        id: 102,
        center: { lat: 46.6002, lon: 16.199 },
        tags: { bridge: 'yes', name: 'FP bridge' },
      },
      {
        type: 'way',
        id: 103,
        center: { lat: 46.6, lon: 16.13 },
        tags: { bridge: 'yes', name: 'False bridge' },
      },
    ],
  };
  await page.route('https://maps.mail.ru/osm/tools/overpass/**', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(controlData) });
  });
  let importIndex = 0;
  const importAttempts = new Map<string, number>();
  const importUrlOrder: string[] = [];
  await page.route('https://ipi.eprostor.gov.si/**', async (route) => {
    const url = route.request().url();
    const width = new URL(url).searchParams.get('WIDTH');
    if (width === '1600') {
      if (!importUrlOrder.includes(url)) importUrlOrder.push(url);
      const attempts = importAttempts.get(url) ?? 0;
      importAttempts.set(url, attempts + 1);
      const transientFailures = importUrlOrder.indexOf(url) === 1 ? 2 : 1;
      if (attempts < transientFailures) {
        await route.fulfill({ status: 404, contentType: 'text/plain', body: 'Transient failure' });
        return;
      }
    }
    const body = width === '1600' ? importJpegs[importIndex++] : importJpegs[0];
    await route.fulfill({ status: 200, contentType: 'image/jpeg', body });
  });
  await page.goto('/');
  await page.locator('#waypoints').fill('SP,46.6,16.0\nTP1,46.6,16.1\nFP,46.6,16.2');
  await goToStage(page, 2);
  await page.locator('#findControlPhotoOptions').click();
  await expect(page.locator('#controlPhotoReview')).toBeVisible();
  await expect(page.locator('#controlPhotoProgress')).toBeVisible();
  await expect(page.locator('#controlPhotoProgressCount')).toHaveText('3 of 3 controls processed');
  await expect(page.locator('#controlPhotoProgressPhase')).toContainText('3 photo proposals ready');
  await expect(page.locator('#controlPhotoProgressBar')).toHaveJSProperty('value', 4);
  await expect(page.locator('#controlPhotoProgressBar')).toHaveJSProperty('max', 4);
  await expect(page.locator('.control-photo-row')).toHaveCount(3);
  await expect(page.locator('.control-photo-waypoint')).toHaveText(['SP', 'TP1', 'FP']);
  await expect(page.locator('.control-photo-row').nth(0).locator('.control-photo-option')).toHaveCount(1);
  await expect(page.locator('.control-photo-row').nth(1).locator('.control-photo-option')).toHaveCount(2);
  await expect(page.locator('.control-photo-row').nth(2).locator('.control-photo-option')).toHaveCount(1);
  await page.locator('.control-photo-option img').evaluateAll((images) => {
    (window as typeof window & { controlPreviewNodes?: Element[] }).controlPreviewNodes = images;
  });
  await page.locator('input[data-control-waypoint="TP1"][value="false"]').check();
  const previewsWerePreserved = await page
    .locator('.control-photo-option img')
    .evaluateAll((images) =>
      images.every(
        (image, index) =>
          image ===
          (window as typeof window & { controlPreviewNodes?: Element[] }).controlPreviewNodes?.[index]
      )
    );
  expect(previewsWerePreserved).toBe(true);
  await expect(page.locator('#importControlPhotos')).toHaveText('Import chosen control photos');
  await page.locator('#importControlPhotos').click();
  await expect(page.locator('#photoProgressText')).toContainText('2 of 3 DOF025 crops imported');
  await expect(page.locator('#importControlPhotos')).toHaveText('Retry 1 failed control photo');
  await expect(page.locator('.control-photo-row input:disabled')).toHaveCount(2);
  await page.locator('#importControlPhotos').click();
  await expect(page.locator('#photoProgressText')).toContainText('1 of 1 DOF025 crops imported');
  await expect(page.locator('#controlPhotoReview')).toBeHidden();
  expect([...importAttempts.values()]).toEqual([2, 3, 2]);
  await goToStage(page, 3);
  await expect(page.locator('.photo-card')).toHaveCount(3);
  await expect(page.locator('.photo-card').nth(0)).toContainText('Correct control photo');
  await expect(page.locator('.photo-card').nth(0)).toContainText('46.600000_16.000000');
  await expect(page.locator('.photo-card').nth(1)).toContainText('False control photo');
  await expect(page.locator('.photo-card').nth(2)).toContainText('Correct control photo');
  await expect(page.locator('.photo-card').nth(2)).toContainText('46.600000_16.200000');
});

test('preview failure preserves successful map downloads', async ({ page }) => {
  await page.route('**/maps/previews/**/*.webp', (route) => route.abort());
  await page.goto('/');
  await goToStage(page, 3);
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
  await goToStage(page, 2);
  await page.locator('#loadPhotoExample').click({ noWaitAfter: true });
  await goToStage(page, 3);
  await expect(page.locator('#photoFiles')).toBeDisabled();
  await expect(page.locator('#loadPhotoExample')).toBeDisabled();
  await page.locator('#generate').click();
  await expect(page.locator('#status')).toContainText('Wait for the current photo import');
});

test('a completed route can be generated repeatedly', async ({ page }) => {
  await page.goto('/');
  for (let run = 0; run < 2; run += 1) {
    await goToStage(page, 3);
    await page.locator('#generate').click();
    await expect(page.locator('#status')).toHaveAttribute('aria-busy', 'false', { timeout: 60_000 });
    await expect(page.locator('[data-artifact="map"]')).toHaveAttribute('data-state', 'ok');
    await expect(page.locator('#downloadPdf')).toBeVisible();
  }
});

test('generation can be cancelled without starting a second operation', async ({ page }) => {
  await page.goto('/');
  await page.locator('[data-map-key="p250"]').click();
  await goToStage(page, 3);
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
  await expect(page.locator('#routeContinue')).toBeVisible();
  await goToStage(page, 2);
  await expect(page.locator('#photoDropzone')).toBeVisible();
});

test('every workflow stage remains usable at mobile width', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 800 });
  await page.goto('/');
  for (const stage of [1, 2, 3, 4] as const) {
    await goToStage(page, stage);
    await expect(page.locator(`#workflowStage${stage}`)).toBeVisible();
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow).toBeLessThanOrEqual(1);
  }
});

test('oversized photo is rejected before decoding', async ({ page, browserName }) => {
  test.skip(browserName !== 'chromium', 'The shared import-limit path only needs one large allocation.');
  await page.goto('/');
  await goToStage(page, 2);
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
  await goToStage(page, 2);
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
  await goToStage(page, 3);
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
  await page.locator('[data-map-key="p250"]').click();
  await goToStage(page, 2);
  await page.locator('#loadPhotoExample').click();
  await expect(page.locator('.photo-card')).toHaveCount(29, { timeout: 120_000 });
  await goToStage(page, 3);
  await expect(page.locator('.photo-card')).toHaveCount(29);
  const reviewLayout = await page.evaluate(() => ({
    stageWidth: document.querySelector('#workflowStage3')?.getBoundingClientRect().width ?? 0,
    viewportWidth: window.innerWidth,
    photoColumns: getComputedStyle(document.querySelector('.photo-list') as HTMLElement)
      .gridTemplateColumns.split(' ')
      .filter(Boolean).length,
  }));
  expect(reviewLayout.stageWidth).toBeGreaterThan(reviewLayout.viewportWidth * 0.7);
  expect(reviewLayout.photoColumns).toBeGreaterThanOrEqual(2);
  await expect(page.locator('.photo-metadata-details').first()).not.toHaveAttribute('open', '');
  const exceptionButton = page.locator('.photo-exception-button:visible').first();
  await expect(exceptionButton).toBeVisible();
  await exceptionButton.click();
  await expect(page.locator('.photo-status[data-tone="accepted"]')).toHaveCount(1);
  await page.locator('#generate').click();
  await expect(page.locator('#status')).toHaveAttribute('aria-busy', 'false', { timeout: 120_000 });
  await expect(page.locator('[data-artifact="map"]')).toHaveAttribute('data-state', 'ok');
  await expect(page.locator('[data-artifact="preview"]')).toHaveAttribute('data-state', 'ok');
  await expect(page.locator('[data-artifact="handout"]')).toHaveAttribute('data-state', 'ok');
  await expect(page.locator('[data-artifact="competitorHandout"]')).toHaveAttribute('data-state', 'ok');
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
  await expectPdfBlob(page, '#downloadCompetitorPhotoHandout');
});
