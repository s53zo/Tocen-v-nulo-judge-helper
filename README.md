# Precision Route Planner

A browser-only tool for building precision-flying routes, analysing route photos, checking calculable judging rules, and producing calibrated map overlays and photo handouts. Route, photo, EXIF, and PDF processing happens locally in the browser. OpenStreetMap is an explicit opt-in because third-party tile requests reveal the approximate displayed route area.

The production page is the single root [`index.html`](./index.html). GitHub Pages serves it directly from the `main` branch root; there is no second web application or Python pipeline.

## Features

- Decimal latitude/longitude and Slovenian D96/TM waypoint input
- Knots, mph, and km/h groundspeed support
- Route distances, normalized bearings, and waypoint timing
- Automated Slovenian rally-rule checks with a clear OK/against-rules result
- Nautical-mile totals and leg distances alongside metric values
- Configurable minute markers and overlay styling
- Calibrated PDF maps plus consent-gated OpenStreetMap
- Three matched true-scale map handouts: judge solutions, competitor route, and an empty base map
- Native-detail cropped previews assembled from calibrated 1,024 px WebP tiles, never by decoding the whole chart
- Unicode waypoint labels in generated PDFs
- Searchable location library
- Explicit chart-edition warnings
- Multi-JPEG drag-and-drop import with previews, progress, SHA-256 duplicate detection, removal, and ordering
- Local EXIF extraction with explicit provenance and editable GPS, time, heading, AGL altitude, and focal-length fields
- Route projection, along-track/lateral distance, leg assignment, camera-heading comparison, and post-control spacing
- En-route, correct/false control, sign-task, and reference classifications with waypoint linking
- Per-photo and route-wide `OK for automated checks`, `Against the rules`, or `Manual review required` findings
- Auditable per-photo judge exceptions that retain every original violation in the UI, CSV, JSON, and handout
- Separate camera and task/object positions, conservative ambiguous-leg handling, and manual leg overrides
- Per-artifact progress, cancellation, and partial-success handling so an optional handout/preview failure does not discard maps
- `photo_analysis.csv`, `photo_overlay_key.csv`, photo-aware `route_summary.json`, and an accepted-photo-only judge handout
- An opt-in 29-photo historical TVN 2025 example recovered from the original workflow

## Development

Requires Node.js 24 or newer.

```bash
npm ci
npm run check
npm run build
npx playwright install chromium webkit
npm run test:e2e
```

`npm run check` performs TypeScript checking, unit tests, schema checks, and linting. `npm run test:e2e` builds the current source and exercises the real bundled charts in Chromium and WebKit. `npm run build` creates the committed browser bundle in `assets/`. Commit bundle changes with their corresponding source changes because GitHub Pages currently publishes the repository root directly.

## Project structure

```text
index.html              Single production page
src/main.ts             Browser UI, map rendering, and PDF orchestration
src/domain.ts           Route, bearing, speed, and timing calculations
src/csv.ts              Standards-aware CSV parser
src/data-url.ts         CSP-safe decoder for bundled data URL assets
src/maps.ts             Map-preset validation
src/map-preview.ts      Bounded high-resolution tiled preview renderer
src/map-presets.json    Versioned map metadata and calibration
src/styles.css          Application styles
src/photo-metadata.ts   EXIF normalization and metadata provenance
src/photo-analysis.ts   Pure route/photo geographic calculations
src/photo-compliance.ts Photo-task judging-rule checks
src/photo-workflow.ts   Import, editing, ordering, and UI state
src/photo-output.ts     CSV and JSON schemas
src/photo-image.ts      EXIF orientation correction
src/photo-handout.ts    Browser-generated A4 handout
tests/                  Calculation and configuration regression tests
e2e/                    Chromium/WebKit end-to-end tests
schemas/                Versioned JSON export schemas
maps/                   Browser map assets
maps/previews/          Bounded calibrated WebP preview maps
assets/                 Generated production bundle
examples/photos/        Historical browser test/example photos
docs/                   Event and judging reference documents
locations.txt           Location library data
```

## Map safety and calibration

Map editions and calibration data live in `src/map-presets.json`. PDF presets must declare dimensions and either exactly three affine control points or a valid TFW transform. Routes that project outside the calibrated PDF are rejected instead of producing a misleading overlay.

## Rule compliance

After every successful generation, the app evaluates the calculable requirements in `docs/Pravilnik Aerorally.pdf`: official 1:250,000 map scale (A1.4), permitted groundspeeds (A1.5), 70-120 NM route length (A2.1.1), minimum 5 NM legs, control-point limit, and SP/FP identifiers (A2.1.2). Photo checks cover false control-object separation (A2.4.2), the 100 m sign-task route-axis limit (A2.4.4), the 12 en-route-photo limit plus reliable 50-70 mm equivalent focal length, 500-1,000 ft AGL, 300 m camera route-axis distance, and 45-degree true-camera-angle limits (A2.4.5), and the 15-task and 1 NM post-control restrictions (A2.4.6).

Every finding records the rule, measured value, permitted value, stable photo ID, and affected photo. Camera GPS is not substituted for task/object coordinates. Unreliable or unavailable task position, GPS, AGL, heading reference, focal length, or false-object coordinates produce `Manual review required`; the app does not guess or silently discard a photo. Magnetic or reference-less headings are not compared automatically with true route bearings. EXIF GPS altitude remains labelled MSL and is never treated as AGL.

A judge may use **Accept exception** on an against-rules photo. This is an explicit waiver for using that photo in the output, not a compliance result: the badge continues to say that the photo is against the rules, all findings remain visible, and the acceptance flag and timestamp are exported.

The judge solution map contains the route plus photos that have no automated violation or have an explicit accepted exception. The competitor map contains the route, hollow circular SP/TP/FP markers, labels, timing, and bearings but no photo answers. The empty map uses the identical crop without any generated markings. En-route photo answers follow the historical Python convention: a violet tick perpendicular to the assigned route leg with its alphabetical label; classification squares and diamonds are not used. Accepted SP/TP/FP photos retain a violet label outside the waypoint circle but do not draw an additional photo tick line.

The historical example intentionally demonstrates discrepancies: it contains 20 en-route photos (over the maximum of 12), several post-control tasks inside 1 NM, unreliable repeated EXIF GPS, no camera heading or AGL, and a 6 mm-equivalent lens. Its GPX-interpolated example coordinates are stored as explicit overrides so the original EXIF remains auditable.

The result is deliberately limited to checks supported by reliable inputs. Every competition photo retains an explicit judge-content review finding for requirements that metadata cannot prove. An “OK for automated checks” result is not approval, certification, or a replacement for a judge. Landing, control-point descriptions, timed-control designation, task correctness/visibility, presentation, GPS logging, chart approval, weather, airspace, and VFR requirements still require manual confirmation.

## Resource limits and artifact behavior

Photo import is limited to 60 files, 25 MB per file, 250 MB total source data, and 50 megapixels per image. Browser thumbnails are capped at 480 px and handout images at 1,600 px. High-resolution map tiles are individually capped at 3 MB; a preview loads at most 128 tiles/40 MB and renders at up to 4,096 px or 12 megapixels with a 30-second deadline. Only intersecting tiles are decoded in four-tile batches, and only one full PDF chart buffer is retained.

Map, overlay, crop, preview, CSV/JSON, and handout results have independent status indicators. Preview and handout failures preserve completed map downloads. Cancelling stops fetch/background stages at the next safe interruption point; long synchronous PDF operations may finish their current step before the browser can process cancellation.

## Browser support and privacy

Use a current Chrome, Edge, Firefox, or Safari release with Web Crypto, File/Blob, and Canvas support. Chromium and WebKit are exercised in CI; `HTMLImageElement` and UUID fallbacks cover browsers without `createImageBitmap` or `crypto.randomUUID`. Photo hashes, previews, EXIF metadata, route analysis, and generated downloads live only in memory in the current tab. Reloading clears imported user photos. Bundled PDF maps are the local-only mode. OpenStreetMap remains disabled until the user consents to public tile requests; it does not send photo files or EXIF payloads, and exact photo positions do not expand the requested tile viewport.

Bundled aeronautical charts display a validity warning. Confirm that a chart is current and approved for the event before operational use. Updating a chart requires updating its file, metadata, calibration, tests, and generated bundle together.

## Deployment

GitHub Pages is configured for `main` and `/`. After changes are merged, Pages serves the root `index.html` and committed `assets/`. CI verifies source, types, unit/schema tests, lint, production bundle parity, a 3 MB application asset budget, and Chromium/WebKit end-to-end flows using the real charts. JSON consumers should validate against `schemas/route-summary.schema.json` and `schemas/photo-summary.schema.json`.
