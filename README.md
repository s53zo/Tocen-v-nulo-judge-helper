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
- Reviewed GURS DOF025 photo crops selected from identifiable OpenStreetMap features or reserved `PHOTO_name,latitude,longitude` rows
- Separate camera and task/object positions, conservative ambiguous-leg handling, and manual leg overrides
- Per-artifact progress, cancellation, and partial-success handling so an optional handout/preview failure does not discard maps
- `photo_analysis.csv`, `photo_overlay_key.csv`, photo-aware `route_summary.json`, an accepted-photo-only judge handout, and a minimal competitor photo handout
- An opt-in 29-photo historical TVN 2025 example recovered from the original workflow
- Wide review workspace with compact route controls, multi-column photo cards, a persistent generation bar, and responsive result panels

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

After every successful generation, the app evaluates the calculable requirements in `docs/Pravilnik Aerorally.pdf`: official 1:250,000 map scale (A1.4), permitted groundspeeds (A1.5), 70-120 NM route length (A2.1.1), minimum 5 NM legs, control-point limit, and SP/FP identifiers (A2.1.2). Photo checks cover false control-object separation (A2.4.2), the 100 m sign-task route-axis limit (A2.4.4), the 12 en-route-photo limit plus reliable 50-70 mm equivalent focal length, 500-1,000 ft AGL, a configured 500 m camera route-axis screening limit, and 45-degree true-camera-angle limits (A2.4.5), and the 15-task and 1 NM post-control restrictions (A2.4.6). The 500 m camera screening threshold allows for a camera 200 m AGL viewing at 45 degrees toward an object at the rule's 300 m route-axis boundary; it is an application policy rather than a literal replacement of the rule's object-distance wording.

Photo review separates primary rule issues, required setup actions, and technical audit details. Only primary issues and required actions affect the operational status and judge-photo acceptance. Missing heading/AGL metadata, focal-length discrepancies, and the generic content-review reminder remain available in the collapsed technical audit and exported audit data without flooding each photo's main issue list.

En-route photo letters are automatically mixed independently on either side of the selected handout split after import. Use **Remix en-route letters** after changing classifications, route positions, or the split. Generated handout sections are displayed alphabetically by identifier, while the randomized letter-to-location mapping prevents the page order from revealing the route sequence required to remain hidden by A2.4.5. The competitor handout contains only the accepted photo images, their letters, and `Before TP...` / `After TP...` section headings; judge-only filenames, classifications, statuses, source details, and compliance findings are omitted.

Every finding records the rule, measured value, permitted value, stable photo ID, and affected photo. Camera GPS is not substituted for task/object coordinates. Unreliable or unavailable task position, GPS, AGL, heading reference, focal length, or false-object coordinates produce `Manual review required`; the app does not guess or silently discard a photo. Magnetic or reference-less headings are not compared automatically with true route bearings. EXIF GPS altitude remains labelled MSL and is never treated as AGL.

A judge may use **Accept exception** on an against-rules photo. This is an explicit waiver for using that photo in the output, not a compliance result: the badge continues to say that the photo is against the rules, all findings remain visible, and the acceptance flag and timestamp are exported.

The judge solution map contains the route plus photos that have no automated violation or have an explicit accepted exception. The competitor map contains the route, hollow circular SP/TP/FP markers, labels, timing, and bearings but no photo answers. The empty map uses the identical crop without any generated markings. En-route photo answers follow the historical Python convention: a violet tick perpendicular to the assigned route leg with its alphabetical label; classification squares and diamonds are not used. Accepted SP/TP/FP photos retain a violet label outside the waypoint circle but do not draw an additional photo tick line.

The historical example intentionally demonstrates discrepancies: it contains 20 en-route photos (over the maximum of 12), several post-control tasks inside 1 NM, unreliable repeated EXIF GPS, no camera heading or AGL, and a 6 mm-equivalent lens. Its GPX-interpolated example coordinates are stored as explicit overrides so the original EXIF remains auditable.

The result is deliberately limited to checks supported by reliable inputs. Every competition photo retains an explicit judge-content review finding for requirements that metadata cannot prove. An “OK for automated checks” result is not approval, certification, or a replacement for a judge. Landing, control-point descriptions, timed-control designation, task correctness/visibility, presentation, GPS logging, chart approval, weather, airspace, and VFR requirements still require manual confirmation.

## Resource limits and artifact behavior

Photo import is limited to 60 files, 25 MB per file, 250 MB total source data, and 50 megapixels per image. Browser thumbnails are capped at 480 px and handout images at 1,600 px. High-resolution map tiles are individually capped at 3 MB; a preview loads at most 128 tiles/40 MB and renders at up to 4,096 px or 12 megapixels with a 30-second deadline. Only intersecting tiles are decoded in four-tile batches, and only one full PDF chart buffer is retained.

Map, overlay, crop, preview, CSV/JSON, and handout results have independent status indicators. Preview and handout failures preserve completed map downloads. Cancelling stops fetch/background stages at the next safe interruption point; long synchronous PDF operations may finish their current step before the browser can process cancellation.

### GURS DOF025 route photos

The photo section can request orthophoto crops from the official GURS DOF025 WMS. **Find identifiable OSM targets** searches the route corridor for bridges, level crossings, road junctions, churches, sports grounds, cemeteries, water, landmarks, infrastructure, and distinctive buildings. Forest, scrub, grass, meadow, and farmland are never candidate classes. Targets are scored and deduplicated, and every eligible discovered target is shown for review before imagery is downloaded. **Initial selection** controls only the randomized suggested set: you can select any number of targets. Count, spacing, leg-coverage, and handout-split problems are shown as warnings rather than blocking the import. The separate 60-photo application safety limit still applies. Use **Replace selection** to obtain another diverse suggestion. Clicking the discovery or import button authorizes its associated external request.

Clicking **Find identifiable OSM targets** sends one narrow rotated corridor polygon for each straight route leg to public Overpass instances. Legs run sequentially because live diagnostics showed that concurrent requests can queue and time out on an otherwise healthy public server. Every leg first receives one bounded request. Failed requests are queued while the remaining legs continue, then retried in a deferred pass using alternate public instances before returning to the preferred endpoint. Recovered legs are shown in the progress display; results from successful legs are never discarded. Per-leg road-junction and building queries run only when the stronger candidates cannot form a complete, correctly split and spaced proposal. The suggested route-photo set is filled round-robin across route legs and displayed chronologically from SP to FP. The UI provides explicit cancellation, while the bounded discovery window gracefully retains completed-leg results. Complete results are cached for ten minutes using the route, requested count and split profile; partial results are retained in the current tab and merged into a repeated discovery. Every discovery and **Replace suggestion** action receives a new random selection salt. If no alternative valid set exists, the UI says so rather than claiming that the photos changed. Selected targets retain their OSM name, type, score, selection salt, element identifier, and attribution in the photo record and exports.

**Find SP/TP/FP photo options** searches for the nearest identifiable OSM object within 3 km of every control. SP and FP are fixed to their true object. For each TP, the app also searches bounded sample areas up to 10 NM away for the same feature type and accepts a false option only when it is at least 1 NM from the true object. The organizer reviews DOF025 previews and chooses true or false for every TP before import. Control photos retain their linked waypoint and solution role, are kept separate from the 12 en-route letter photos, and appear as a separate section in judge and competitor handouts; the competitor handout does not reveal whether a TP photo is true or false.

Alternatively, add rows such as `PHOTO_CHURCH,46.6301,16.1792` to the WAYPOINTS field and choose **Load PHOTO_ coordinates**. `PHOTO_` rows are retained for reproducibility but excluded from route geometry.

The default coverage model is 100 m AGL, 60 mm 35 mm-equivalent focal length, 45° depression, and 16:9 framing, producing an approximately 102 × 69 m top-down crop. These values are adjustable, but impractically large footprints are rejected. The crop approximates ground coverage; it does not synthesize the oblique perspective of a real camera photo. Downloads, response bytes, JPEG dimensions, and decoding are bounded and cancellable. A basic image-detail guard rejects blank or nearly uniform WMS results. Generated crops are included under the normal compliance checks; remove any crop whose target is not clear or usable. Requests send only the reviewed coordinates to `ipi.eprostor.gov.si`, and generated records retain GURS and OSM attribution and model parameters in the UI, CSV, JSON, and judge handout.

## Browser support and privacy

Use a current Chrome, Edge, Firefox, or Safari release with Web Crypto, File/Blob, and Canvas support. Chromium and WebKit are exercised in CI; `HTMLImageElement` and UUID fallbacks cover browsers without `createImageBitmap` or `crypto.randomUUID`. Photo hashes, previews, EXIF metadata, route analysis, and generated downloads live only in memory in the current tab. Reloading clears imported user photos. Bundled PDF maps are the local-only mode. Clicking an external-data action authorizes its request: feature discovery sends the route corridor to public Overpass servers, while importing DOF025 imagery sends only the selected coordinates to GURS. Neither service receives photo files or EXIF payloads.

Bundled aeronautical charts display a validity warning. Confirm that a chart is current and approved for the event before operational use. Updating a chart requires updating its file, metadata, calibration, tests, and generated bundle together.

## Deployment

GitHub Pages is configured for `main` and `/`. After changes are merged, Pages serves the root `index.html` and committed `assets/`. CI verifies source, types, unit/schema tests, lint, production bundle parity, a 3 MB application asset budget, and Chromium/WebKit end-to-end flows using the real charts. JSON consumers should validate against `schemas/route-summary.schema.json` and `schemas/photo-summary.schema.json`.
