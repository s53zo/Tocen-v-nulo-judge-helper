# Precision Route Planner

A browser-only tool for building precision-flying routes, calculating waypoint schedules and minute markers, and producing calibrated map overlays as PDFs. All route and PDF processing happens locally in the browser.

The production page is the single root [`index.html`](./index.html). GitHub Pages serves it directly from the `main` branch root; there is no second web application or Python pipeline.

## Features

- Decimal latitude/longitude and Slovenian D96/TM waypoint input
- Knots, mph, and km/h groundspeed support
- Route distances, normalized bearings, and waypoint timing
- Configurable minute markers and overlay styling
- Calibrated PDF maps plus an OpenStreetMap preview
- Marked, overlay-only, and true-scale A4 PDF downloads
- Unicode waypoint labels in generated PDFs
- Searchable location library
- Explicit chart-edition warnings

## Development

Requires Node.js 24 or newer.

```bash
npm ci
npm run check
npm run build
```

`npm run check` performs TypeScript checking, unit tests, and linting. `npm run build` creates the committed browser bundle in `assets/`. Commit bundle changes with their corresponding source changes because GitHub Pages currently publishes the repository root directly.

## Project structure

```text
index.html              Single production page
src/main.ts             Browser UI, map rendering, and PDF orchestration
src/domain.ts           Route, bearing, speed, and timing calculations
src/csv.ts              Standards-aware CSV parser
src/maps.ts             Map-preset validation
src/map-presets.json    Versioned map metadata and calibration
src/styles.css          Application styles
tests/                  Calculation and configuration regression tests
maps/                   Browser map assets
assets/                 Generated production bundle
docs/                   Event and judging reference documents
locations.txt           Location library data
```

## Map safety and calibration

Map editions and calibration data live in `src/map-presets.json`. PDF presets must declare dimensions and either exactly three affine control points or a valid TFW transform. Routes that project outside the calibrated PDF are rejected instead of producing a misleading overlay.

Bundled aeronautical charts display a validity warning. Confirm that a chart is current and approved for the event before operational use. Updating a chart requires updating its file, metadata, calibration, tests, and generated bundle together.

## Deployment

GitHub Pages is configured for `main` and `/`. After changes are merged, Pages serves the root `index.html` and committed `assets/`. CI verifies that source, tests, types, formatting, and the production bundle remain consistent.
