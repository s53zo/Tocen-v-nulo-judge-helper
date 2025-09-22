# Route Overlay Generator

This repository contains a single Python entry point, `create_route_overlay.py`, that builds briefing material for a precision flying route. The script parses GPX data, aligns onboard photos, draws map overlays, and assembles a handout with an executive summary plus a cropped map page.

## Prerequisites

- Python 3.10+
- macOS command‐line tools
  - `sips` (used for rasterising the base PDF and reading EXIF timestamps)
- Required Python packages (install inside your virtual environment):

```bash
pip install PyPDF2 reportlab Pillow
```

## Expected Inputs

Place the following files in the repository root before running the script:

- `1 200k original karta.pdf` – base map PDF (page 1 is used)
- `activity_20435991472.gpx` – GPX track of the flight
- `IMG_*.jpg` – onboard photos (timestamps read via `sips`)

The script will generate `map_page.png` automatically if it does not exist.

## Usage

```bash
python create_route_overlay.py --speed 75
```

- `--speed` accepts values such as `75`, `75kt`, `75 mph`, or `80kts`.
- If omitted, the default speed (configured near the top of the script) is used.

The command produces:

| Output | Description |
| --- | --- |
| `photo_analysis.csv` | Interpolated photo positions, leg distances, and timing data |
| `route_overlay.pdf` | Overlay (vector) drawn on top of the base map |
| `route_marked.pdf` | Base map merged with the overlay |
| `photo_overlay_key.csv` | Handy legend linking photo IDs to legs and map distances |
| `route_cropped.pdf` | Cropped true-scale map page (portrait or landscape A4, auto-selected) |
| `photo_handout.pdf` | Handout containing the executive summary, photo grid, and cropped map |

## Handout Structure

1. **Executive Summary** – Metrics (speed, distance, photo counts, waypoint ETAs, leg bearings) plus a route footprint schematic. If `map_page.png` is present, a desaturated background map is rendered behind the route trace.
2. **Photo Grid** – Control photos first, then enroute photos. Enroute shots use deterministic pseudo-random lettering (stable across runs). Control photos retain their fix labels (`SP`, `TPx`, `FP`).
3. **Cropped Map** – True-scale crop positioned at 100% print size. The script automatically chooses portrait or landscape orientation based on the route footprint.

## Configuration Cheatsheet

All tuning happens near the top of `create_route_overlay.py`:

- **Paths & Outputs** – Update `PDF_PATH`, `GPX_PATH`, and file names as needed.
- **Drawing Toggles** – Enable/disable layers via `DRAW_ROUTE`, `DRAW_TURNPOINTS`, `DRAW_PHOTO_MARKERS`, and `DRAW_HEADINGS`.
- **Visual Styles** – Fonts, line widths, colours, and label offsets are grouped with the other constants.
- **Handout Behaviour**
  - `HANDOUT_LETTER_SALT` – Salt used for stable pseudo-random photo lettering.
  - `HANDOUT_SPLIT_TP` – Turnpoint that divides enroute photos into “first half” and “second half” for lettering.
  - `HANDOUT_LETTER_SCALE` – Controls the size of the red letters over handout photos.
- **Cropped Map**
  - `CROPPED_MAP_MARGIN_MM` – Desired margin (per axis) inserted around the route while maintaining scale. If the route nearly fills the chosen orientation, margins collapse automatically.

Every change to geometry or spacing rules should be validated by re-running the script and reviewing `route_marked.pdf` plus the handout.

## Operational Notes

- The route footprint schematic always reflects the generated overlay (not a generic template).
- Cropped maps never scale the base map; they simply crop, pad, and center on A4.
- Console output summarises key results, including warnings when photos are suppressed or kept despite proximity constraints.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `ModuleNotFoundError: PyPDF2` or `reportlab` | Install the packages inside the active environment: `pip install PyPDF2 reportlab` |
| `sips` errors | Ensure you are running on macOS and that the CLI tools are available. |
| Blank `route_cropped.pdf` | Verify the overlay run completed without errors and that the route bounds are valid. The script logs the final crop dimensions and orientation. |
| Handout missing summary background | Confirm `map_page.png` exists. The summary falls back to a neutral background if the image cannot be drawn. |

## Contributing

- Keep new constants grouped with the existing configuration block.
- When editing geometry or timing logic, re-run the script and inspect both PDFs plus the generated CSVs.
- The repository intentionally does not track generated artefacts; delete or move them before committing.

Enjoy streamlined handout generation! If you tailor the route, update the constants accordingly and rerun the single entry point.
