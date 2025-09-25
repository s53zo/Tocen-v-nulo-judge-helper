# Repository Guidelines

## Project Structure & Module Organization
- `create_route_overlay.py` is the single entry point. It handles GPX parsing, photo analysis, overlay rendering, and handout generation. All configuration constants and feature toggles are defined near the top of the module.
- Required inputs live in the repository root: the base map (`1 200k original karta.pdf`), the GPX track (`activity_20435991472.gpx`), and camera photos matching `IMG_*.jpg`.
- Generated artefacts (`photo_analysis.csv`, `route_overlay.pdf`, `route_marked.pdf`, `photo_overlay_key.csv`, `photo_handout.pdf`) are written alongside the script. The raster cache `map_page.png` is created automatically if missing.

## Build, Test & Development Commands
- `pip install PyPDF2 reportlab` — install mandatory PDF toolchain into your virtual environment.
- `python create_route_overlay.py --speed 75` — full pipeline run. Adjust the `--speed` flag to test different groundspeeds. The script prints spacing/offset warnings that should be reviewed after each run.

## Configuration Highlights
- Map overlay visibility is controlled with module constants: `DRAW_ROUTE`, `DRAW_TURNPOINTS`, `DRAW_PHOTO_MARKERS`, `DRAW_CONTROL_PHOTO_MARKERS`, `DRAW_PHOTO_DOTS`, `DRAW_HEADINGS`, `DRAW_MINUTE_MARKERS`.
- Setting `DRAW_PHOTO_DOTS = True` draws exact GPS dots and enables the lateral-offset warning threshold (`PHOTO_MAX_LATERAL_DISTANCE_M`, default 300 m). When dots are disabled no lateral distance check is performed.
- Turnpoint time labels now respect `DRAW_MINUTE_MARKERS`; enabling minute markers automatically enables timing text, while `DRAW_TURNPOINTS` controls only the TP name glyph.
- Handout behaviour is configured via `HANDOUT_*` constants. Set `HANDOUT_INCLUDE_SUMMARY = False` to skip the executive-summary appendix. Photos are grouped into “before/after {HANDOUT_SPLIT_TP}` sections, and margins are tuned for near edge-to-edge prints.

## Coding Style & Naming Conventions
- Target Python 3.10+, PEP 8 compliant (4-space indentation, snake_case identifiers, upper-case module constants).
- Prefer type hints and f-strings. Keep new constants with the existing block at the top of `create_route_overlay.py`.

## Testing Guidelines
- There is no automated test suite. Validate changes by re-running `python create_route_overlay.py --speed 75` and inspecting regenerated PDFs/CSV outputs.
- When modifying geographic calculations, spacing logic, or control-photo handling, review console warnings and visually confirm affected photos in `route_marked.pdf` and the handout.

## Commit & Pull Request Guidelines
- Use concise, imperative commit messages (e.g., `Add control-photo exceptions for 1.0 NM rule`).
- PRs should summarise intent, list key impacts (new inputs, outputs, or toggles), and record manual verification (`Rebuilt overlay and reviewed TP4 timing photo`).
- Provide before/after overlays or handout excerpts for visual changes and reference related issues or tasks where applicable.
