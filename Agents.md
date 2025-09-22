 # Repository Guidelines

  ## Project Structure & Module Organization

  - create_route_overlay.py ??? single entry point; orchestrates GPX parsing, photo analysis, PDF overlay generation, and handout creation. Configuration constants and style toggles live near the top.
  - Data inputs live at repo root: source map (1 200k original karta.pdf), GPX track (activity_20435991472.gpx), and photos matching IMG_*.jpg.
  - Generated outputs are written alongside the script: photo_analysis.csv, route_overlay.pdf, route_marked.pdf, photo_overlay_key.csv, and photo_handout.pdf. Temporary raster map_page.png is auto-
  created if missing.

  ## Build, Test & Development Commands

  - pip install PyPDF2 reportlab ??? install required PDF libraries inside your virtual environment.
  - python create_route_overlay.py --speed 75 ??? end-to-end run: parses inputs, enforces 1???NM rules, generates overlays and handout. Adjust --speed to test different groundspeeds.

  ## Coding Style & Naming Conventions

  - Python 3.10+ with PEP???8 conventions: 4-space indentation, snake_case variables, UPPER_CASE module constants.
  - Prefer type hints (used throughout) and f-strings for formatting.
  - Configuration toggles and colour palettes belong with the existing constant block; keep new constants grouped there.

  ## Testing Guidelines

  - No automated test suite; validate changes by re-running python create_route_overlay.py --speed 75 and reviewing regenerated PDFs/CSV.
  - When altering geographic maths or exclusion rules, spot-check the console warnings and verify affected photos visually in route_marked.pdf.

  ## Commit & Pull Request Guidelines

  - Write concise, imperative commit messages (e.g., ???Add control-photo exceptions for 1???NM rule???).
  - PRs should summarize intent, list key impacts (new inputs, outputs, or settings), and note manual verification steps (e.g., ???Rebuilt overlay and reviewed TP4 timing photo???).
  - Include before/after screenshots of overlays or handout pages when visual changes are introduced, and reference related issues or tasks.

