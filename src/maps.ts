export type ControlPoint = [name: string, latitude: number, longitude: number, x: number, y: number];

export interface PreviewTileSet {
  baseUrl: string;
  width: number;
  height: number;
  tileSize: number;
  columns: number;
  rows: number;
}

export interface PdfMapPreset {
  label: string;
  type: 'pdf';
  edition: string;
  requiresValidityReview: boolean;
  fileName: string;
  assetPath: string;
  url: string;
  previewAssetPath: string;
  previewUrl: string;
  previewTiles: PreviewTileSet;
  baseWidth: number;
  baseHeight: number;
  scaleDenominator: number;
  printScale?: number;
  styleScale?: number;
  transform?: 'affine' | 'tfw';
  controlPoints?: ControlPoint[];
  tfw?: Record<string, number>;
}

export interface OsmMapPreset {
  label: string;
  type: 'osm';
  edition: string;
  requiresValidityReview: false;
  scaleDenominator: null;
}

export type MapPreset = PdfMapPreset | OsmMapPreset;
export type MapPresets = Record<string, MapPreset>;

function positiveNumber(value: unknown, field: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new Error(`Map preset ${field} must be a positive number.`);
  }
  return value;
}

export function loadMapPresets(raw: unknown, baseUrl: URL): MapPresets {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw))
    throw new Error('Map presets must be an object.');
  const presets: MapPresets = {};
  for (const [id, candidate] of Object.entries(raw)) {
    if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) {
      throw new Error(`Invalid map preset: ${id}`);
    }
    const preset = candidate as Record<string, unknown>;
    if (typeof preset.label !== 'string' || typeof preset.edition !== 'string') {
      throw new Error(`Map preset ${id} is missing its label or edition.`);
    }
    if (preset.type === 'osm') {
      presets[id] = {
        label: preset.label,
        type: 'osm',
        edition: preset.edition,
        requiresValidityReview: false,
        scaleDenominator: null,
      };
      continue;
    }
    if (
      preset.type !== 'pdf' ||
      typeof preset.fileName !== 'string' ||
      typeof preset.assetPath !== 'string' ||
      typeof preset.previewAssetPath !== 'string' ||
      !preset.previewTiles ||
      typeof preset.previewTiles !== 'object'
    ) {
      throw new Error(`PDF map preset ${id} is incomplete.`);
    }
    const transform = preset.transform === 'tfw' ? 'tfw' : 'affine';
    const controlPoints = preset.controlPoints as ControlPoint[] | undefined;
    if (transform === 'affine' && (!Array.isArray(controlPoints) || controlPoints.length !== 3)) {
      throw new Error(`Affine map preset ${id} requires exactly three control points.`);
    }
    if (transform === 'tfw' && (!preset.tfw || typeof preset.tfw !== 'object')) {
      throw new Error(`TFW map preset ${id} is missing its world-file parameters.`);
    }
    const rawTiles = preset.previewTiles as Record<string, unknown>;
    if (typeof rawTiles.basePath !== 'string') {
      throw new Error(`PDF map preset ${id} is missing its high-resolution preview tile path.`);
    }
    presets[id] = {
      ...(preset as unknown as Omit<PdfMapPreset, 'url'>),
      type: 'pdf',
      transform,
      baseWidth: positiveNumber(preset.baseWidth, `${id}.baseWidth`),
      baseHeight: positiveNumber(preset.baseHeight, `${id}.baseHeight`),
      scaleDenominator: positiveNumber(preset.scaleDenominator, `${id}.scaleDenominator`),
      printScale: preset.printScale === undefined ? 1 : positiveNumber(preset.printScale, `${id}.printScale`),
      url: new URL(preset.assetPath, baseUrl).href,
      previewUrl: new URL(preset.previewAssetPath, baseUrl).href,
      previewTiles: {
        baseUrl: new URL(rawTiles.basePath, baseUrl).href,
        width: positiveNumber(rawTiles.width, `${id}.previewTiles.width`),
        height: positiveNumber(rawTiles.height, `${id}.previewTiles.height`),
        tileSize: positiveNumber(rawTiles.tileSize, `${id}.previewTiles.tileSize`),
        columns: positiveNumber(rawTiles.columns, `${id}.previewTiles.columns`),
        rows: positiveNumber(rawTiles.rows, `${id}.previewTiles.rows`),
      },
    };
  }
  if (!presets.vfr) throw new Error('The default VFR map preset is missing.');
  return presets;
}
