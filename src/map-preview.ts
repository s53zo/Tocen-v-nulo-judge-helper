import { choosePreviewScale } from './crop';
import type { PreviewTileSet } from './maps';

export interface PdfCropBounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export interface PreviewPoint {
  x: number;
  y: number;
  label?: string;
  color?: string;
  warning?: boolean;
}

export interface PreviewGeometry {
  sourceX: number;
  sourceY: number;
  sourceWidth: number;
  sourceHeight: number;
  outputWidth: number;
  outputHeight: number;
}

export interface MapPreviewOptions {
  imageUrl?: string;
  tileSet?: PreviewTileSet;
  pageWidth: number;
  pageHeight: number;
  crop: PdfCropBounds;
  route: PreviewPoint[];
  photos: PreviewPoint[];
  signal?: AbortSignal;
  maximumBytes?: number;
  maximumEdge?: number;
  maximumPixels?: number;
  maximumTiles?: number;
  maximumTotalTileBytes?: number;
}

const DEFAULT_MAXIMUM_BYTES = 3_000_000;
const DEFAULT_MAXIMUM_EDGE = 4096;
const DEFAULT_MAXIMUM_PIXELS = 12_000_000;
const DEFAULT_MAXIMUM_TILES = 128;
const DEFAULT_MAXIMUM_TOTAL_TILE_BYTES = 40_000_000;

export function waitForAbortSignal<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(signal.reason ?? new DOMException('Cancelled', 'AbortError'));
  return new Promise<T>((resolve, reject) => {
    const abort = () => reject(signal.reason ?? new DOMException('Cancelled', 'AbortError'));
    signal.addEventListener('abort', abort, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener('abort', abort);
        resolve(value);
      },
      (error) => {
        signal.removeEventListener('abort', abort);
        reject(error);
      }
    );
  });
}

function validateCrop(pageWidth: number, pageHeight: number, crop: PdfCropBounds): void {
  const values = [pageWidth, pageHeight, crop.minX, crop.minY, crop.maxX, crop.maxY];
  if (!values.every(Number.isFinite)) throw new Error('Preview geometry must be finite.');
  if (pageWidth <= 0 || pageHeight <= 0 || crop.maxX <= crop.minX || crop.maxY <= crop.minY) {
    throw new Error('Preview geometry must have positive dimensions.');
  }
  if (crop.minX < 0 || crop.minY < 0 || crop.maxX > pageWidth || crop.maxY > pageHeight) {
    throw new Error('Preview crop must stay within the source page.');
  }
}

export function computePreviewGeometry(
  imageWidth: number,
  imageHeight: number,
  pageWidth: number,
  pageHeight: number,
  crop: PdfCropBounds,
  maximumEdge = DEFAULT_MAXIMUM_EDGE,
  maximumPixels = DEFAULT_MAXIMUM_PIXELS
): PreviewGeometry {
  validateCrop(pageWidth, pageHeight, crop);
  if (![imageWidth, imageHeight].every(Number.isFinite) || imageWidth <= 0 || imageHeight <= 0) {
    throw new Error('Preview image dimensions must be positive and finite.');
  }
  const sourceX = (crop.minX / pageWidth) * imageWidth;
  const sourceY = ((pageHeight - crop.maxY) / pageHeight) * imageHeight;
  const sourceWidth = ((crop.maxX - crop.minX) / pageWidth) * imageWidth;
  const sourceHeight = ((crop.maxY - crop.minY) / pageHeight) * imageHeight;
  const scale = choosePreviewScale(sourceWidth, sourceHeight, 1, maximumEdge, maximumPixels);
  return {
    sourceX,
    sourceY,
    sourceWidth,
    sourceHeight,
    outputWidth: Math.max(1, Math.round(sourceWidth * scale)),
    outputHeight: Math.max(1, Math.round(sourceHeight * scale)),
  };
}

function canvasToBlob(canvas: HTMLCanvasElement, signal?: AbortSignal): Promise<Blob> {
  return waitForAbortSignal(
    new Promise((resolve, reject) => {
      if (signal?.aborted) return reject(signal.reason ?? new DOMException('Cancelled', 'AbortError'));
      canvas.toBlob(
        (blob) => (blob ? resolve(blob) : reject(new Error('Could not encode the map preview.'))),
        'image/jpeg',
        0.93
      );
    }),
    signal
  );
}

async function loadImage(
  url: string,
  signal: AbortSignal | undefined,
  maximumBytes: number
): Promise<{ image: HTMLImageElement; bytes: number }> {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Preview map returned ${response.status}.`);
  const declaredBytes = Number(response.headers.get('content-length'));
  if (Number.isFinite(declaredBytes) && declaredBytes > maximumBytes) {
    throw new Error(`Preview map exceeds the ${maximumBytes} byte limit.`);
  }
  const blob = await response.blob();
  if (blob.size > maximumBytes) throw new Error(`Preview map exceeds the ${maximumBytes} byte limit.`);
  const objectUrl = URL.createObjectURL(blob);
  try {
    const image = new Image();
    image.decoding = 'async';
    image.src = objectUrl;
    await waitForAbortSignal(image.decode(), signal);
    return { image, bytes: blob.size };
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

export interface PreviewTileCoordinate {
  column: number;
  row: number;
}

export function tilesForGeometry(
  geometry: PreviewGeometry,
  tileSet: PreviewTileSet
): PreviewTileCoordinate[] {
  const firstColumn = Math.max(0, Math.floor(geometry.sourceX / tileSet.tileSize));
  const lastColumn = Math.min(
    tileSet.columns - 1,
    Math.ceil((geometry.sourceX + geometry.sourceWidth) / tileSet.tileSize) - 1
  );
  const firstRow = Math.max(0, Math.floor(geometry.sourceY / tileSet.tileSize));
  const lastRow = Math.min(
    tileSet.rows - 1,
    Math.ceil((geometry.sourceY + geometry.sourceHeight) / tileSet.tileSize) - 1
  );
  const tiles: PreviewTileCoordinate[] = [];
  for (let row = firstRow; row <= lastRow; row += 1) {
    for (let column = firstColumn; column <= lastColumn; column += 1) tiles.push({ column, row });
  }
  return tiles;
}

async function drawTileSet(
  context: CanvasRenderingContext2D,
  geometry: PreviewGeometry,
  tileSet: PreviewTileSet,
  signal: AbortSignal | undefined,
  maximumTileBytes: number,
  maximumTiles: number,
  maximumTotalBytes: number
): Promise<void> {
  const tiles = tilesForGeometry(geometry, tileSet);
  if (tiles.length > maximumTiles) {
    throw new Error(`High-resolution preview requires ${tiles.length} tiles; limit is ${maximumTiles}.`);
  }
  let totalBytes = 0;
  const scaleX = geometry.outputWidth / geometry.sourceWidth;
  const scaleY = geometry.outputHeight / geometry.sourceHeight;
  for (let batchStart = 0; batchStart < tiles.length; batchStart += 4) {
    signal?.throwIfAborted();
    const batch = tiles.slice(batchStart, batchStart + 4);
    const loaded = await Promise.all(
      batch.map(async ({ column, row }) => ({
        column,
        row,
        ...(await loadImage(
          new URL(`${column}-${row}.webp`, tileSet.baseUrl).href,
          signal,
          maximumTileBytes
        )),
      }))
    );
    for (const { column, row, image, bytes } of loaded) {
      try {
        totalBytes += bytes;
        if (totalBytes > maximumTotalBytes) {
          throw new Error(`High-resolution preview exceeds the ${maximumTotalBytes} byte budget.`);
        }
        const tileX = column * tileSet.tileSize;
        const tileY = row * tileSet.tileSize;
        const intersectionX = Math.max(tileX, geometry.sourceX);
        const intersectionY = Math.max(tileY, geometry.sourceY);
        const intersectionRight = Math.min(
          tileX + image.naturalWidth,
          geometry.sourceX + geometry.sourceWidth
        );
        const intersectionBottom = Math.min(
          tileY + image.naturalHeight,
          geometry.sourceY + geometry.sourceHeight
        );
        if (intersectionRight <= intersectionX || intersectionBottom <= intersectionY) continue;
        const width = intersectionRight - intersectionX;
        const height = intersectionBottom - intersectionY;
        context.drawImage(
          image,
          intersectionX - tileX,
          intersectionY - tileY,
          width,
          height,
          (intersectionX - geometry.sourceX) * scaleX,
          (intersectionY - geometry.sourceY) * scaleY,
          width * scaleX,
          height * scaleY
        );
      } finally {
        image.removeAttribute('src');
      }
    }
  }
}

function mapPoint(point: PreviewPoint, crop: PdfCropBounds, geometry: PreviewGeometry): [number, number] {
  return [
    ((point.x - crop.minX) / (crop.maxX - crop.minX)) * geometry.outputWidth,
    ((crop.maxY - point.y) / (crop.maxY - crop.minY)) * geometry.outputHeight,
  ];
}

export async function renderBoundedMapPreview(options: MapPreviewOptions): Promise<Blob> {
  const maximumBytes = options.maximumBytes ?? DEFAULT_MAXIMUM_BYTES;
  if (!options.tileSet && !options.imageUrl) throw new Error('A preview image or tile set is required.');
  let fallbackImage: HTMLImageElement | null = null;
  const imageDimensions = options.tileSet
    ? [options.tileSet.width, options.tileSet.height]
    : await loadImage(options.imageUrl as string, options.signal, maximumBytes).then(({ image }) => {
        fallbackImage = image;
        return [image.naturalWidth, image.naturalHeight];
      });
  options.signal?.throwIfAborted();
  const geometry = computePreviewGeometry(
    imageDimensions[0],
    imageDimensions[1],
    options.pageWidth,
    options.pageHeight,
    options.crop,
    options.maximumEdge,
    options.maximumPixels
  );
  const canvas = document.createElement('canvas');
  canvas.width = geometry.outputWidth;
  canvas.height = geometry.outputHeight;
  const context = canvas.getContext('2d', { alpha: false });
  if (!context) throw new Error('Canvas rendering is unavailable.');
  context.fillStyle = '#fff';
  context.fillRect(0, 0, canvas.width, canvas.height);
  if (options.tileSet) {
    await drawTileSet(
      context,
      geometry,
      options.tileSet,
      options.signal,
      maximumBytes,
      options.maximumTiles ?? DEFAULT_MAXIMUM_TILES,
      options.maximumTotalTileBytes ?? DEFAULT_MAXIMUM_TOTAL_TILE_BYTES
    );
  } else {
    context.drawImage(
      fallbackImage as HTMLImageElement,
      geometry.sourceX,
      geometry.sourceY,
      geometry.sourceWidth,
      geometry.sourceHeight,
      0,
      0,
      geometry.outputWidth,
      geometry.outputHeight
    );
    fallbackImage?.removeAttribute('src');
  }

  if (options.route.length > 1) {
    context.save();
    context.strokeStyle = '#d21f26';
    context.lineWidth = Math.max(2, geometry.outputWidth / 650);
    context.lineJoin = 'round';
    context.beginPath();
    options.route.forEach((point, index) => {
      const [x, y] = mapPoint(point, options.crop, geometry);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
    context.restore();
  }

  context.font = `600 ${Math.max(10, Math.round(geometry.outputWidth / 120))}px sans-serif`;
  context.textBaseline = 'bottom';
  for (const point of options.route) {
    const [x, y] = mapPoint(point, options.crop, geometry);
    context.fillStyle = '#fff';
    context.strokeStyle = '#d21f26';
    context.lineWidth = 2;
    context.beginPath();
    context.arc(x, y, Math.max(4, geometry.outputWidth / 300), 0, Math.PI * 2);
    context.fill();
    context.stroke();
    if (point.label) {
      context.lineWidth = 3;
      context.strokeStyle = '#fff';
      context.strokeText(point.label, x + 7, y - 5);
      context.fillStyle = '#8f1017';
      context.fillText(point.label, x + 7, y - 5);
    }
  }

  for (const point of options.photos) {
    const [x, y] = mapPoint(point, options.crop, geometry);
    context.fillStyle = point.color ?? '#7331a5';
    context.strokeStyle = point.warning ? '#d00000' : '#fff';
    context.lineWidth = point.warning ? 3 : 2;
    context.beginPath();
    context.arc(x, y, Math.max(5, geometry.outputWidth / 240), 0, Math.PI * 2);
    context.fill();
    context.stroke();
    if (point.label) {
      context.lineWidth = 3;
      context.strokeStyle = '#fff';
      context.strokeText(point.label, x + 8, y - 6);
      context.fillStyle = point.color ?? '#7331a5';
      context.fillText(point.label, x + 8, y - 6);
    }
  }

  options.signal?.throwIfAborted();
  return canvasToBlob(canvas, options.signal);
}
