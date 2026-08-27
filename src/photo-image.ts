import { orientationOutputSize } from './photo-metadata';

export type CanvasTransform = [number, number, number, number, number, number];

export function orientationTransform(orientation: number, width: number, height: number): CanvasTransform {
  switch (orientation) {
    case 2:
      return [-1, 0, 0, 1, width, 0];
    case 3:
      return [-1, 0, 0, -1, width, height];
    case 4:
      return [1, 0, 0, -1, 0, height];
    case 5:
      return [0, 1, 1, 0, 0, 0];
    case 6:
      return [0, 1, -1, 0, height, 0];
    case 7:
      return [0, -1, -1, 0, height, width];
    case 8:
      return [0, -1, 1, 0, 0, width];
    default:
      return [1, 0, 0, 1, 0, 0];
  }
}

function canvasToBlob(canvas: HTMLCanvasElement, type: string, quality: number): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('Could not encode the corrected photo.'))),
      type,
      quality
    );
  });
}

export async function preparePhotoJpeg(file: File, orientation: number, maxEdge = 2200): Promise<Uint8Array> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: 'none' });
  } catch {
    bitmap = await createImageBitmap(file);
    orientation = 1;
  }
  const [orientedWidth, orientedHeight] = orientationOutputSize(bitmap.width, bitmap.height, orientation);
  const scale = Math.min(1, maxEdge / Math.max(orientedWidth, orientedHeight));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(orientedWidth * scale));
  canvas.height = Math.max(1, Math.round(orientedHeight * scale));
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Canvas rendering is unavailable.');
  context.fillStyle = '#fff';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.scale(scale, scale);
  context.transform(...orientationTransform(orientation, bitmap.width, bitmap.height));
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  const blob = await canvasToBlob(canvas, 'image/jpeg', 0.9);
  return new Uint8Array(await blob.arrayBuffer());
}
