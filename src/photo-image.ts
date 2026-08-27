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
  let source: CanvasImageSource;
  let sourceWidth: number;
  let sourceHeight: number;
  let close: () => void = () => undefined;
  if (typeof createImageBitmap === 'function') {
    try {
      // Both supported engines decode EXIF orientation with `from-image`.
      // Normalise the transform here so it is never applied a second time.
      const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
      source = bitmap;
      sourceWidth = bitmap.width;
      sourceHeight = bitmap.height;
      orientation = 1;
      close = () => bitmap.close();
    } catch {
      const bitmap = await createImageBitmap(file);
      source = bitmap;
      sourceWidth = bitmap.width;
      sourceHeight = bitmap.height;
      orientation = 1;
      close = () => bitmap.close();
    }
  } else {
    const url = URL.createObjectURL(file);
    const image = new Image();
    try {
      image.src = url;
      await image.decode();
      source = image;
      sourceWidth = image.naturalWidth;
      sourceHeight = image.naturalHeight;
      orientation = 1;
    } catch (error) {
      URL.revokeObjectURL(url);
      throw error;
    }
    close = () => URL.revokeObjectURL(url);
  }
  try {
    const [orientedWidth, orientedHeight] = orientationOutputSize(sourceWidth, sourceHeight, orientation);
    const scale = Math.min(1, maxEdge / Math.max(orientedWidth, orientedHeight));
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(orientedWidth * scale));
    canvas.height = Math.max(1, Math.round(orientedHeight * scale));
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Canvas rendering is unavailable.');
    context.fillStyle = '#fff';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.scale(scale, scale);
    context.transform(...orientationTransform(orientation, sourceWidth, sourceHeight));
    context.drawImage(source, 0, 0);
    const blob = await canvasToBlob(canvas, 'image/jpeg', 0.88);
    return new Uint8Array(await blob.arrayBuffer());
  } finally {
    close();
  }
}
