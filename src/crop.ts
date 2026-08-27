export interface CropPageChoice {
  width: number;
  height: number;
  format: 'A4 portrait' | 'A4 landscape' | 'custom true-scale';
}

export function choosePreviewScale(
  width: number,
  height: number,
  maximumScale: number,
  maximumEdge: number,
  maximumPixels: number
): number {
  if (![width, height, maximumScale, maximumEdge, maximumPixels].every(Number.isFinite)) {
    throw new Error('Preview dimensions must be finite.');
  }
  if ([width, height, maximumScale, maximumEdge, maximumPixels].some((value) => value <= 0)) {
    throw new Error('Preview dimensions must be positive.');
  }
  return Math.min(
    maximumScale,
    maximumEdge / Math.max(width, height),
    Math.sqrt(maximumPixels / (width * height))
  );
}

export function chooseTrueScaleCropPage(
  contentWidth: number,
  contentHeight: number,
  a4Portrait: readonly [number, number]
): CropPageChoice {
  if (![contentWidth, contentHeight, ...a4Portrait].every(Number.isFinite)) {
    throw new Error('Crop dimensions must be finite.');
  }
  if (contentWidth <= 0 || contentHeight <= 0 || a4Portrait.some((value) => value <= 0)) {
    throw new Error('Crop dimensions must be positive.');
  }

  const portrait: CropPageChoice = {
    width: a4Portrait[0],
    height: a4Portrait[1],
    format: 'A4 portrait',
  };
  const landscape: CropPageChoice = {
    width: a4Portrait[1],
    height: a4Portrait[0],
    format: 'A4 landscape',
  };
  const preferred = contentWidth > contentHeight ? landscape : portrait;
  const alternate = preferred === landscape ? portrait : landscape;
  const fittingA4 = [preferred, alternate].find(
    ({ width, height }) => contentWidth <= width && contentHeight <= height
  );
  return (
    fittingA4 ?? {
      width: contentWidth,
      height: contentHeight,
      format: 'custom true-scale',
    }
  );
}
