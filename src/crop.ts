export interface CropPageChoice {
  width: number;
  height: number;
  format: 'A4 portrait' | 'A4 landscape' | 'A3 portrait' | 'A3 landscape';
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
  a4Portrait: readonly [number, number],
  a3Portrait: readonly [number, number]
): CropPageChoice | null {
  if (![contentWidth, contentHeight, ...a4Portrait, ...a3Portrait].every(Number.isFinite)) {
    throw new Error('Crop dimensions must be finite.');
  }
  if (
    contentWidth <= 0 ||
    contentHeight <= 0 ||
    a4Portrait.some((value) => value <= 0) ||
    a3Portrait.some((value) => value <= 0)
  ) {
    throw new Error('Crop dimensions must be positive.');
  }

  const a4PortraitChoice: CropPageChoice = {
    width: a4Portrait[0],
    height: a4Portrait[1],
    format: 'A4 portrait',
  };
  const a4LandscapeChoice: CropPageChoice = {
    width: a4Portrait[1],
    height: a4Portrait[0],
    format: 'A4 landscape',
  };
  const a3PortraitChoice: CropPageChoice = {
    width: a3Portrait[0],
    height: a3Portrait[1],
    format: 'A3 portrait',
  };
  const a3LandscapeChoice: CropPageChoice = {
    width: a3Portrait[1],
    height: a3Portrait[0],
    format: 'A3 landscape',
  };
  const orient = (portrait: CropPageChoice, landscape: CropPageChoice) =>
    contentWidth > contentHeight ? [landscape, portrait] : [portrait, landscape];
  return (
    [...orient(a4PortraitChoice, a4LandscapeChoice), ...orient(a3PortraitChoice, a3LandscapeChoice)].find(
      ({ width, height }) => contentWidth <= width && contentHeight <= height
    ) ?? null
  );
}
