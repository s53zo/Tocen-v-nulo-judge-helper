export interface DecodedDataUrl {
  mimeType: string;
  bytes: Uint8Array;
}

export function decodeDataUrl(value: string): DecodedDataUrl {
  if (!value.startsWith('data:')) {
    throw new Error('Expected a data URL.');
  }
  const commaIndex = value.indexOf(',');
  if (commaIndex < 0) {
    throw new Error('Malformed data URL.');
  }

  const metadata = value.slice(5, commaIndex).split(';');
  const mimeType = metadata[0] || 'application/octet-stream';
  const payload = value.slice(commaIndex + 1);
  if (metadata.includes('base64')) {
    const binary = atob(payload);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return { mimeType, bytes };
  }

  return { mimeType, bytes: new TextEncoder().encode(decodeURIComponent(payload)) };
}
