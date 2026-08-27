import { describe, expect, it } from 'vitest';
import { decodeDataUrl } from '../src/data-url';

describe('data URL decoding', () => {
  it('decodes base64 assets without a network request', () => {
    const decoded = decodeDataUrl('data:font/woff;base64,SGk=');

    expect(decoded.mimeType).toBe('font/woff');
    expect(Array.from(decoded.bytes)).toEqual([72, 105]);
  });

  it('decodes percent-encoded assets', () => {
    const decoded = decodeDataUrl('data:text/plain,hello%20world');

    expect(new TextDecoder().decode(decoded.bytes)).toBe('hello world');
  });

  it('rejects malformed values', () => {
    expect(() => decodeDataUrl('data:text/plain')).toThrow(/Malformed/);
    expect(() => decodeDataUrl('https://example.test/file')).toThrow(/Expected/);
  });
});
