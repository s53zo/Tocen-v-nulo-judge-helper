import { describe, expect, it } from 'vitest';
import { parseCsv } from '../src/csv';

describe('CSV parsing', () => {
  it('supports commas, escaped quotes, and CRLF', () => {
    expect(parseCsv('name,id\r\n"Field, North","A""1"\r\n')).toEqual([
      ['name', 'id'],
      ['Field, North', 'A"1'],
    ]);
  });

  it('rejects unterminated fields', () => {
    expect(() => parseCsv('name\n"unfinished')).toThrow(/unterminated/);
  });
});
