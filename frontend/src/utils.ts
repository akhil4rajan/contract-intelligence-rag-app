import type { Citation } from './types';

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/** "Page 3, lines 12–18" or "Page 3 line 40 → Page 4 line 5" */
export function formatLocation(c: Citation): string {
  if (c.page_start === c.page_end) {
    return c.line_start === c.line_end
      ? `Page ${c.page_start}, line ${c.line_start}`
      : `Page ${c.page_start}, lines ${c.line_start}–${c.line_end}`;
  }
  return `Page ${c.page_start} line ${c.line_start} → Page ${c.page_end} line ${c.line_end}`;
}

export function isPdf(file: File): boolean {
  return file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
}

export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

/** "2025-01-01" -> "January 1, 2025" (parsed by hand: `new Date(iso)` shifts by timezone). */
export function formatDay(iso: string): string {
  const [y, m, d] = iso.split('-').map(Number);
  if (!y || !m || !d) return iso;
  return new Date(y, m - 1, d).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
}
