import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';
import { api } from '../api/client';
import { usePage } from '../hooks/queries';
import type { Citation } from '../types';
import { formatLocation } from '../utils';

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString();

interface Props {
  citation: Citation | null;
  expanded: boolean;
  onToggleExpand: () => void;
  onClose: () => void;
}

type Fit = 'passage' | 'width' | 'page';
type Box = [number, number, number, number]; // normalised [x0, y0, x1, y1]

const PAD = 10; // px of breathing room around the page inside the scroll area
const MIN_WIDTH = 200;
const MAX_ZOOM = 4; // × the page's natural width
const ZOOM_STEP = 1.25;
const PASSAGE_MARGIN = 0.05; // normalised margin kept either side of the passage when zooming to it
const MAX_PASSAGE_SCALE = 3; // × fit-width
const FIT_KEY = 'viewer.fit';

function loadFit(): Fit {
  try {
    const v = localStorage.getItem(FIT_KEY);
    if (v === 'passage' || v === 'width' || v === 'page') return v;
  } catch {
    /* storage unavailable */
  }
  return 'passage';
}

function saveFit(fit: Fit) {
  try {
    localStorage.setItem(FIT_KEY, fit);
  } catch {
    /* storage unavailable */
  }
}

/** Merge per-line boxes into one padded block per contiguous run of lines in a column. */
function mergeBoxes(boxes: number[][]): Box[] {
  const sorted = boxes
    .map((b) => [b[0], b[1], b[2], b[3]] as Box)
    .sort((a, b) => a[1] - b[1] || a[0] - b[0]);
  const out: Box[] = [];
  for (const b of sorted) {
    const lineH = b[3] - b[1];
    const host = out.find((m) => {
      const overlap = Math.min(m[2], b[2]) - Math.max(m[0], b[0]);
      const narrower = Math.min(m[2] - m[0], b[2] - b[0]);
      return overlap > 0.5 * narrower && b[1] - m[3] < lineH;
    });
    if (host) {
      host[0] = Math.min(host[0], b[0]);
      host[2] = Math.max(host[2], b[2]);
      host[3] = Math.max(host[3], b[3]);
    } else {
      out.push(b);
    }
  }
  const clamp = (n: number) => Math.min(Math.max(n, 0), 1);
  return out.map(([x0, y0, x1, y1]) => [
    clamp(x0 - 0.004),
    clamp(y0 - 0.003),
    clamp(x1 + 0.004),
    clamp(y1 + 0.003),
  ]);
}

function unionBox(boxes: Box[]): Box | null {
  if (!boxes.length) return null;
  return boxes.reduce<Box>(
    (u, b) => [Math.min(u[0], b[0]), Math.min(u[1], b[1]), Math.max(u[2], b[2]), Math.max(u[3], b[3])],
    [...boxes[0]] as Box,
  );
}

function useElementSize<T extends HTMLElement>() {
  const [el, setEl] = useState<T | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ width: Math.floor(width), height: Math.floor(height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [el]);
  return [el, setEl, size] as const;
}

function TextView({ citation, page }: { citation: Citation; page: number }) {
  const { data, isLoading, error } = usePage(citation.contract_id, page, true);
  if (isLoading) return <p className="muted pad">Loading page text…</p>;
  if (error || !data) return <p className="error pad">Could not load page text.</p>;
  const ranges = citation.line_ranges.filter((r) => r.page === page);
  const inRange = (n: number) => ranges.some((r) => n >= r.line_start && n <= r.line_end);
  return (
    <ol className="text-lines">
      {data.lines.map((l) => (
        <li key={l.line_number} className={inRange(l.line_number) ? 'hit' : ''} value={l.line_number}>
          <span className="ln">{l.line_number}</span>
          <span>{l.text}</span>
        </li>
      ))}
    </ol>
  );
}

function ExpandIcon({ expanded }: { expanded: boolean }) {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
      {expanded ? (
        <path d="M5 1v4H1M9 13V9h4M5 5 1 1M9 9l4 4" />
      ) : (
        <path d="M1 5V1h4M13 9v4H9M1 1l4 4M13 13 9 9" />
      )}
    </svg>
  );
}

export function SourceViewer({ citation, expanded, onToggleExpand, onClose }: Props) {
  const [page, setPage] = useState(1);
  const [numPages, setNumPages] = useState<number | null>(null);
  const [mode, setMode] = useState<'pdf' | 'text'>('pdf');
  const [pdfError, setPdfError] = useState(false);
  const [fit, setFitState] = useState<Fit>(loadFit);
  const [customWidth, setCustomWidth] = useState<number | null>(null);
  const [focusTick, setFocusTick] = useState(0);
  const [loaded, setLoaded] = useState<{ page: number; w: number; h: number } | null>(null);
  const [bodyEl, setBodyEl, size] = useElementSize<HTMLDivElement>();
  const anchorRef = useRef<HTMLDivElement>(null);
  const centre = useRef({ rx: 0, ry: 0 });
  const widthRef = useRef(0);
  const maxWidthRef = useRef(MIN_WIDTH);

  const chunkId = citation?.chunk_id;
  const contractId = citation?.contract_id;

  useEffect(() => {
    if (citation) {
      setPage(citation.page_start);
      setCustomWidth(null); // opening a citation always zooms back to the passage
    }
  }, [chunkId, contractId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    setPdfError(false);
    setNumPages(null);
    setLoaded(null);
  }, [contractId]);

  // Highlights on the current page: one merged block per run of lines.
  const blocks = useMemo(
    () => mergeBoxes((citation?.highlights ?? []).filter((h) => h.page === page).flatMap((h) => h.bboxes)),
    [citation, page],
  );
  const passage = useMemo(() => unionBox(blocks), [blocks]);

  const total = numPages ?? Math.max(citation?.page_end ?? 1, page);
  const useText = mode === 'text' || pdfError;

  // ---- sizing: derive the rendered page width from the fit mode ----
  const aspect = loaded ? loaded.h / loaded.w : 1.414;
  const availW = Math.max(size.width - 2 * PAD, MIN_WIDTH);
  const availH = Math.max(size.height - 2 * PAD, MIN_WIDTH);
  const maxWidth = Math.max((loaded?.w ?? 600) * MAX_ZOOM, availW);
  const effectiveFit: Fit = fit === 'passage' && !passage ? 'width' : fit;

  let target = availW;
  if (customWidth != null) {
    target = customWidth;
  } else if (effectiveFit === 'page') {
    target = Math.min(availW, availH / aspect);
  } else if (effectiveFit === 'passage' && passage) {
    const span = passage[2] - passage[0] + 2 * PASSAGE_MARGIN;
    target = Math.min(availW / Math.min(span, 1), availW * MAX_PASSAGE_SCALE);
  }
  const renderWidth = Math.round(Math.min(Math.max(target, MIN_WIDTH), maxWidth));
  widthRef.current = renderWidth;
  maxWidthRef.current = maxWidth;

  const zoomBy = (factor: number) =>
    setCustomWidth((prev) =>
      Math.min(Math.max((prev ?? widthRef.current) * factor, MIN_WIDTH), maxWidthRef.current),
    );

  const chooseFit = (f: Fit) => {
    setFitState(f);
    saveFit(f);
    setCustomWidth(null);
    setFocusTick((t) => t + 1);
  };

  // Keep whatever the user was looking at centred while the page width changes.
  const rememberCentre = () => {
    if (!bodyEl) return;
    centre.current = {
      rx: bodyEl.scrollWidth ? (bodyEl.scrollLeft + bodyEl.clientWidth / 2) / bodyEl.scrollWidth : 0,
      ry: bodyEl.scrollHeight ? (bodyEl.scrollTop + bodyEl.clientHeight / 2) / bodyEl.scrollHeight : 0,
    };
  };
  useLayoutEffect(() => {
    if (!bodyEl) return;
    const { rx, ry } = centre.current;
    bodyEl.scrollLeft = rx * bodyEl.scrollWidth - bodyEl.clientWidth / 2;
    bodyEl.scrollTop = ry * bodyEl.scrollHeight - bodyEl.clientHeight / 2;
  }, [renderWidth]); // eslint-disable-line react-hooks/exhaustive-deps

  // Bring the cited passage into view whenever a page loads, a citation is opened or a fit preset is picked.
  useEffect(() => {
    if (!bodyEl || useText || !loaded || loaded.page !== page) return;
    const id = requestAnimationFrame(() => {
      const anchor = anchorRef.current;
      if (!anchor) {
        bodyEl.scrollTo({ left: 0, top: 0 });
        return;
      }
      const er = bodyEl.getBoundingClientRect();
      const ar = anchor.getBoundingClientRect();
      const dx = ar.left + ar.width / 2 - (er.left + er.width / 2);
      let dy = ar.top + ar.height / 2 - (er.top + er.height / 2);
      // A passage taller than the viewport: line its top edge up instead of centring it.
      if (ar.height > er.height - 2 * PAD) dy = ar.top - er.top - 2 * PAD;
      bodyEl.scrollBy({ left: dx, top: dy, behavior: 'smooth' });
    });
    return () => cancelAnimationFrame(id);
  }, [bodyEl, useText, loaded, page, chunkId, focusTick]);

  // Ctrl/⌘ + wheel zooms the page instead of the browser.
  useEffect(() => {
    if (!bodyEl || useText) return;
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      zoomBy(Math.exp(-e.deltaY * 0.002));
    };
    bodyEl.addEventListener('wheel', onWheel, { passive: false });
    return () => bodyEl.removeEventListener('wheel', onWheel);
  }, [bodyEl, useText]);

  if (!citation) {
    return (
      <div className="empty small-empty">
        <h3>Source viewer</h3>
        <p className="muted">
          Click a citation (e.g. <span className="chip">C1</span>) to see the exact passage highlighted
          in the original contract.
        </p>
      </div>
    );
  }

  const zoomPct = loaded ? Math.round((renderWidth / loaded.w) * 100) : null;
  const activeFit = customWidth == null ? effectiveFit : null;
  const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 4096 / renderWidth));

  return (
    <div className="source">
      <div className="source-head">
        <div className="source-heading">
          <div className="source-title" title={citation.contract_name}>
            {citation.contract_name}
          </div>
          <div className="muted small">{formatLocation(citation)}</div>
        </div>
        <div className="source-actions">
          <button
            className="icon-btn"
            onClick={onToggleExpand}
            aria-label={expanded ? 'Collapse source viewer' : 'Expand source viewer'}
            title={expanded ? 'Collapse viewer' : 'Expand viewer'}
          >
            <ExpandIcon expanded={expanded} />
          </button>
          <button className="icon-btn" onClick={onClose} aria-label="Close source viewer">
            ✕
          </button>
        </div>
      </div>

      <div className="source-toolbar">
        <button disabled={page <= 1} onClick={() => setPage((p) => p - 1)} aria-label="Previous page">
          ‹
        </button>
        <span>
          Page {page} / {total}
        </span>
        <button disabled={page >= total} onClick={() => setPage((p) => p + 1)} aria-label="Next page">
          ›
        </button>
        <span className="spacer" />
        <div className="seg">
          <button className={!useText ? 'on' : ''} onClick={() => setMode('pdf')} disabled={pdfError}>
            PDF
          </button>
          <button className={useText ? 'on' : ''} onClick={() => setMode('text')}>
            Text
          </button>
        </div>
      </div>

      {!useText && (
        <div className="source-toolbar compact">
          <button
            onClick={() => zoomBy(1 / ZOOM_STEP)}
            disabled={renderWidth <= MIN_WIDTH}
            aria-label="Zoom out"
            title="Zoom out (Ctrl + scroll)"
          >
            −
          </button>
          <span className="zoom-pct" aria-live="polite">
            {zoomPct != null ? `${zoomPct}%` : '…'}
          </span>
          <button
            onClick={() => zoomBy(ZOOM_STEP)}
            disabled={renderWidth >= maxWidth}
            aria-label="Zoom in"
            title="Zoom in (Ctrl + scroll)"
          >
            +
          </button>
          <span className="spacer" />
          <div className="seg" role="group" aria-label="Fit">
            <button
              className={activeFit === 'passage' ? 'on' : ''}
              onClick={() => chooseFit('passage')}
              disabled={!passage}
              title={passage ? 'Zoom to the cited passage' : 'No highlighted passage on this page'}
            >
              Passage
            </button>
            <button
              className={activeFit === 'width' ? 'on' : ''}
              onClick={() => chooseFit('width')}
              title="Fit page width"
            >
              Width
            </button>
            <button
              className={activeFit === 'page' ? 'on' : ''}
              onClick={() => chooseFit('page')}
              title="Fit whole page"
            >
              Page
            </button>
          </div>
        </div>
      )}

      <div className="source-body" ref={setBodyEl} onScroll={rememberCentre}>
        {useText ? (
          <TextView citation={citation} page={page} />
        ) : (
          size.width > 0 && (
            <Document
              file={api.fileUrl(citation.contract_id)}
              loading={<p className="muted pad">Loading PDF…</p>}
              error={<p className="error pad">Could not render the PDF.</p>}
              onLoadSuccess={(d) => setNumPages(d.numPages)}
              onLoadError={() => setPdfError(true)}
            >
              <div className="pdf-stage">
                <div className="pdf-page">
                  <Page
                    pageNumber={page}
                    width={renderWidth}
                    devicePixelRatio={dpr}
                    renderAnnotationLayer={false}
                    onLoadSuccess={(p) =>
                      setLoaded({ page: p.pageNumber, w: p.originalWidth, h: p.originalHeight })
                    }
                  />
                  <div className="overlay">
                    {blocks.map(([x0, y0, x1, y1], i) => (
                      <div key={i} className="hl" style={boxStyle(x0, y0, x1, y1)} />
                    ))}
                    {passage && (
                      <div ref={anchorRef} className="hl-anchor" style={boxStyle(...passage)} />
                    )}
                  </div>
                </div>
              </div>
            </Document>
          )
        )}
      </div>
    </div>
  );
}

function boxStyle(x0: number, y0: number, x1: number, y1: number) {
  return {
    left: `${x0 * 100}%`,
    top: `${y0 * 100}%`,
    width: `${(x1 - x0) * 100}%`,
    height: `${(y1 - y0) * 100}%`,
  };
}
