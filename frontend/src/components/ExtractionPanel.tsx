import type { ReactNode } from 'react';
import { useExtraction, useRefreshExtraction } from '../hooks/queries';
import type { Citation, Contract, ContractTerms } from '../types';
import { formatDay, formatLocation } from '../utils';

interface Props {
  contracts: Contract[];
  selectedIds: string[];
  activeChunkId: number | null;
  onOpenCitation: (c: Citation) => void;
}

interface SourceProps {
  citations: Citation[];
  activeChunkId: number | null;
  onOpen: (c: Citation) => void;
}

/** Links to the passage(s) a value was read from; the viewer highlights those lines. */
function Source({ citations, activeChunkId, onOpen }: SourceProps) {
  if (!citations.length) return <span className="muted small">no source located</span>;
  return (
    <>
      {citations.map((c) => (
        <button
          key={`${c.chunk_id}-${c.line_start}`}
          className={`chip chip-btn source-link${c.chunk_id === activeChunkId ? ' active' : ''}`}
          onClick={() => onOpen(c)}
          title="Open in source viewer"
        >
          {formatLocation(c)}
        </button>
      ))}
    </>
  );
}

function Row({ label, hint, children }: { label: string; hint: string; children: ReactNode }) {
  return (
    <div className="term-row">
      <dt title={hint}>{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

function Terms({
  terms,
  activeChunkId,
  onOpen,
}: {
  terms: ContractTerms;
  activeChunkId: number | null;
  onOpen: (c: Citation) => void;
}) {
  const src = (citations: Citation[]) => (
    <Source citations={citations} activeChunkId={activeChunkId} onOpen={onOpen} />
  );
  const none = <span className="muted">Not found</span>;
  const { start_date: start, end_date: end, supplier_name: supplier, rates } = terms;

  return (
    <dl className="terms">
      <Row label="Supplier" hint="Party supplying the goods, services or licence">
        {supplier.value ? (
          <>
            <span className="term-value">{supplier.value}</span> {src(supplier.citations)}
          </>
        ) : (
          none
        )}
      </Row>
      <Row label="Start date" hint="Effective / commencement date">
        {start.value ? (
          <>
            <span className="term-value">{formatDay(start.value)}</span> {src(start.citations)}
          </>
        ) : (
          none
        )}
      </Row>
      <Row label="End date" hint="Expiry / termination date">
        {end.value ? (
          <>
            <span className="term-value">{formatDay(end.value)}</span> {src(end.citations)}
            {end.derived_from && <div className="muted small">{end.derived_from}</div>}
          </>
        ) : (
          <>
            {none}
            <div className="muted small">No end date or term length is stated.</div>
          </>
        )}
      </Row>
      <Row label="Rates" hint="Prices, fees and rates">
        {rates.length ? (
          <ul className="rates">
            {rates.map((r, i) => (
              <li key={i}>
                <span className="term-value">{r.amount}</span>
                {r.unit && <span className="muted"> {r.unit}</span>} {src(r.citations)}
                <div className="muted small">{r.description}</div>
              </li>
            ))}
          </ul>
        ) : (
          none
        )}
      </Row>
    </dl>
  );
}

function Card({
  contract,
  activeChunkId,
  onOpenCitation,
}: {
  contract: Contract;
  activeChunkId: number | null;
  onOpenCitation: (c: Citation) => void;
}) {
  const query = useExtraction(contract.id);
  const refresh = useRefreshExtraction(contract.id);
  const data = query.data;
  const busy = query.isFetching || refresh.isPending;
  const error = query.error?.message ?? refresh.error?.message ?? (data?.status === 'failed' ? data.error : null);

  return (
    <section className="answer-col">
      <header>
        <span className="doc-icon" aria-hidden>
          📄
        </span>
        <span className="answer-title" title={contract.filename}>
          {contract.filename}
        </span>
        <button
          className="icon-btn"
          title="Re-run extraction"
          aria-label={`Re-run extraction for ${contract.filename}`}
          disabled={busy}
          onClick={() => refresh.mutate()}
        >
          ↻
        </button>
      </header>
      {busy && !data?.terms ? (
        <p className="muted typing">Extracting terms…</p>
      ) : (
        <>
          {error && (
            <div className="banner error">
              {error}{' '}
              <button disabled={busy} onClick={() => refresh.mutate()}>
                Retry
              </button>
            </div>
          )}
          {data?.terms && (
            <Terms terms={data.terms} activeChunkId={activeChunkId} onOpen={onOpenCitation} />
          )}
          {data?.terms && (
            <div className="muted small">
              Extracted by {data.provider}
              {data.provider === 'mock' && ' (rule-based; add an API key for LLM extraction)'}
              {busy && ' · refreshing…'}
            </div>
          )}
        </>
      )}
    </section>
  );
}

export function ExtractionPanel({ contracts, selectedIds, activeChunkId, onOpenCitation }: Props) {
  const selected = contracts.filter((c) => selectedIds.includes(c.id));

  if (!selected.length) {
    return (
      <div className="chat-scroll">
        <div className="empty">
          <h2>Extracted terms</h2>
          <p className="muted">
            Select one or more ready contracts on the left to see their supplier, start and end
            dates and rates, each linked to the passage it was read from.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-scroll">
      <div className="answer-grid">
        {selected.map((c) => (
          <Card
            key={c.id}
            contract={c}
            activeChunkId={activeChunkId}
            onOpenCitation={onOpenCitation}
          />
        ))}
      </div>
    </div>
  );
}
