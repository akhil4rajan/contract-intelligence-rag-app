import type { Contract } from '../types';
import { formatBytes, formatDate } from '../utils';

interface Props {
  contracts: Contract[];
  loading: boolean;
  error: string | null;
  selected: string[];
  onToggle: (id: string) => void;
  onSelectAll: (ids: string[]) => void;
  onDelete: (c: Contract) => void;
}

export function ContractList({
  contracts,
  loading,
  error,
  selected,
  onToggle,
  onSelectAll,
  onDelete,
}: Props) {
  const readyIds = contracts.filter((c) => c.status === 'ready').map((c) => c.id);
  const allSelected = readyIds.length > 0 && readyIds.every((id) => selected.includes(id));

  if (loading) return <p className="muted pad">Loading contracts…</p>;
  if (error) return <p className="error pad">{error}</p>;
  if (!contracts.length)
    return <p className="muted pad">No contracts yet. Upload a PDF to get started.</p>;

  return (
    <div>
      <div className="list-head">
        <label>
          <input
            type="checkbox"
            checked={allSelected}
            onChange={() => onSelectAll(allSelected ? [] : readyIds)}
          />{' '}
          Select all ready
        </label>
        <span className="muted">{selected.length} selected</span>
      </div>
      <ul className="contracts">
        {contracts.map((c) => {
          const ready = c.status === 'ready';
          return (
            <li key={c.id} className={`contract${selected.includes(c.id) ? ' selected' : ''}`}>
              <input
                type="checkbox"
                aria-label={`Select ${c.filename}`}
                disabled={!ready}
                checked={selected.includes(c.id)}
                onChange={() => onToggle(c.id)}
              />
              <div className="contract-body">
                <div className="contract-name" title={c.filename}>
                  {c.filename}
                </div>
                <div className="muted small">
                  {formatDate(c.uploaded_at)} · {c.page_count} pp · {formatBytes(c.size_bytes)}
                </div>
                {c.status === 'failed' && <div className="error small">{c.error}</div>}
              </div>
              <span className={`badge ${c.status}`}>{c.status}</span>
              <button
                className="icon-btn"
                title="Delete contract"
                aria-label={`Delete ${c.filename}`}
                onClick={() => onDelete(c)}
              >
                ✕
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
