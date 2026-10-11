import type { Citation } from '../types';
import { formatLocation } from '../utils';

interface Props {
  citation: Citation;
  active?: boolean;
  onOpen: (c: Citation) => void;
}

export function CitationCard({ citation, active, onOpen }: Props) {
  return (
    <button
      className={`citation${active ? ' active' : ''}${citation.cited ? '' : ' uncited'}`}
      onClick={() => onOpen(citation)}
      title="Open in source viewer"
    >
      <div className="citation-head">
        <span className="chip">{citation.tag}</span>
        <span className="citation-loc">{formatLocation(citation)}</span>
        {citation.score !== null && (
          <span className="muted small">match {(citation.score * 100).toFixed(0)}%</span>
        )}
      </div>
      <blockquote>{citation.snippet}</blockquote>
      {!citation.cited && (
        <div className="muted small">Closest matching passage (not explicitly cited)</div>
      )}
    </button>
  );
}
