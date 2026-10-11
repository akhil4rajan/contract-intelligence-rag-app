import type { AssistantPayload, Citation, ContractAnswer } from '../types';
import { CitationCard } from './CitationCard';

interface Props {
  payload: AssistantPayload;
  activeChunkId: number | null;
  onOpenCitation: (c: Citation) => void;
}

const TAG_SPLIT = /(\[C\d+\])/g;

function AnswerText({
  answer,
  citations,
  onOpen,
}: {
  answer: string;
  citations: Citation[];
  onOpen: (c: Citation) => void;
}) {
  const byTag = new Map(citations.map((c) => [c.tag, c]));
  return (
    <div className="answer-text">
      {answer.split('\n').map((line, i) => (
        <p key={i}>
          {line.split(TAG_SPLIT).map((part, j) => {
            const m = /^\[(C\d+)\]$/.exec(part);
            const cit = m ? byTag.get(m[1]) : undefined;
            if (m && cit)
              return (
                <button
                  key={j}
                  className="chip chip-btn"
                  onClick={() => onOpen(cit)}
                  title={`Page ${cit.page_start}, lines ${cit.line_start}–${cit.line_end}`}
                >
                  {m[1]}
                </button>
              );
            return <span key={j}>{part}</span>;
          })}
        </p>
      ))}
    </div>
  );
}

function Column({
  result,
  activeChunkId,
  onOpenCitation,
}: {
  result: ContractAnswer;
  activeChunkId: number | null;
  onOpenCitation: (c: Citation) => void;
}) {
  return (
    <section className="answer-col">
      <header>
        <span className="doc-icon" aria-hidden>
          📄
        </span>
        <span className="answer-title" title={result.contract_name}>
          {result.contract_name}
        </span>
      </header>
      {result.error ? (
        <p className="error">{result.error}</p>
      ) : result.not_found ? (
        <>
          <p className="muted">{result.answer}</p>
          <p className="muted small">
            Nothing in this document was close enough to your question. Try naming specific content
            in it, or ask &ldquo;What is this document about?&rdquo;
          </p>
        </>
      ) : (
        <AnswerText answer={result.answer} citations={result.citations} onOpen={onOpenCitation} />
      )}
      {result.citations.length > 0 && (
        <div className="citations">
          <div className="citations-title">Sources</div>
          {result.citations.map((c) => (
            <CitationCard
              key={c.chunk_id}
              citation={c}
              active={c.chunk_id === activeChunkId}
              onOpen={onOpenCitation}
            />
          ))}
        </div>
      )}
    </section>
  );
}

export function AnswerGrid({ payload, activeChunkId, onOpenCitation }: Props) {
  return (
    <div className="answer">
      {payload.comparison && (
        <div className="comparison">
          <strong>Comparison.</strong> {payload.comparison}
        </div>
      )}
      <div className="answer-grid">
        {payload.results.map((r) => (
          <Column
            key={r.contract_id}
            result={r}
            activeChunkId={activeChunkId}
            onOpenCitation={onOpenCitation}
          />
        ))}
      </div>
    </div>
  );
}
