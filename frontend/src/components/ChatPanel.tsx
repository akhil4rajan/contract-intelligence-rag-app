import { useEffect, useRef, useState } from 'react';
import { useAsk, useConversation } from '../hooks/queries';
import type { Citation, Contract } from '../types';
import { AnswerGrid } from './AnswerGrid';

interface Props {
  contracts: Contract[];
  selectedIds: string[];
  conversationId: string | null;
  activeChunkId: number | null;
  onConversationChange: (id: string | null) => void;
  onOpenCitation: (c: Citation) => void;
}

const SUGGESTIONS = [
  'What is the termination notice period?',
  'Who are the parties to this agreement?',
  'What are the payment terms?',
  'Is there a limitation of liability clause?',
];

export function ChatPanel({
  contracts,
  selectedIds,
  conversationId,
  activeChunkId,
  onConversationChange,
  onOpenCitation,
}: Props) {
  const [text, setText] = useState('');
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const conv = useConversation(conversationId);
  const ask = useAsk();

  const names = new Map(contracts.map((c) => [c.id, c.filename]));
  const messages = conv.data?.messages ?? [];

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages.length, pending]);

  const send = (question: string) => {
    const q = question.trim();
    if (!q || ask.isPending || !selectedIds.length) return;
    setError(null);
    setPending(q);
    setText('');
    ask.mutate(
      { question: q, contract_ids: selectedIds, conversation_id: conversationId },
      {
        onSuccess: (res) => {
          onConversationChange(res.conversation_id);
          setPending(null);
        },
        onError: (err) => {
          setPending(null);
          setText(q);
          setError(err.message);
        },
      },
    );
  };

  const canSend = !!text.trim() && selectedIds.length > 0 && !ask.isPending;

  return (
    <div className="chat">
      <div className="chat-scroll">
        {messages.length === 0 && !pending && (
          <div className="empty">
            <h2>Ask about your contracts</h2>
            <p className="muted">
              {selectedIds.length
                ? `Querying ${selectedIds.length} contract${selectedIds.length > 1 ? 's' : ''}. Answers are grounded in the source text with page and line references.`
                : 'Select one or more ready contracts on the left, then ask a question.'}
            </p>
            {selectedIds.length > 0 && (
              <div className="suggestions">
                {SUGGESTIONS.map((s) => (
                  <button key={s} className="suggestion" onClick={() => send(s)}>
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {messages.map((m) =>
          m.role === 'user' ? (
            <div key={m.id} className="msg user">
              <div className="bubble">{m.content}</div>
              <div className="muted small msg-meta">
                {m.contract_ids.map((id) => names.get(id) ?? 'deleted contract').join(' · ')}
              </div>
            </div>
          ) : (
            <div key={m.id} className="msg assistant">
              {m.results ? (
                <AnswerGrid
                  payload={m.results}
                  activeChunkId={activeChunkId}
                  onOpenCitation={onOpenCitation}
                />
              ) : (
                <div className="bubble">{m.content}</div>
              )}
            </div>
          ),
        )}

        {pending && (
          <>
            <div className="msg user">
              <div className="bubble">{pending}</div>
            </div>
            <div className="msg assistant">
              <div className="bubble typing">Searching contracts and drafting an answer…</div>
            </div>
          </>
        )}
        {error && <div className="banner error">{error}</div>}
        <div ref={bottomRef} />
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send(text);
        }}
      >
        <textarea
          value={text}
          rows={2}
          maxLength={2000}
          placeholder={
            selectedIds.length ? 'Ask a question about the selected contracts…' : 'Select a contract first'
          }
          disabled={!selectedIds.length}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              send(text);
            }
          }}
        />
        <button type="submit" disabled={!canSend}>
          {ask.isPending ? '…' : 'Ask'}
        </button>
      </form>
    </div>
  );
}
