import { useConversations, useDeleteConversation } from '../hooks/queries';
import { formatDate } from '../utils';

interface Props {
  activeId: string | null;
  onSelect: (id: string) => void;
  onDeleted: (id: string) => void;
}

export function HistoryList({ activeId, onSelect, onDeleted }: Props) {
  const { data, isLoading, error } = useConversations();
  const del = useDeleteConversation();

  if (isLoading) return <p className="muted pad">Loading history…</p>;
  if (error) return <p className="error pad">{error.message}</p>;
  if (!data?.length) return <p className="muted pad">No conversations yet.</p>;

  return (
    <ul className="history">
      {data.map((c) => (
        <li key={c.id} className={c.id === activeId ? 'active' : ''}>
          <button className="history-main" onClick={() => onSelect(c.id)}>
            <span className="history-title">{c.title}</span>
            <span className="muted small">
              {Math.ceil(c.message_count / 2)} Q · {formatDate(c.updated_at)}
            </span>
          </button>
          <button
            className="icon-btn"
            aria-label="Delete conversation"
            onClick={() =>
              del.mutate(c.id, {
                onSuccess: () => onDeleted(c.id),
              })
            }
          >
            ✕
          </button>
        </li>
      ))}
    </ul>
  );
}
