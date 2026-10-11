import { useEffect, useState, type CSSProperties } from 'react';
import { ChatPanel } from './components/ChatPanel';
import { ContractList } from './components/ContractList';
import { ExtractionPanel } from './components/ExtractionPanel';
import { HistoryList } from './components/HistoryList';
import { PaneResizer } from './components/PaneResizer';
import { SourceViewer } from './components/SourceViewer';
import { UploadDropzone } from './components/UploadDropzone';
import { useContracts, useDeleteContract, useHealth } from './hooks/queries';
import type { Citation, Contract } from './types';

const VIEWER_WIDTH_KEY = 'viewer.width';
const VIEWER_MIN_WIDTH = 340;

function loadViewerWidth(): number | null {
  try {
    const n = Number(localStorage.getItem(VIEWER_WIDTH_KEY));
    return Number.isFinite(n) && n >= VIEWER_MIN_WIDTH ? n : null;
  } catch {
    return null;
  }
}

export default function App() {
  const contracts = useContracts();
  const health = useHealth();
  const deleteContract = useDeleteContract();

  const [selected, setSelected] = useState<string[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [citation, setCitation] = useState<Citation | null>(null);
  const [tab, setTab] = useState<'contracts' | 'history'>('contracts');
  const [view, setView] = useState<'chat' | 'terms'>('chat');
  const [actionError, setActionError] = useState<string | null>(null);
  const [viewerExpanded, setViewerExpanded] = useState(false);
  const [viewerWidth, setViewerWidth] = useState<number | null>(loadViewerWidth);

  const list = contracts.data ?? [];

  useEffect(() => {
    try {
      if (viewerWidth == null) localStorage.removeItem(VIEWER_WIDTH_KEY);
      else localStorage.setItem(VIEWER_WIDTH_KEY, String(viewerWidth));
    } catch {
      /* storage unavailable */
    }
  }, [viewerWidth]);

  // Drop selections for contracts that no longer exist or are not ready.
  useEffect(() => {
    if (!contracts.data) return;
    const ready = new Set(contracts.data.filter((c) => c.status === 'ready').map((c) => c.id));
    setSelected((s) => (s.every((id) => ready.has(id)) ? s : s.filter((id) => ready.has(id))));
  }, [contracts.data]);

  const toggle = (id: string) =>
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));

  const remove = (c: Contract) => {
    if (!window.confirm(`Delete "${c.filename}" and its indexed content?`)) return;
    setActionError(null);
    deleteContract.mutate(c.id, {
      onSuccess: () => {
        setSelected((s) => s.filter((id) => id !== c.id));
        setCitation((cur) => (cur?.contract_id === c.id ? null : cur));
      },
      onError: (e) => setActionError(e.message),
    });
  };

  const newChat = () => {
    setConversationId(null);
    setCitation(null);
  };

  return (
    <div className="app">
      <header className="topbar">
        <h1>Contract Intelligence</h1>
        <div className="topbar-right">
          {health.data && (
            <span className="pill" title={`Embeddings: ${health.data.embeddings}`}>
              LLM: {health.data.llm_provider} · {health.data.retrieval}
            </span>
          )}
          {health.isError && <span className="pill error">Backend offline</span>}
          <button onClick={newChat}>New chat</button>
        </div>
      </header>

      <main
        className={`layout${viewerExpanded && citation ? ' expanded' : ''}`}
        style={viewerWidth ? ({ '--viewer-w': `${viewerWidth}px` } as CSSProperties) : undefined}
      >
        <aside className="pane left">
          <UploadDropzone />
          <div className="tabs">
            <button className={tab === 'contracts' ? 'on' : ''} onClick={() => setTab('contracts')}>
              Contracts{list.length ? ` (${list.length})` : ''}
            </button>
            <button className={tab === 'history' ? 'on' : ''} onClick={() => setTab('history')}>
              History
            </button>
          </div>
          {actionError && <div className="banner error">{actionError}</div>}
          <div className="pane-scroll">
            {tab === 'contracts' ? (
              <ContractList
                contracts={list}
                loading={contracts.isLoading}
                error={contracts.error?.message ?? null}
                selected={selected}
                onToggle={toggle}
                onSelectAll={setSelected}
                onDelete={remove}
              />
            ) : (
              <HistoryList
                activeId={conversationId}
                onSelect={(id) => {
                  setConversationId(id);
                  setCitation(null);
                  setTab('contracts');
                }}
                onDeleted={(id) => id === conversationId && newChat()}
              />
            )}
          </div>
        </aside>

        <section className="pane center">
          <div className="tabs">
            <button className={view === 'chat' ? 'on' : ''} onClick={() => setView('chat')}>
              Chat
            </button>
            <button className={view === 'terms' ? 'on' : ''} onClick={() => setView('terms')}>
              Extracted terms
            </button>
          </div>
          {/* Both stay mounted so an unsent draft or a pending answer survives a tab switch. */}
          <div className="center-body" hidden={view !== 'chat'}>
            <ChatPanel
              contracts={list}
              selectedIds={selected}
              conversationId={conversationId}
              activeChunkId={citation?.chunk_id ?? null}
              onConversationChange={setConversationId}
              onOpenCitation={setCitation}
            />
          </div>
          <div className="center-body" hidden={view !== 'terms'}>
            <ExtractionPanel
              contracts={list}
              selectedIds={selected}
              activeChunkId={citation?.chunk_id ?? null}
              onOpenCitation={setCitation}
            />
          </div>
        </section>

        <aside className="pane right">
          {!viewerExpanded && (
            <PaneResizer
              min={VIEWER_MIN_WIDTH}
              max={Math.max(VIEWER_MIN_WIDTH, window.innerWidth - 300 - 380 - 48)}
              onChange={setViewerWidth}
              onReset={() => setViewerWidth(null)}
            />
          )}
          <SourceViewer
            citation={citation}
            expanded={viewerExpanded}
            onToggleExpand={() => setViewerExpanded((v) => !v)}
            onClose={() => {
              setCitation(null);
              setViewerExpanded(false);
            }}
          />
        </aside>
      </main>
    </div>
  );
}
