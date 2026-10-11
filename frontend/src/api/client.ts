import type {
  Contract,
  Conversation,
  ConversationSummary,
  Extraction,
  Health,
  PageData,
  QueryResponse,
  UploadResult,
} from '../types';

const BASE = '/api';

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code: string,
    public details: Record<string, unknown> = {},
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError('Cannot reach the server. Is the backend running?', 0, 'network_error');
  }
  if (!res.ok) {
    let body: { error?: { code: string; message: string; details?: Record<string, unknown> } } = {};
    try {
      body = await res.json();
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(
      body.error?.message ?? `Request failed (${res.status})`,
      res.status,
      body.error?.code ?? 'http_error',
      body.error?.details,
    );
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>('/health'),

  listContracts: () => request<Contract[]>('/contracts'),
  uploadContracts: async (files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append('files', f));
    const res = await request<{ results: UploadResult[] }>('/contracts', {
      method: 'POST',
      body: form,
    });
    return res.results;
  },
  deleteContract: (id: string) => request<void>(`/contracts/${id}`, { method: 'DELETE' }),
  getPage: (id: string, page: number) => request<PageData>(`/contracts/${id}/pages/${page}`),
  fileUrl: (id: string) => `${BASE}/contracts/${id}/file`,
  getExtraction: (id: string) => request<Extraction>(`/contracts/${id}/extraction`),
  refreshExtraction: (id: string) =>
    request<Extraction>(`/contracts/${id}/extraction`, { method: 'POST' }),

  query: (body: {
    question: string;
    contract_ids: string[];
    conversation_id?: string | null;
    top_k?: number;
  }) => request<QueryResponse>('/chat/query', json(body)),

  listConversations: () => request<ConversationSummary[]>('/conversations'),
  getConversation: (id: string) => request<Conversation>(`/conversations/${id}`),
  deleteConversation: (id: string) => request<void>(`/conversations/${id}`, { method: 'DELETE' }),
};
