// Mirrors backend/app/schemas.py

export type ContractStatus = 'processing' | 'ready' | 'failed';

export interface Contract {
  id: string;
  filename: string;
  size_bytes: number;
  page_count: number;
  chunk_count: number;
  status: ContractStatus;
  error: string | null;
  uploaded_at: string;
}

export interface UploadResult {
  filename: string;
  status: 'accepted' | 'duplicate' | 'rejected';
  contract: Contract | null;
  error: { code: string; message: string } | null;
}

export interface LineRange {
  page: number;
  line_start: number;
  line_end: number;
}

export interface Highlight {
  page: number;
  bboxes: number[][]; // normalised [x0, y0, x1, y1]
}

export interface Citation {
  tag: string;
  chunk_id: number;
  contract_id: string;
  contract_name: string;
  page_start: number;
  page_end: number;
  line_start: number;
  line_end: number;
  line_ranges: LineRange[];
  snippet: string;
  score: number | null; // null for overview answers (no query relevance)
  cited: boolean;
  highlights: Highlight[];
}

export interface ContractAnswer {
  contract_id: string;
  contract_name: string;
  answer: string;
  not_found: boolean;
  citations: Citation[];
  error: string | null;
}

export interface QueryResponse {
  conversation_id: string;
  message_id: number;
  question: string;
  results: ContractAnswer[];
  comparison: string | null;
  provider: string;
}

/** Shape stored in Message.results for assistant messages. */
export type AssistantPayload = Omit<QueryResponse, 'message_id'>;

export interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  contract_ids: string[];
  results: AssistantPayload | null;
  created_at: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: Message[];
}

export interface PageLine {
  line_number: number;
  text: string;
  bbox: number[];
}

export interface PageData {
  contract_id: string;
  page_number: number;
  page_count: number;
  lines: PageLine[];
}

export interface Health {
  status: 'ok';
  llm_provider: string;
  embeddings: string;
  retrieval: string;
}

// ---- Extraction (mirrors ContractTerms / ExtractionOut) ----

export interface DateField {
  value: string | null; // ISO date (YYYY-MM-DD)
  derived_from: string | null; // set when calculated rather than quoted
  citations: Citation[];
}

export interface TextField {
  value: string | null;
  citations: Citation[];
}

export interface RateItem {
  description: string;
  amount: string;
  unit: string | null;
  citations: Citation[];
}

export interface ContractTerms {
  start_date: DateField;
  end_date: DateField;
  supplier_name: TextField;
  rates: RateItem[];
}

export interface Extraction {
  contract_id: string;
  contract_name: string;
  status: 'ready' | 'failed';
  provider: string;
  extracted_at: string | null;
  error: string | null;
  terms: ContractTerms | null;
}
