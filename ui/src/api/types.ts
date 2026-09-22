/**
 * Types mirroring the Lightspeed Stack REST API shapes.
 *
 * Kept in sync by hand with:
 *  - lightspeed/app/models/requests/query.py     (QueryRequest)
 *  - lightspeed/app/models/responses/success/query.py   (QueryResponse)
 *  - lightspeed/app/models/responses/success/stream.py  (SSE payloads)
 *  - lightspeed/app/models/responses/success/models.py  (ModelsResponse)
 */

export interface QueryRequest {
  query: string;
  conversation_id?: string;
  provider?: string;
  model?: string;
  system_prompt?: string;
  stream?: boolean;
}

export interface QueryResponse {
  conversation_id: string | null;
  response: string;
  rag_chunks: unknown[];
  referenced_documents: unknown[];
  truncated: boolean;
  input_tokens: number;
  output_tokens: number;
  available_quotas: Record<string, number>;
  tool_calls: unknown[];
  tool_results: unknown[];
}

export interface ModelInfo {
  identifier: string;
  provider_name: string;
  model_name: string;
}

export interface ModelsResponse {
  models: ModelInfo[];
}

// --- Server-sent event payloads for streaming `/v1/query` ---

export interface StartStreamEvent {
  event: 'start';
  data: { conversation_id: string };
}

export interface TokenStreamEvent {
  event: 'token';
  data: { id: number; token: string };
}

export interface TurnCompleteStreamEvent {
  event: 'turn_complete';
  data: { id: number; token: string };
}

export interface EndStreamEvent {
  event: 'end';
  data: { input_tokens: number; output_tokens: number };
}

export interface ErrorStreamEvent {
  event: 'error';
  data: { status_code: number; response: string; cause: string };
}

export type StreamEvent =
  | StartStreamEvent
  | TokenStreamEvent
  | TurnCompleteStreamEvent
  | EndStreamEvent
  | ErrorStreamEvent;

export interface ApiErrorDetail {
  response: string;
  cause: string;
}

export class ApiError extends Error {
  status: number;
  cause_?: string;

  constructor(status: number, detail: ApiErrorDetail) {
    super(detail.response);
    this.status = status;
    this.cause_ = detail.cause;
  }
}
