/**
 * Thin fetch client for the Lightspeed Stack REST API.
 *
 * Requests go to `/v1/*`, which `vite.config.ts` proxies to the backend
 * (default `http://localhost:8080`) during local development. In
 * production, serve this build behind the same origin as the API, or set
 * `VITE_API_BASE_URL`.
 */

import { ApiError, type ModelsResponse, type QueryRequest, type QueryResponse, type StreamEvent } from './types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '';

async function parseErrorResponse(response: Response): Promise<never> {
  let detail = { response: response.statusText, cause: `HTTP ${response.status}` };
  try {
    const body = await response.json();
    if (body?.detail) {
      detail = body.detail;
    }
  } catch {
    // Body wasn't JSON; fall back to the default detail above.
  }
  throw new ApiError(response.status, detail);
}

export async function listModels(): Promise<ModelsResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/models`);
  if (!response.ok) {
    await parseErrorResponse(response);
  }
  return response.json();
}

/** Send a non-streaming query and wait for the full response. */
export async function sendQuery(request: QueryRequest): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...request, stream: false }),
  });
  if (!response.ok) {
    await parseErrorResponse(response);
  }
  return response.json();
}

/**
 * Send a streaming query and yield each decoded SSE payload as it arrives.
 *
 * The backend writes plain `data: <json>\n\n` lines (see
 * `StreamPayloadBase.serialize_json`); the `event` discriminator lives
 * inside the JSON body rather than in an SSE `event:` line, so this only
 * needs to split on blank-line-delimited chunks and strip the `data: `
 * prefix.
 */
export async function* streamQuery(
  request: QueryRequest,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${API_BASE_URL}/v1/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...request, stream: true }),
    signal,
  });
  if (!response.ok || !response.body) {
    await parseErrorResponse(response);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let separatorIndex: number;
      // eslint-disable-next-line no-cond-assign
      while ((separatorIndex = buffer.indexOf('\n\n')) !== -1) {
        const chunk = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);
        const line = chunk.startsWith('data:') ? chunk.slice(5).trim() : chunk.trim();
        if (!line) continue;
        yield JSON.parse(line) as StreamEvent;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
