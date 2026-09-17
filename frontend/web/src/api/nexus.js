/**
 * Thin fetch client for NEXUS (the LLM coach service, nexus/api/main.py).
 *
 * Kept separate from src/api/client.js on purpose. NEXUS is optional: the
 * core football backend on :8000 works perfectly well with NEXUS down, and
 * only the Coach Chat tab depends on it. Sharing one client would mean one
 * "backend is down" message for two services that fail independently, so
 * this module carries its own.
 */

import { ApiError, authHeaders, formatDetail, requestWith } from './client.js';

// IPv4 literal for the same reason as client.js's DEFAULT_BASE -- see the
// comment there. Override with VITE_NEXUS_BASE_URL.
const DEFAULT_BASE = 'http://127.0.0.1:8100';

export const NEXUS_BASE = String(
  import.meta.env?.VITE_NEXUS_BASE_URL || DEFAULT_BASE,
).replace(/\/+$/, '');

// Only needed when NEXUS runs with NEXUS_API_KEY set. Public once bundled; see
// the note on VITE_SSC_API_KEY in client.js.
const NEXUS_API_KEY = String(import.meta.env?.VITE_NEXUS_API_KEY || '');

export const NEXUS_UNREACHABLE =
  `Coach Chat is unreachable — confirm NEXUS is running on port 8100 (${NEXUS_BASE}). ` +
  'This service is separate from the core backend on :8000, which can be up while ' +
  'this one is down. If it is running, add this origin to NEXUS_CORS_ALLOWED_ORIGINS.';

/** The exact RoutingPolicy values from nexus/core/types.py, with their labels. */
export const ROUTING_POLICIES = [
  { value: 'BALANCED', label: 'Balanced (Default)' },
  { value: 'LOW_COST', label: 'LOW_COST' },
  { value: 'LOW_LATENCY', label: 'LOW_LATENCY' },
  { value: 'MAX_QUALITY', label: 'MAX_QUALITY' },
  { value: 'LOCAL_ONLY', label: 'Local Models Only' },
];

function request(path, options) {
  return requestWith(NEXUS_BASE, path, {
    ...options,
    headers: { ...authHeaders(NEXUS_API_KEY), ...options?.headers },
    unreachableMessage: NEXUS_UNREACHABLE,
    service: 'nexus',
  });
}

/**
 * Streaming chat over SSE.
 *
 * Not built on requestWith(): that helper resolves the whole body as JSON,
 * which is exactly what must NOT happen for a stream. The error handling is
 * mirrored here instead so a dead NEXUS still produces the same
 * NEXUS_UNREACHABLE message it would on any other call.
 *
 * Two distinct failure modes are covered:
 *   1. the request never lands (service down / CORS)  -> ApiError, status 0
 *   2. the stream opens and then the provider fails   -> the server yields
 *      {"error": "...", "done": true} as a data frame (see
 *      nexus/api/routes/chat.py::_stream_response), surfaced via onError
 */
export async function streamChat(body, { signal, onDelta, onDone, onError } = {}) {
  let response;
  try {
    response = await fetch(`${NEXUS_BASE}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
        ...authHeaders(NEXUS_API_KEY),
      },
      body: JSON.stringify({ ...body, stream: true }),
      signal,
    });
  } catch (error) {
    if (error?.name === 'AbortError') throw error;
    throw new ApiError(NEXUS_UNREACHABLE, { status: 0, service: 'nexus' });
  }

  if (!response.ok) {
    let detail = null;
    try {
      const payload = await response.json();
      detail = payload?.detail ?? payload;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(
      formatDetail(detail) || `NEXUS chat failed: ${response.status} ${response.statusText}`,
      { status: response.status, detail, service: 'nexus' },
    );
  }

  if (!response.body) {
    throw new ApiError('NEXUS returned no response stream.', {
      status: response.status,
      service: 'nexus',
    });
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const handleFrame = (frame) => {
    for (const line of frame.split('\n')) {
      if (!line.startsWith('data:')) continue;
      const raw = line.slice(5).trim();
      if (!raw || raw === '[DONE]') continue;
      let event;
      try {
        event = JSON.parse(raw);
      } catch {
        continue;
      }
      if (event.error) {
        onError?.(event.error);
        continue;
      }
      if (event.delta) onDelta?.(event.delta);
      if (event.done) onDone?.(event);
    }
  };

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split('\n\n');
      buffer = frames.pop() ?? '';
      frames.forEach(handleFrame);
    }
    if (buffer.trim()) handleFrame(buffer);
  } finally {
    // Abort mid-stream leaves the reader open otherwise; releasing it lets
    // the connection tear down instead of hanging until the server gives up.
    try {
      reader.releaseLock();
    } catch {
      /* already released */
    }
  }
}

export const nexus = {
  health: (signal) => request('/api/health', { signal }),

  /** Non-streaming chat. Response shape: nexus/api/schemas.py::ChatResponse. */
  chat: (body, signal) =>
    request('/api/chat', { method: 'POST', json: { ...body, stream: false }, signal }),

  streamChat,

  // The LLM coach report for a processed match: tactical read, game plan,
  // and in-game adjustments. Everything but `narrative` is derived
  // deterministically upstream, so the UI can render it as measured fact.
  getCoachReport: (matchId, signal) =>
    request(`/api/sports/${encodeURIComponent(matchId)}/report`, { signal }),

  getPlayerCoachReport: (matchId, playerId, signal) =>
    request(
      `/api/sports/${encodeURIComponent(matchId)}/player/${encodeURIComponent(playerId)}`,
      { signal },
    ),

  /**
   * GET /api/sports/psychology/{player_id}/report
   *
   * Every number in the response is copied verbatim from the football
   * backend's psychology scoring (see nexus/api/routes/psychology.py) --
   * NEXUS writes only `narrative`. The UI renders those two provenances
   * separately rather than as one block, so a reader can tell computed from
   * generated.
   */
  getPsychologyReport: (playerId, matchId, signal) => {
    const params = new URLSearchParams();
    if (matchId) params.set('match_id', matchId);
    const query = params.toString();
    return request(
      `/api/sports/psychology/${encodeURIComponent(playerId)}/report${query ? `?${query}` : ''}`,
      { signal },
    );
  },
};

export default nexus;
