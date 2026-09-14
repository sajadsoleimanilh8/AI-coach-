                                                                                                                                                                                                                                                                                                                                                                                                                                  

import { ApiError, formatDetail, requestWith } from './client.js';

// IPv4 literal for the same reason as client.js's DEFAULT_BASE -- see the
// comment there. Override with VITE_NEXUS_BASE_URL.
const DEFAULT_BASE = 'http://127.0.0.1:8100';

export const NEXUS_BASE = String(
  import.meta.env?.VITE_NEXUS_BASE_URL || DEFAULT_BASE,
).replace(/\/+$/, '');

export const NEXUS_UNREACHABLE =
  `Coach Chat is unreachable — confirm NEXUS is running on port 8100 (${NEXUS_BASE}). ` +
  'This service is separate from the core backend on :8000, which can be up while ' +
  'this one is down. If it is running, add this origin to NEXUS_CORS_ALLOWED_ORIGINS.';

                                                                                  
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
    unreachableMessage: NEXUS_UNREACHABLE,
    service: 'nexus',
  });
}

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
export async function streamChat(body, { signal, onDelta, onDone, onError } = {}) {
  let response;
  try {
    response = await fetch(`${NEXUS_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
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
                                                                           
                                                                             
    try {
      reader.releaseLock();
    } catch {
                            
    }
  }
}

export const nexus = {
  health: (signal) => request('/api/health', { signal }),

                                                                                
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
