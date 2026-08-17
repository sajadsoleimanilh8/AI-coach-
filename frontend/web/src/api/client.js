                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           

// IPv4 literal, not the name 'localhost'. On Windows, Node/Chrome resolve
// 'localhost' to ::1 before 127.0.0.1, but uvicorn binds IPv4 only -- so every
// request would spend ~2s failing against ::1 before falling back, and when the
// browser gives up instead, the UI reports CORE_UNREACHABLE (which blames CORS).
// Override with VITE_API_BASE_URL when deploying anywhere but a dev machine.
const DEFAULT_BASE = 'http://127.0.0.1:8000';

export const API_BASE = String(
  import.meta.env?.VITE_API_BASE_URL || DEFAULT_BASE,
).replace(/\/+$/, '');

export const CORE_UNREACHABLE =
  `Core backend unreachable at ${API_BASE} — confirm the FastAPI server is running ` +
  '(uvicorn backend.api.main:app --port 8000) and that this origin is listed in ' +
  'CORS_ALLOWED_ORIGINS in backend/.env.';

export class ApiError extends Error {
  constructor(message, { status = 0, detail = null, service = 'core' } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
    this.service = service;
                                                                                     
    this.unreachable = status === 0;
    this.notFound = status === 404;
  }
}

                                                                                                                                                                                                                                                                                                                                                    
export function formatDetail(detail) {
  if (detail === null || detail === undefined) return null;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) => {
        if (typeof item === 'string') return item;
        if (!item || typeof item !== 'object') return null;
        const loc = Array.isArray(item.loc)
          ? item.loc.filter((segment) => segment !== 'body').join('.')
          : '';
        const msg = item.msg || item.type || '';
        if (!msg) return null;
        return loc ? `${loc}: ${msg}` : msg;
      })
      .filter(Boolean);
    return parts.length ? parts.join(' · ') : null;
  }
  if (typeof detail === 'object' && typeof detail.msg === 'string') return detail.msg;
  try {
    return JSON.stringify(detail);
  } catch {
    return null;
  }
}

                                                                                                                                                           
export async function requestWith(
  base,
  path,
  {
    method = 'GET',
    body,
    json,
    signal,
    headers,
    unreachableMessage = CORE_UNREACHABLE,
    service = 'core',
  } = {},
) {
  const init = { method, signal, headers: { Accept: 'application/json', ...headers } };

  if (json !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(json);
  } else if (body !== undefined) {
                                                                       
                                               
    init.body = body;
  }

  let response;
  try {
    response = await fetch(`${base}${path}`, init);
  } catch (error) {
                                                                     
                                                                       
                                                               
    if (error?.name === 'AbortError') throw error;
    throw new ApiError(unreachableMessage, { status: 0, service });
  }

  if (!response.ok) {
    let detail = null;
    try {
      const payload = await response.json();
      detail = payload?.detail ?? payload;
    } catch {
                                                                        
    }
    const message =
      formatDetail(detail) || `Request failed: ${response.status} ${response.statusText}`;
    throw new ApiError(message, { status: response.status, detail, service });
  }

  if (response.status === 204) return null;
  const contentType = response.headers.get('content-type') || '';
  if (!contentType.includes('application/json')) return response.text();
  return response.json();
}

function request(path, options) {
  return requestWith(API_BASE, path, options);
}

export const api = {
                                                                             
  health: (signal) => request('/health', { signal }),

                                                                            

                                                                                                                                                                                                                                                                 
  uploadVideo: (file, metadata, signal) => {
    const form = new FormData();
    form.append('file', file);
    if (metadata && Object.keys(metadata).length > 0) {
      form.append('metadata', JSON.stringify(metadata));
    }
    return request('/api/videos/upload', { method: 'POST', body: form, signal });
  },

  getProcessingStatus: (jobId, signal) =>
    request(`/api/processing/${encodeURIComponent(jobId)}`, { signal }),

                                                                                                                                                                                                                                                                     
  getMatchSummary: (matchId, signal) =>
    request(`/api/matches/${encodeURIComponent(matchId)}`, { signal }),

                                                                                                                                                                                                                         
  listMatches: (signal) => request('/api/matches', { signal }),

                                                                                                                                                                                                          
  videoFileUrl: (videoId) => `${API_BASE}/api/videos/${encodeURIComponent(videoId)}/file`,

                                                                                                                                                                                                                                                                                                                                                        
  processedVideoUrl: (videoId) =>
    `${API_BASE}/api/videos/${encodeURIComponent(videoId)}/processed`,

  getTracking: (matchId, { startFrame = 0, endFrame } = {}, signal) => {
    const params = new URLSearchParams({ start_frame: String(startFrame) });
    if (endFrame !== undefined && endFrame !== null) {
      params.set('end_frame', String(endFrame));
    }
    return request(
      `/api/matches/${encodeURIComponent(matchId)}/tracking?${params.toString()}`,
      { signal },
    );
  },

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
  getEvents: (matchId, { eventType, limit } = {}, signal) => {
    const params = new URLSearchParams();
    if (eventType) params.set('event_type', eventType);
    if (limit) params.set('limit', String(limit));
    const query = params.toString();
    return request(
      `/api/matches/${encodeURIComponent(matchId)}/events${query ? `?${query}` : ''}`,
      { signal },
    );
  },

                                                                                                                                                                                                                                                                                                                                                              
  getHeatmap: (matchId, playerId, space = 'pitch', signal) =>
    request(
      `/api/matches/${encodeURIComponent(matchId)}/heatmap/${encodeURIComponent(playerId)}`
        + `?space=${encodeURIComponent(space)}`,
      { signal },
    ),

                                                                                                                                                         
  getPipelineLatency: (jobId, signal) =>
    request(`/api/pipeline/latency/${encodeURIComponent(jobId)}`, { signal }),

                                                                            
  getPlayerIntelligence: (matchId, signal) =>
    request(`/api/player_intelligence/${encodeURIComponent(matchId)}`, { signal }),

  getPlayerMetrics: (matchId, playerId, signal) =>
    request(
      `/api/player_intelligence/${encodeURIComponent(matchId)}/${encodeURIComponent(playerId)}`,
      { signal },
    ),

                                                                            

                                                                     
                                                                          
                                                                     
                                                                      
                                                                   
  getFormation: (matchId, teamId, signal) => {
    const params = new URLSearchParams();
    if (teamId) params.set('team_id', teamId);
    const query = params.toString();
    return request(
      `/api/tactical/formation/${encodeURIComponent(matchId)}${query ? `?${query}` : ''}`,
      { signal },
    );
  },

  getTeamShape: (matchId, teamId, signal) => {
    const params = new URLSearchParams();
    if (teamId) params.set('team_id', teamId);
    const query = params.toString();
    return request(
      `/api/tactical/team_shape/${encodeURIComponent(matchId)}${query ? `?${query}` : ''}`,
      { signal },
    );
  },

  getTeamIntelligence: (matchId, signal) =>
    request(`/api/team_intelligence/${encodeURIComponent(matchId)}`, { signal }),

                                                                             
  getCalibrationDebug: (matchId, frameNumber, signal) =>
    request(
      `/api/matches/${encodeURIComponent(matchId)}/calibration-debug/${encodeURIComponent(frameNumber)}`,
      { signal },
    ),

                                                                             
  runSimulation: (matchId, interventions, signal) =>
    request(`/api/simulation/${encodeURIComponent(matchId)}`, {
      method: 'POST',
      json: { interventions },
      signal,
    }),

                                                                            
  submitPreMatchHealth: (playerId, payload, signal) =>
    request(`/api/prematch_health/${encodeURIComponent(playerId)}/submit`, {
      method: 'POST',
      json: payload,
      signal,
    }),

  getPreMatchHealthLatest: (playerId, signal) =>
    request(`/api/prematch_health/${encodeURIComponent(playerId)}/latest`, { signal }),

  getPreMatchHealthHistory: (playerId, limit = 20, signal) =>
    request(
      `/api/prematch_health/${encodeURIComponent(playerId)}/history?limit=${encodeURIComponent(limit)}`,
      { signal },
    ),

                                                                            
  submitPsychology: (playerId, payload, signal) =>
    request(`/api/psychology/${encodeURIComponent(playerId)}/submit`, {
      method: 'POST',
      json: payload,
      signal,
    }),

  getPsychologyLatest: (playerId, matchId, signal) => {
    const params = new URLSearchParams();
    if (matchId) params.set('match_id', matchId);
    const query = params.toString();
    return request(
      `/api/psychology/${encodeURIComponent(playerId)}/latest${query ? `?${query}` : ''}`,
      { signal },
    );
  },

  getPsychologyHistory: (playerId, { limit = 20, matchId } = {}, signal) => {
    const params = new URLSearchParams({ limit: String(limit) });
    if (matchId) params.set('match_id', matchId);
    return request(
      `/api/psychology/${encodeURIComponent(playerId)}/history?${params.toString()}`,
      { signal },
    );
  },
};

export default api;
