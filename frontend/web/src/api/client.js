/**
 * Thin fetch client for the CORE backend (FastAPI, backend/api/main.py).
 *
 * Deliberately separate from src/api/nexus.js: the two services run on
 * different ports, start and stop independently, and one being down says
 * nothing about the other. Each client therefore carries its OWN
 * "unreachable" message so a user staring at an error knows which process to
 * go restart.
 *
 * Every call takes an AbortSignal. Tabs pass one from useAsync so an
 * in-flight request is cancelled when the tab unmounts or the active match
 * changes -- otherwise a slow response from a previous match can land after
 * the switch and paint stale numbers under the new match's header.
 */

// IPv4 literal, not the name 'localhost'. On Windows, Node/Chrome resolve
// 'localhost' to ::1 before 127.0.0.1, but uvicorn binds IPv4 only -- so every
// request would spend ~2s failing against ::1 before falling back, and when the
// browser gives up instead, the UI reports CORE_UNREACHABLE (which blames CORS).
// Override with VITE_API_BASE_URL when deploying anywhere but a dev machine.
const DEFAULT_BASE = 'http://127.0.0.1:8000';

export const API_BASE = String(
  import.meta.env?.VITE_API_BASE_URL || DEFAULT_BASE,
).replace(/\/+$/, '');

// Only needed when the backend runs with SSC_API_KEY set. VITE_* values are
// compiled into the public bundle, so this keeps the API off the open network;
// it does not hide the key from anyone who can load the dashboard.
const API_KEY = String(import.meta.env?.VITE_SSC_API_KEY || '');

export function authHeaders(key) {
  return key ? { 'X-API-Key': key } : {};
}

// <video src> cannot send headers; the backend accepts ?api_key= on the two
// media routes for exactly this reason.
function withMediaKey(url) {
  return API_KEY ? `${url}?api_key=${encodeURIComponent(API_KEY)}` : url;
}

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
    /** true when the request never reached the server at all (down / CORS / DNS). */
    this.unreachable = status === 0;
    this.notFound = status === 404;
  }
}

/**
 * FastAPI puts its error under {"detail": ...} -- but `detail` is a plain
 * string for HTTPException and a list of {loc, msg, type} objects for a 422
 * validation failure. Rendering the raw object gives the user "[object
 * Object]", so both shapes are flattened to readable text here rather than
 * at eight separate call sites.
 */
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

/**
 * Shared transport for both clients. `unreachableMessage` and `service` are
 * what make the core/NEXUS distinction; everything else is identical.
 */
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
    // FormData: the browser must set its own multipart boundary, so no
    // Content-Type is written here on purpose.
    init.body = body;
  }

  let response;
  try {
    response = await fetch(`${base}${path}`, init);
  } catch (error) {
    // An aborted request is a normal control-flow event (tab switch,
    // unmount), not a backend outage -- it must propagate untouched so
    // useAsync can drop it instead of painting an error state.
    if (error?.name === 'AbortError') throw error;
    throw new ApiError(unreachableMessage, { status: 0, service });
  }

  if (!response.ok) {
    let detail = null;
    try {
      const payload = await response.json();
      detail = payload?.detail ?? payload;
    } catch {
      /* non-JSON error body; fall through to the status-code message */
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

function request(path, options = {}) {
  return requestWith(API_BASE, path, {
    ...options,
    headers: { ...authHeaders(API_KEY), ...options.headers },
  });
}

export const api = {
  // --- health -------------------------------------------------------------
  health: (signal) => request('/health', { signal }),

  // --- TAB 1: Match Analysis ---------------------------------------------

  /**
   * POST /api/videos/upload -- multipart. `metadata` is sent as a JSON
   * *string* field because the endpoint declares it as Form(str) and parses
   * it itself; sending an object would arrive as "[object Object]" and come
   * back as a 400.
   */
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

  /**
   * Resolves an already-processed match to its video_id / job_id, which the
   * upload response would otherwise be the only source of. Without this,
   * attaching a pre-existing match leaves playback and the latency report
   * permanently empty.
   */
  getMatchSummary: (matchId, signal) =>
    request(`/api/matches/${encodeURIComponent(matchId)}`, { signal }),

  /**
   * Every match in the database, newest first. This is how a match id is
   * DISCOVERED rather than remembered -- without it the only source of one was
   * an upload made in the current browser session.
   */
  listMatches: (signal) => request('/api/matches', { signal }),

  /**
   * Not a fetch: this is the URL handed straight to <video src>. The clip is
   * written to disk at upload time, before the pipeline runs, so it plays
   * regardless of processing status.
   */
  videoFileUrl: (videoId) =>
    withMediaKey(`${API_BASE}/api/videos/${encodeURIComponent(videoId)}/file`),

  /**
   * The ANNOTATED render: the same clip with the pipeline's own tracking
   * boxes, track ids and team colours burned in. Written by the last pipeline
   * stage, so it exists only after a completed run -- callers check
   * `processed_video_exists` on the match summary rather than probing this
   * and treating a 404 as normal.
   */
  processedVideoUrl: (videoId) =>
    withMediaKey(`${API_BASE}/api/videos/${encodeURIComponent(videoId)}/processed`),

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

  /**
   * Detected pass / shot / turnover / first-touch events for a match,
   * chronological.
   *
   * ADDED: the pipeline has always written these rows and no endpoint read
   * them back, so nothing in the product could show them. The response's
   * `space` says whether possession was resolved in pitch metres or in image
   * pixels (the fallback used when calibration does not validate) -- the UI
   * must render that label, not assume the stronger of the two.
   */
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

  /**
   * `space` selects the coordinate system: "pitch" (metres, needs a valid
   * homography) or "image" (pixels in the video frame, needs none). They are
   * not interchangeable readings -- see HeatmapResponse.space in
   * backend/api/schemas.py -- so the caller must render the label the
   * response reports, not the one it asked for.
   */
  getHeatmap: (matchId, playerId, space = 'pitch', signal) =>
    request(
      `/api/matches/${encodeURIComponent(matchId)}/heatmap/${encodeURIComponent(playerId)}`
        + `?space=${encodeURIComponent(space)}`,
      { signal },
    ),

  /**
   * 404 here means "no stage has finished yet", not "something broke" --
   * callers check `error.notFound` and render a "not ready" state.
   */
  getPipelineLatency: (jobId, signal) =>
    request(`/api/pipeline/latency/${encodeURIComponent(jobId)}`, { signal }),

  // --- TAB 2: Player Intelligence ----------------------------------------
  getPlayerIntelligence: (matchId, signal) =>
    request(`/api/player_intelligence/${encodeURIComponent(matchId)}`, { signal }),

  getPlayerMetrics: (matchId, playerId, signal) =>
    request(
      `/api/player_intelligence/${encodeURIComponent(matchId)}/${encodeURIComponent(playerId)}`,
      { signal },
    ),

  // --- TAB 3: Team Intelligence ------------------------------------------

  // Both of these are scoped to a team_id. The endpoint's default is
  // "unassigned", which only matches runs where team assignment failed --
  // a run that DID split the teams writes its rows under its own ids
  // (e.g. team-home / team-away), and omitting the param 404s against
  // perfectly good data. So the caller passes the team explicitly.
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

  // --- TAB 4: Calibration -------------------------------------------------
  getCalibrationDebug: (matchId, frameNumber, signal) =>
    request(
      `/api/matches/${encodeURIComponent(matchId)}/calibration-debug/${encodeURIComponent(frameNumber)}`,
      { signal },
    ),

  // --- TAB 5: Simulation --------------------------------------------------
  runSimulation: (matchId, interventions, signal) =>
    request(`/api/simulation/${encodeURIComponent(matchId)}`, {
      method: 'POST',
      json: { interventions },
      signal,
    }),

  // --- TAB 6: Pre-Match Health -------------------------------------------
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

  // --- TAB 8: Pre-Match Psychology ---------------------------------------
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
