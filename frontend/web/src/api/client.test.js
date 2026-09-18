import { afterEach, describe, expect, it, vi } from 'vitest';

import { ApiError, CORE_UNREACHABLE, authHeaders, formatDetail, requestWith } from './client.js';

const BASE = 'http://api.test';

function jsonResponse(status, body, { contentType = 'application/json', statusText = '' } = {}) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText,
    headers: { get: (name) => (name.toLowerCase() === 'content-type' ? contentType : null) },
    json: async () => (typeof body === 'string' ? JSON.parse(body) : body),
    text: async () => (typeof body === 'string' ? body : JSON.stringify(body)),
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('formatDetail', () => {
  it('returns null for a missing detail', () => {
    expect(formatDetail(null)).toBeNull();
    expect(formatDetail(undefined)).toBeNull();
  });

  it('passes an HTTPException string through unchanged', () => {
    expect(formatDetail('Match not found')).toBe('Match not found');
  });

  it('flattens a 422 validation list and drops the "body" loc segment', () => {
    const detail = [
      { loc: ['body', 'sleep_hours'], msg: 'Input should be greater than 0', type: 'greater_than' },
      { loc: ['query', 'team_id'], msg: 'Field required', type: 'missing' },
    ];
    expect(formatDetail(detail)).toBe(
      'sleep_hours: Input should be greater than 0 · query.team_id: Field required',
    );
  });

  it('falls back to the error type when msg is absent', () => {
    expect(formatDetail([{ loc: ['body', 'x'], type: 'missing' }])).toBe('x: missing');
  });

  it('keeps plain strings inside a list and skips unusable entries', () => {
    expect(formatDetail(['first', null, 42, { loc: ['body'] }, 'second'])).toBe('first · second');
  });

  it('returns null for a list with nothing readable, never "[object Object]"', () => {
    expect(formatDetail([{}, null])).toBeNull();
  });

  it('reads a single {msg} object', () => {
    expect(formatDetail({ msg: 'bad value' })).toBe('bad value');
  });

  it('serializes any other object as JSON rather than "[object Object]"', () => {
    expect(formatDetail({ code: 7 })).toBe('{"code":7}');
  });
});

describe('ApiError', () => {
  it('flags status 0 as unreachable and 404 as notFound', () => {
    expect(new ApiError('x', { status: 0 }).unreachable).toBe(true);
    expect(new ApiError('x', { status: 404 }).notFound).toBe(true);
    const e = new ApiError('x', { status: 500, service: 'nexus' });
    expect(e.unreachable).toBe(false);
    expect(e.notFound).toBe(false);
    expect(e.service).toBe('nexus');
    expect(e.name).toBe('ApiError');
  });
});

describe('authHeaders', () => {
  it('sends nothing when no key is configured, so local dev is unchanged', () => {
    expect(authHeaders('')).toEqual({});
    expect(authHeaders(undefined)).toEqual({});
  });

  it('sends X-API-Key when a key is configured', () => {
    expect(authHeaders('secret')).toEqual({ 'X-API-Key': 'secret' });
  });
});

describe('requestWith', () => {
  it('returns parsed JSON on success', async () => {
    const fetchMock = vi.fn(async () => jsonResponse(200, { ok: 1 }));
    vi.stubGlobal('fetch', fetchMock);
    await expect(requestWith(BASE, '/health')).resolves.toEqual({ ok: 1 });
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/health`, expect.objectContaining({ method: 'GET' }));
  });

  it('returns null for 204 No Content', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(204, '')));
    await expect(requestWith(BASE, '/x', { method: 'DELETE' })).resolves.toBeNull();
  });

  it('returns text for a non-JSON body', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(200, 'plain', { contentType: 'text/plain' })));
    await expect(requestWith(BASE, '/x')).resolves.toBe('plain');
  });

  it('turns a FastAPI error body into an ApiError with a readable message', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(404, { detail: 'No such match' })));
    const error = await requestWith(BASE, '/m/1').catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(404);
    expect(error.notFound).toBe(true);
    expect(error.message).toBe('No such match');
  });

  it('falls back to the status line when the error body is not JSON', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ...jsonResponse(502, '', { statusText: 'Bad Gateway' }),
      json: async () => { throw new SyntaxError('not json'); },
    })));
    const error = await requestWith(BASE, '/x').catch((e) => e);
    expect(error.message).toBe('Request failed: 502 Bad Gateway');
  });

  it('reports a network failure as unreachable (status 0), not as a server error', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('Failed to fetch'); }));
    const error = await requestWith(BASE, '/x').catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.unreachable).toBe(true);
    expect(error.message).toBe(CORE_UNREACHABLE);
  });

  it('lets an AbortError through untouched so callers can ignore it', async () => {
    const abort = new DOMException('aborted', 'AbortError');
    vi.stubGlobal('fetch', vi.fn(async () => { throw abort; }));
    await expect(requestWith(BASE, '/x')).rejects.toBe(abort);
  });

  it('sends a JSON body with a JSON content type', async () => {
    const fetchMock = vi.fn(async () => jsonResponse(200, {}));
    vi.stubGlobal('fetch', fetchMock);
    await requestWith(BASE, '/x', { method: 'POST', json: { a: 1 } });
    const init = fetchMock.mock.calls[0][1];
    expect(init.headers['Content-Type']).toBe('application/json');
    expect(init.body).toBe('{"a":1}');
  });

  it('leaves Content-Type unset for FormData so the browser adds the boundary', async () => {
    const fetchMock = vi.fn(async () => jsonResponse(200, {}));
    vi.stubGlobal('fetch', fetchMock);
    const form = new FormData();
    await requestWith(BASE, '/upload', { method: 'POST', body: form });
    const init = fetchMock.mock.calls[0][1];
    expect(init.body).toBe(form);
    expect(init.headers['Content-Type']).toBeUndefined();
  });

  it('merges caller headers over the defaults', async () => {
    const fetchMock = vi.fn(async () => jsonResponse(200, {}));
    vi.stubGlobal('fetch', fetchMock);
    await requestWith(BASE, '/x', { headers: { 'X-API-Key': 'k' } });
    expect(fetchMock.mock.calls[0][1].headers).toEqual({ Accept: 'application/json', 'X-API-Key': 'k' });
  });
});
