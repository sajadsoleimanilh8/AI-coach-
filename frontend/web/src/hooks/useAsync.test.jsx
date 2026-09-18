// @vitest-environment jsdom
import { StrictMode } from 'react';
import { act, renderHook, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { useAction, useAsync } from './useAsync.js';

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe('useAsync', () => {
  it('goes loading -> success with the loader result', async () => {
    const { result } = renderHook(() => useAsync(async () => 42, []));
    expect(result.current.status).toBe('loading');
    await waitFor(() => expect(result.current.status).toBe('success'));
    expect(result.current.data).toBe(42);
    expect(result.current.error).toBeNull();
  });

  it('goes loading -> error and exposes the error', async () => {
    const boom = new Error('boom');
    const { result } = renderHook(() => useAsync(async () => { throw boom; }, []));
    await waitFor(() => expect(result.current.status).toBe('error'));
    expect(result.current.error).toBe(boom);
    expect(result.current.data).toBeNull();
  });

  it('stays idle and never calls the loader while disabled', async () => {
    const loader = vi.fn(async () => 1);
    const { result } = renderHook(() => useAsync(loader, [], { enabled: false }));
    expect(result.current.status).toBe('idle');
    expect(loader).not.toHaveBeenCalled();
  });

  it('aborts the previous request when deps change, so a stale response cannot land', async () => {
    const signals = [];
    const pending = [];
    const loader = (signal) => {
      signals.push(signal);
      const d = deferred();
      pending.push(d);
      return d.promise;
    };
    const { result, rerender } = renderHook(({ id }) => useAsync(loader, [id]), {
      initialProps: { id: 'match-A' },
    });
    rerender({ id: 'match-B' });

    expect(signals[0].aborted).toBe(true);
    expect(signals.at(-1).aborted).toBe(false);

    // The slow response for match A arrives AFTER the switch: it must be ignored.
    await act(async () => { pending[0].resolve('A-data'); });
    await act(async () => { pending.at(-1).resolve('B-data'); });
    await waitFor(() => expect(result.current.data).toBe('B-data'));
  });

  it('does not report its own abort as an error', async () => {
    const loader = (signal) => new Promise((_, reject) => {
      signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
    });
    const { result, unmount } = renderHook(() => useAsync(loader, []));
    unmount();
    expect(result.current.status).toBe('loading');
  });

  it('reload() runs the loader again', async () => {
    const loader = vi.fn(async () => 'x');
    const { result } = renderHook(() => useAsync(loader, []));
    await waitFor(() => expect(result.current.status).toBe('success'));
    const callsBefore = loader.mock.calls.length;
    act(() => result.current.reload());
    await waitFor(() => expect(loader.mock.calls.length).toBe(callsBefore + 1));
  });
});

describe('useAction', () => {
  it('starts idle, then run() resolves to success and returns the data', async () => {
    const { result } = renderHook(() => useAction(async (x) => x * 2));
    expect(result.current.status).toBe('idle');
    let returned;
    await act(async () => { returned = await result.current.run(21); });
    expect(returned).toBe(42);
    expect(result.current.status).toBe('success');
    expect(result.current.data).toBe(42);
  });

  it('passes an AbortSignal as the last argument', async () => {
    const action = vi.fn(async () => 'ok');
    const { result } = renderHook(() => useAction(action));
    await act(async () => { await result.current.run('a', 'b'); });
    const args = action.mock.calls[0];
    expect(args.slice(0, 2)).toEqual(['a', 'b']);
    expect(args[2]).toBeInstanceOf(AbortSignal);
  });

  it('reports a failure as error state', async () => {
    const boom = new Error('nope');
    const { result } = renderHook(() => useAction(async () => { throw boom; }));
    await act(async () => { await result.current.run(); });
    expect(result.current.status).toBe('error');
    expect(result.current.error).toBe(boom);
  });

  it('firing again aborts the previous run', async () => {
    const signals = [];
    const { result } = renderHook(() => useAction((signal) => {
      signals.push(signal);
      return new Promise(() => {});
    }));
    act(() => { result.current.run(); });
    act(() => { result.current.run(); });
    expect(signals[0].aborted).toBe(true);
    expect(signals[1].aborted).toBe(false);
  });

  it('reset() returns to idle', async () => {
    const { result } = renderHook(() => useAction(async () => 1));
    await act(async () => { await result.current.run(); });
    act(() => result.current.reset());
    expect(result.current.status).toBe('idle');
  });

  it('still delivers results under StrictMode (regression: hung forever on every submit)', async () => {
    // StrictMode runs effect setup -> cleanup -> setup. The cleanup used to
    // leave mountedRef false for good, so every later run() result was
    // silently discarded and the UI sat on "loading" forever.
    const { result } = renderHook(() => useAction(async () => 'saved'), { wrapper: StrictMode });
    await act(async () => { await result.current.run(); });
    expect(result.current.status).toBe('success');
    expect(result.current.data).toBe('saved');
  });
});
