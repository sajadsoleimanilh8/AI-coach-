import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Runs an async loader and exposes the three states every view in this
 * dashboard is required to have: loading, populated, error/empty. There is
 * deliberately no fourth "silently blank" state -- `status` is always one of
 * idle / loading / success / error, and idle only happens when `enabled` is
 * false (i.e. a real precondition like "no match selected yet" is unmet,
 * which the caller renders as its own empty state).
 *
 * The loader receives an AbortSignal and MUST pass it through to the API
 * client. When deps change or the component unmounts the previous request is
 * aborted, so a slow response for match A can never land after the user has
 * switched to match B.
 *
 * @param {(signal: AbortSignal) => Promise<any>} loader
 * @param {Array} deps
 * @param {{enabled?: boolean}} options
 */
export function useAsync(loader, deps = [], { enabled = true } = {}) {
  const [state, setState] = useState({
    status: enabled ? 'loading' : 'idle',
    data: null,
    error: null,
  });

  // Kept in a ref so `reload` is stable across renders and callers can drop
  // it into an onClick without re-subscribing anything.
  const loaderRef = useRef(loader);
  loaderRef.current = loader;

  const [nonce, setNonce] = useState(0);
  const reload = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    if (!enabled) {
      setState({ status: 'idle', data: null, error: null });
      return undefined;
    }

    const controller = new AbortController();
    let active = true;

    setState((previous) => ({ ...previous, status: 'loading', error: null }));

    loaderRef
      .current(controller.signal)
      .then((data) => {
        if (!active) return;
        setState({ status: 'success', data, error: null });
      })
      .catch((error) => {
        // An abort is this hook's own doing, not a failure to report.
        if (!active || error?.name === 'AbortError') return;
        setState({ status: 'error', data: null, error });
      });

    return () => {
      active = false;
      controller.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled, nonce]);

  return { ...state, reload };
}

/**
 * A one-shot action (form submit, "run simulation", "render frame") rather
 * than a load-on-mount fetch. Same abort discipline: firing again cancels the
 * previous attempt, and unmounting cancels whatever is in flight.
 */
export function useAction(action) {
  const [state, setState] = useState({ status: 'idle', data: null, error: null });
  const controllerRef = useRef(null);
  const mountedRef = useRef(true);
  const actionRef = useRef(action);
  actionRef.current = action;

  useEffect(() => {
    // StrictMode (dev only) deliberately runs this effect's setup, then its
    // cleanup, then setup again, to surface exactly the bug this guards
    // against: without re-arming it true here, the cleanup below leaves
    // mountedRef permanently false after that dance, even though the
    // component is genuinely mounted -- silently discarding the result of
    // every future run() for the rest of this component's life.
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      controllerRef.current?.abort();
    };
  }, []);

  const run = useCallback(async (...args) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    setState({ status: 'loading', data: null, error: null });
    try {
      const data = await actionRef.current(...args, controller.signal);
      if (!mountedRef.current || controller.signal.aborted) return undefined;
      setState({ status: 'success', data, error: null });
      return data;
    } catch (error) {
      if (!mountedRef.current || error?.name === 'AbortError') return undefined;
      setState({ status: 'error', data: null, error });
      return undefined;
    }
  }, []);

  const reset = useCallback(() => {
    controllerRef.current?.abort();
    setState({ status: 'idle', data: null, error: null });
  }, []);

  return { ...state, run, reset };
}

export default useAsync;
