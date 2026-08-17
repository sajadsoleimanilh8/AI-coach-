import { useCallback, useEffect, useRef, useState } from 'react';

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
export function useAsync(loader, deps = [], { enabled = true } = {}) {
  const [state, setState] = useState({
    status: enabled ? 'loading' : 'idle',
    data: null,
    error: null,
  });

                                                                            
                                                        
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
                                                                      
        if (!active || error?.name === 'AbortError') return;
        setState({ status: 'error', data: null, error });
      });

    return () => {
      active = false;
      controller.abort();
    };
                                                           
  }, [...deps, enabled, nonce]);

  return { ...state, reload };
}

                                                                                                                                                                                                                                     
export function useAction(action) {
  const [state, setState] = useState({ status: 'idle', data: null, error: null });
  const controllerRef = useRef(null);
  const mountedRef = useRef(true);
  const actionRef = useRef(action);
  actionRef.current = action;

  useEffect(() => {
                                                                            
                                                                        
                                                                        
                                                                     
                                                                          
                                                                
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
