import { useCallback, useEffect, useRef, useState } from 'react';
import api from '../../../api/client.js';

export const TERMINAL = new Set(['completed', 'failed']);

export const POLL_MS = 1500;

/* ==================================================================== */
/* Processing pipeline                                                  */
/* ==================================================================== */

/**
 * Polls GET /api/processing/{job_id} until the job reaches a terminal state.
 *
 * Polling STOPS on a transport error rather than hammering a dead backend;
 * the user gets a Retry. It also stops on `completed`/`failed`, which
 * includes the broker-down case -- the upload endpoint marks the job failed
 * with an explanatory `error` when Celery is unreachable, so that arrives
 * here as a legitimate terminal status and not as a crash.
 */
export function useJobStatus(jobId) {
  const [state, setState] = useState({ status: jobId ? 'loading' : 'idle', data: null, error: null });
  const [nonce, setNonce] = useState(0);
  const retry = useCallback(() => setNonce((value) => value + 1), []);

  // A new job (or a retry) resets to loading -- or idle with no job -- in the
  // same render, rather than from the effect below (same pattern as useAsync).
  const [prevInputs, setPrevInputs] = useState({ jobId, nonce });
  if (prevInputs.jobId !== jobId || prevInputs.nonce !== nonce) {
    setPrevInputs({ jobId, nonce });
    setState({ status: jobId ? 'loading' : 'idle', data: null, error: null });
  }

  useEffect(() => {
    if (!jobId) return undefined;

    const controller = new AbortController();
    let active = true;
    let timer = null;

    const tick = async () => {
      try {
        const data = await api.getProcessingStatus(jobId, controller.signal);
        if (!active) return;
        setState({ status: 'success', data, error: null });
        if (!TERMINAL.has(data.status)) {
          timer = setTimeout(tick, POLL_MS);
        }
      } catch (error) {
        if (!active || error?.name === 'AbortError') return;
        setState({ status: 'error', data: null, error });
      }
    };

    tick();

    return () => {
      active = false;
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [jobId, nonce]);

  return { ...state, retry };
}

/**
 * Counts transitions INTO `completed` for the current job.
 *
 * Only a transition counts. A job that is *already* completed when it first
 * appears -- the "attach an existing match" path -- must not bump the epoch,
 * because the panels' own first fetch already sees that finished data and a
 * bump would just make every one of them fetch twice.
 */
export function useCompletionEpoch(jobId, jobStatus) {
  const [epoch, setEpoch] = useState(0);
  const seen = useRef({ jobId: null, status: null });

  useEffect(() => {
    const previous = seen.current;
    seen.current = { jobId, status: jobStatus ?? null };

    // First sighting of this job id: record where it started, judge nothing.
    if (previous.jobId !== jobId) return;

    if (previous.status && previous.status !== 'completed' && jobStatus === 'completed') {
      setEpoch((value) => value + 1);
    }
  }, [jobId, jobStatus]);

  return epoch;
}
