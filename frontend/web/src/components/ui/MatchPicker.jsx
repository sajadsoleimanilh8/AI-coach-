import { useMemo } from 'react';

import api from '../../api/client.js';
import { useAction, useAsync } from '../../hooks/useAsync.js';
import { EmptyState, ErrorState, LoadingState } from './States.jsx';

/**
 * The gate every match-scoped tab (Player, Team, Calibration, Simulation)
 * shows when no match is selected.
 *
 * It used to be a dead end: one "Go to Match Analysis" button, and the only
 * ways to actually get a match were to upload a clip in this very browser
 * session or to already know a UUID by heart. A refresh, a deep link, or
 * simply opening the console fresh left four of the eight tabs unusable with
 * nothing on screen that could fix it.
 *
 * So this is a picker, not a sign. It lists the matches that really exist
 * (GET /api/matches) and attaches the chosen one through the SAME
 * getMatchSummary path an upload uses, so video_id / job_id come from the
 * backend rather than being inferred here.
 *
 * Row copy is deliberately blunt about what a match contains: a match with
 * zero tracking rows is offered, and labelled as having none, rather than
 * hidden. Hiding it would leave someone whose job failed unable to see that
 * their upload exists at all.
 */
export function SelectMatchState({ onGoToUpload, onAttachMatch }) {
  // Degrades to the old sign-post when a caller has no way to attach a match,
  // rather than rendering a picker whose every click would do nothing.
  if (!onAttachMatch) {
    return (
      <EmptyState
        title="Select a Match First"
        message="This view is scoped to a single match. Upload a clip in Match Analysis — the match ID it returns is shared with every tab from that point on."
        action={
          onGoToUpload ? (
            <button type="button" className="btn-primary" onClick={onGoToUpload}>
              Go to Match Analysis
            </button>
          ) : null
        }
      />
    );
  }
  return <MatchPicker onGoToUpload={onGoToUpload} onAttachMatch={onAttachMatch} />;
}

export default function MatchPicker({ onGoToUpload, onAttachMatch }) {
  const matches = useAsync((signal) => api.listMatches(signal), []);
  const attach = useAction((matchId, signal) => api.getMatchSummary(matchId, signal));

  const rows = useMemo(
    () => (Array.isArray(matches.data) ? matches.data : []),
    [matches.data],
  );

  const choose = async (matchId) => {
    const summary = await attach.run(matchId);
    if (summary) onAttachMatch?.(summary);
  };

  return (
    <div className="dash-state dash-state-empty dash-match-picker">
      <div className="dash-state-label is-empty">Select a Match First</div>
      <p className="dash-state-message">
        This view is scoped to a single match. Pick one that has already been processed, or
        upload a new clip in Match Analysis.
      </p>

      {matches.status === 'loading' ? (
        <LoadingState label="Reading matches" rows={2} />
      ) : null}

      {matches.status === 'error' ? (
        <ErrorState error={matches.error} onRetry={matches.reload} />
      ) : null}

      {matches.status === 'success' && rows.length === 0 ? (
        <EmptyState
          title="No Matches Yet"
          message="The database holds no matches. Upload a clip in Match Analysis to create the first one."
        />
      ) : null}

      {matches.status === 'success' && rows.length > 0 ? (
        <ul className="dash-match-list">
          {rows.map((row) => {
            const usable = row.tracking_rows > 0;
            return (
              <li key={row.match_id} className="dash-match-row">
                <button
                  type="button"
                  className="dash-match-option"
                  onClick={() => choose(row.match_id)}
                  disabled={attach.status === 'loading'}
                >
                  <span className="dash-match-option-head">
                    <code>{row.match_id}</code>
                    <span className={`dash-match-flag ${usable ? 'is-good' : 'is-warn'}`}>
                      {usable
                        ? `${row.tracking_rows.toLocaleString()} tracking rows`
                        : 'no tracking rows'}
                    </span>
                  </span>
                  <span className="dash-match-option-meta">
                    {row.video_filename || 'no video file'}
                    {' · '}
                    {row.job_status || 'never processed'}
                    {row.created_at ? ` · ${new Date(row.created_at).toLocaleString()}` : ''}
                    {row.video_file_exists ? '' : ' · clip missing from disk'}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      ) : null}

      {attach.status === 'error' ? (
        <div className="dash-inline-error" role="alert">
          <span className="dash-inline-error-key">Could not attach</span>
          {attach.error?.message}
        </div>
      ) : null}

      {onGoToUpload ? (
        <div className="dash-state-action">
          <button type="button" className="btn-ghost" onClick={onGoToUpload}>
            Upload a new clip instead
          </button>
        </div>
      ) : null}
    </div>
  );
}
