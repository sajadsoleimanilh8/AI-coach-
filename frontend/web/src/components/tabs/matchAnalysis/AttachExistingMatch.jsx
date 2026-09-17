import { useState } from 'react';
import api from '../../../api/client.js';
import { useAction } from '../../../hooks/useAsync.js';
import { Chip, MetaRow, Slab } from '../../ui/Primitives.jsx';
import { ErrorState } from '../../ui/States.jsx';
import { SubmitButton, TextField } from '../../ui/Form.jsx';

/* ==================================================================== */
/* Attach to an already-processed match */
/* ==================================================================== */

/**
 * Upload is how a match is normally created, but it is not the only way one
 * can exist: the database routinely already holds matches processed by an
 * earlier run. Without this, every match-scoped tab would sit on "Select a
 * Match First" while real tracking rows for those matches went unreachable,
 * and the only route to them would be re-uploading a video that has already
 * been processed once.
 *
 * This sets the SAME shared matchId the upload path sets. It deliberately
 * does not invent a video_id or job_id: those belong to a specific upload,
 * so playback, the processing poller, and the latency report stay in their
 * honest empty states rather than pointing at a guessed id.
 */
export function AttachExistingMatch({ onAttachMatch }) {
  const [draft, setDraft] = useState('');

  // Resolves the match rather than merely confirming it: the summary carries
  // the video_id and job_id that playback, the processing poller, and the
  // latency report are keyed by. A 404 here means the id is wrong, which is
  // far better caught now than as four separate empty tabs later.
  const check = useAction((id, signal) => api.getMatchSummary(id, signal));

  const submit = async (event) => {
    event.preventDefault();
    const id = draft.trim();
    if (!id) return;
    const summary = await check.run(id);
    if (summary) onAttachMatch(summary);
  };

  return (
    <Slab>
      <div className="dash-subhead-inline">
        <h4 className="dash-card-title">Or Attach an Already-Processed Match</h4>
        <p className="dash-hint">
          Paste a <code>match_id</code> that has already been through the pipeline. Its video_id and
          job_id are resolved from the backend, so playback, calibration, and the latency report
          work too — nothing here is guessed.
        </p>
      </div>

      <form className="dash-form dash-form-inline" onSubmit={submit}>
        <TextField
          label="Existing match_id"
          value={draft}
          onChange={(value) => setDraft(value ?? '')}
          placeholder="e.g. bb58ad9f-e02a-47f3-9aa3-6ced09d009d0"
        />
        <SubmitButton busy={check.status === 'loading'} busyLabel="Checking" disabled={!draft.trim()}>
          Attach Match
        </SubmitButton>
      </form>

      {check.status === 'error' ? (
        <ErrorState
          error={check.error}
          title={check.error?.notFound ? 'No Such Match' : 'Could Not Attach'}
        />
      ) : null}

      {check.status === 'success' && check.data ? (
        <div className="dash-meta-block">
          <div className="dash-badge-row">
            <Chip tone="good">attached</Chip>
            <code className="dash-code-sm">{check.data.match_id}</code>
          </div>
          <MetaRow label="teams">
            {check.data.home_team || '—'} vs {check.data.away_team || '—'}
          </MetaRow>
          <MetaRow label="video">
            {check.data.video_id ? (
              check.data.video_file_exists ? (
                <Chip tone="good">playable</Chip>
              ) : (
                // The row exists but the file is gone -- said plainly rather
                // than letting the player fail with a broken source.
                <Chip tone="warn">row exists, file missing from disk</Chip>
              )
            ) : (
              <span className="dash-absent">no video row</span>
            )}
          </MetaRow>
          <MetaRow label="job">
            {check.data.job_id ? (
              <Chip tone={check.data.job_status === 'completed' ? 'good' : 'neutral'}>
                {check.data.job_status}
              </Chip>
            ) : (
              <span className="dash-absent">no processing job</span>
            )}
          </MetaRow>
        </div>
      ) : null}
    </Slab>
  );
}
