import { useRef, useState } from 'react';
import api from '../../../api/client.js';
import { useAction } from '../../../hooks/useAsync.js';
import { Chip, MetaRow, Panel, Slab } from '../../ui/Primitives.jsx';
import { ErrorState } from '../../ui/States.jsx';
import { SubmitButton, TextField } from '../../ui/Form.jsx';

/* ==================================================================== */
/* Upload                                                               */
/* ==================================================================== */

export function UploadPanel({ onUploadComplete }) {
  const [file, setFile] = useState(null);
  const [team, setTeam] = useState('');
  const [opponent, setOpponent] = useState('');
  const inputRef = useRef(null);

  const upload = useAction((payload, signal) =>
    api.uploadVideo(payload.file, payload.metadata, signal),
  );

  const submit = async (event) => {
    event.preventDefault();
    if (!file) return;

    const metadata = {};
    if (team.trim()) metadata.team = team.trim();
    if (opponent.trim()) metadata.opponent = opponent.trim();

    const result = await upload.run({ file, metadata });
    // Lifting match_id happens here and nowhere else -- it is the single
    // point in the system that knows this upload created this match.
    if (result?.match_id) onUploadComplete(result);
  };

  return (
    <Panel
      number="01"
      title="Upload Match Video"
      subtitle="POST /api/videos/upload — multipart. The clip is written to disk before the pipeline runs, so playback is available immediately."
    >
      <form className="dash-form" onSubmit={submit}>
        <div className="dash-field is-wide">
          <span className="dash-label">Match Clip</span>
          <button
            type="button"
            className="dash-filedrop"
            onClick={() => inputRef.current?.click()}
          >
            {file ? (
              <>
                <span className="dash-filedrop-name">{file.name}</span>
                <span className="dash-filedrop-meta">
                  {(file.size / (1024 * 1024)).toFixed(1)} MB · {file.type || 'unknown type'}
                </span>
              </>
            ) : (
              <>
                <span className="dash-filedrop-name">Select a video file</span>
                <span className="dash-filedrop-meta">mp4 · mpeg · mov · avi</span>
              </>
            )}
          </button>
          <input
            ref={inputRef}
            className="dash-file-input"
            type="file"
            accept="video/mp4,video/mpeg,video/quicktime,video/x-msvideo"
            onChange={(event) => setFile(event.target.files?.[0] || null)}
          />
        </div>

        <div className="dash-field-grid dash-field-grid-2">
          <TextField
            label="Team (optional)"
            value={team}
            onChange={(value) => setTeam(value ?? '')}
            placeholder="Home"
            hint="Stored as Match.home_team. Left blank it stays the honest placeholder “Home”."
          />
          <TextField
            label="Opponent (optional)"
            value={opponent}
            onChange={(value) => setOpponent(value ?? '')}
            placeholder="Away"
            hint="Stored as Match.away_team."
          />
        </div>

        <SubmitButton busy={upload.status === 'loading'} busyLabel="Uploading" disabled={!file}>
          Upload &amp; Queue
        </SubmitButton>
      </form>

      {upload.status === 'error' ? (
        <ErrorState error={upload.error} title="Upload Failed" onRetry={upload.reset} />
      ) : null}

      {upload.status === 'success' && upload.data ? (
        <Slab className="dash-upload-receipt" anim={false}>
          <MetaRow label="match_id">
            <code>{upload.data.match_id}</code>
          </MetaRow>
          <MetaRow label="video_id">
            <code>{upload.data.video_id}</code>
          </MetaRow>
          <MetaRow label="job_id">
            <code>{upload.data.job_id}</code>
          </MetaRow>
          <MetaRow label="status">
            <Chip tone={upload.data.status === 'failed' ? 'bad' : 'good'}>{upload.data.status}</Chip>
          </MetaRow>
          <p className="dash-hint">{upload.data.message}</p>
        </Slab>
      ) : null}
    </Panel>
  );
}
