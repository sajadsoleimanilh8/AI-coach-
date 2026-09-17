import { Chip, MetaRow, Panel } from '../../ui/Primitives.jsx';
import { EmptyState, ErrorState, LoadingState } from '../../ui/States.jsx';
import { formatTime } from './format.jsx';
import { POLL_MS, TERMINAL } from './jobStatus.js';

export function ProcessingPanel({ job, jobId, filename }) {
  if (!jobId) {
    return (
      <Panel number="02" title="Processing Pipeline" subtitle="GET /api/processing/{job_id}">
        <EmptyState
          title="No Job Queued"
          message="Upload a clip to start a processing job. This panel then polls the real job status until it completes or fails."
        />
      </Panel>
    );
  }

  if (job.status === 'loading' && !job.data) {
    return (
      <Panel number="02" title="Processing Pipeline" subtitle="GET /api/processing/{job_id}">
        <LoadingState label="Reading job status" rows={2} />
      </Panel>
    );
  }

  if (job.status === 'error') {
    return (
      <Panel number="02" title="Processing Pipeline" subtitle="GET /api/processing/{job_id}">
        <ErrorState error={job.error} onRetry={job.retry} />
      </Panel>
    );
  }

  const data = job.data;
  const progress = Math.max(0, Math.min(100, Number(data?.progress) || 0));
  const failed = data?.status === 'failed';
  const done = data?.status === 'completed';

  return (
    <Panel
      number="02"
      title="Processing Pipeline"
      subtitle={filename ? `Job for ${filename}` : 'GET /api/processing/{job_id}'}
      tone={failed ? 'bad' : done ? 'good' : ''}
    >
      <div className="dash-progress" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
        <div className={`dash-progress-fill ${failed ? 'is-failed' : ''} ${!TERMINAL.has(data?.status) ? 'is-live' : ''}`.trim()} style={{ width: `${progress}%` }} />
      </div>

      <div className="dash-progress-head">
        <Chip tone={failed ? 'bad' : done ? 'good' : 'neutral'}>{data?.status}</Chip>
        <span className="dash-progress-pct">{progress}%</span>
      </div>

      {data?.message ? <p className="dash-hint">{data.message}</p> : null}

      {failed && data?.error ? (
        <div className="dash-inline-error" role="alert">
          <span className="dash-inline-error-key">Pipeline error</span>
          {data.error}
        </div>
      ) : null}

      <div className="dash-meta-block">
        <MetaRow label="job_id">
          <code>{data?.job_id}</code>
        </MetaRow>
        <MetaRow label="match_id">
          {data?.match_id ? <code>{data.match_id}</code> : <span className="dash-absent">Not linked</span>}
        </MetaRow>
        <MetaRow label="created">{formatTime(data?.created_at)}</MetaRow>
        <MetaRow label="started">{formatTime(data?.started_at)}</MetaRow>
        <MetaRow label="completed">{formatTime(data?.completed_at)}</MetaRow>
      </div>

      {!TERMINAL.has(data?.status) ? (
        <p className="dash-hint dash-polling">
          <span className="dash-polling-dot" aria-hidden="true" />
          Polling every {POLL_MS / 1000}s until this job reaches a terminal state.
        </p>
      ) : null}
    </Panel>
  );
}
