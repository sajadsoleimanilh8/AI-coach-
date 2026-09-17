import api from '../../../api/client.js';
import { useAsync } from '../../../hooks/useAsync.js';
import { MethodStrip, Slab, StatNum, StatRow } from '../../ui/Primitives.jsx';
import { AsyncBlock, EmptyState } from '../../ui/States.jsx';
import { formatTime } from './format.jsx';

/* ==================================================================== */
/* Pipeline latency                                                     */
/* ==================================================================== */

export function LatencyPanel({ jobId, dataEpoch = 0 }) {
  const latency = useAsync(
    (signal) => api.getPipelineLatency(jobId, signal),
    [jobId, dataEpoch],
    { enabled: Boolean(jobId) },
  );

  if (!jobId) {
    return (
      <Slab>
        <EmptyState
          title="No Job Yet"
          message="Stage timings are measured during a real run and keyed by job_id."
        />
      </Slab>
    );
  }

  // A 404 here is "no stage has finished yet", which is a normal state during
  // (or shortly after) an upload -- not a failure to report as one.
  if (latency.status === 'error' && latency.error?.notFound) {
    return (
      <Slab>
        <EmptyState
          title="Latency Report Not Ready"
          message="No stage of this job has completed a measured run yet. The report appears here as soon as one does."
          action={
            <button type="button" className="btn-ghost" onClick={latency.reload}>
              Check Again
            </button>
          }
        />
      </Slab>
    );
  }

  return (
    <AsyncBlock
      state={latency}
      loadingLabel="Reading measured stage timings"
      isEmpty={(data) => !data?.stages?.length}
      emptyTitle="No Stages Recorded"
      emptyMessage="The report exists but contains no stage rows."
    >
      {(data) => {
        const max = Math.max(...data.stages.map((stage) => stage.seconds || 0), 0.0001);
        return (
          <>
            <StatRow>
              <StatNum value={data.total_seconds} label="Seconds Total" />
              <StatNum value={data.stages.length} label="Measured Stages" />
            </StatRow>

            <MethodStrip
              items={[
                { label: 'source', value: data.source, tone: 'good' },
                { label: 'job_id', value: data.job_id },
                { label: 'generated', value: formatTime(data.generated_at) },
              ]}
            />

            <Slab className="dash-stage-table">
              {data.stages.map((stage) => (
                <div key={stage.stage} className="dash-stage-row">
                  <span className="dash-stage-name">{stage.stage}</span>
                  <span className="dash-stage-bar">
                    <span
                      className="dash-stage-bar-fill"
                      style={{ width: `${((stage.seconds || 0) / max) * 100}%` }}
                    />
                  </span>
                  <span className="dash-stage-secs">{(stage.seconds ?? 0).toFixed(3)}s</span>
                  {stage.detail ? <span className="dash-stage-detail">{stage.detail}</span> : null}
                </div>
              ))}
            </Slab>
          </>
        );
      }}
    </AsyncBlock>
  );
}
