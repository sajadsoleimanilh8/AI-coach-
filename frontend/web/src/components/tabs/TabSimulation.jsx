import { useMemo, useState } from 'react';

import api from '../../api/client.js';
import { useAction, useAsync } from '../../hooks/useAsync.js';
import {
  Chip,
  MetaRow,
  MethodStrip,
  MetricValue,
  Panel,
  PanelGrid,
  SectionHeader,
  Slab,
  SubHeading,
  SubScores,
  TabSection,
} from '../ui/Primitives.jsx';
import { EmptyState, ErrorState, LoadingState } from '../ui/States.jsx';
import { SelectMatchState } from '../ui/MatchPicker.jsx';
import { NumberField, SelectField, SubmitButton } from '../ui/Form.jsx';

/**
 * TAB 5 -- the what-if tactical simulator.
 *
 * The single most important thing this tab does is refuse to read as an
 * outcome predictor. The endpoint returns method: "heuristic_proxy" and
 * is_reinforcement_learning: false alongside a `caveats` list, and all three
 * are rendered above the numbers, not tucked under them. A deltas table with
 * the caveats hidden would be a more misleading artefact than no simulator at
 * all.
 */

const KINDS = [
  { value: 'compactness', label: 'Compactness', needs: ['team', 'pct'] },
  { value: 'transition_speed', label: 'Transition Speed', needs: ['team', 'pct'] },
  { value: 'remove_player', label: 'Remove Player', needs: ['player'] },
  { value: 'swap_player', label: 'Swap Player', needs: ['player', 'other'] },
];

const KIND_MAP = new Map(KINDS.map((kind) => [kind.value, kind]));
const MAX_INTERVENTIONS = 8;

let nextKey = 1;
function blankIntervention(kind = 'compactness') {
  return {
    key: nextKey++,
    kind,
    team_id: 'team_a',
    player_id: null,
    other_player_id: null,
    pct: 10,
  };
}

export default function TabSimulation({ matchId, jobId, goToTab, onAttachMatch }) {
  const [interventions, setInterventions] = useState(() => [blankIntervention()]);
  const simulation = useAction((payload, signal) => api.runSimulation(matchId, payload, signal));

  // A one-shot read, not a poll: this only decides whether to show the
  // "needs a completed pipeline run" gate before the user builds a scenario.
  const job = useAsync(
    (signal) => api.getProcessingStatus(jobId, signal),
    [jobId],
    { enabled: Boolean(jobId) },
  );

  const roster = useAsync(
    (signal) => api.getPlayerIntelligence(matchId, signal),
    [matchId],
    { enabled: Boolean(matchId) },
  );

  const playerOptions = useMemo(() => {
    const ids = Array.isArray(roster.data) ? roster.data.map((entry) => entry.player_id) : [];
    return [
      { value: '', label: ids.length ? 'Select a player…' : 'No roster available — type an ID' },
      ...ids.map((id) => ({ value: String(id), label: `Player ${id}` })),
    ];
  }, [roster.data]);

  const teamOptions = useMemo(() => {
    const ids = new Set(['team_a', 'team_b']);
    (Array.isArray(roster.data) ? roster.data : []).forEach((entry) => {
      (entry.metrics || []).forEach((metric) => {
        if (metric.team_id) ids.add(String(metric.team_id));
      });
    });
    return [...ids].map((id) => ({ value: id, label: id }));
  }, [roster.data]);


  const update = (key, patch) => {
    setInterventions((list) =>
      list.map((item) => (item.key === key ? { ...item, ...patch } : item)),
    );
  };

  const submit = (event) => {
    event.preventDefault();
    const payload = interventions.map((item) => {
      const needs = KIND_MAP.get(item.kind)?.needs || [];
      const body = { kind: item.kind };
      if (needs.includes('team')) body.team_id = item.team_id;
      if (needs.includes('pct')) body.pct = Number(item.pct);
      if (needs.includes('player')) body.player_id = Number(item.player_id);
      if (needs.includes('other')) body.other_player_id = Number(item.other_player_id);
      return body;
    });
    simulation.run(payload);
  };

  const incomplete = interventions.some((item) => {
    const needs = KIND_MAP.get(item.kind)?.needs || [];
    if (needs.includes('player') && (item.player_id === null || item.player_id === '')) return true;
    if (needs.includes('other') && (item.other_player_id === null || item.other_player_id === '')) return true;
    return false;
  });

  const pipelineIncomplete = Boolean(jobId) && job.status === 'success' && job.data?.status !== 'completed';

  return (
    <TabSection animKey={`simulation-${matchId}`}>
      <SectionHeader
        tag="What-If Simulation"
        title={<>Heuristic<br />Scenarios</>}
        meta="Adjust a tactical parameter and see how the recomputed proxy metrics move. This is a transparent heuristic, not a learned outcome model."
      />

      <MethodBanner />

      {!matchId ? (
        <SelectMatchState onGoToUpload={() => goToTab('match')} onAttachMatch={onAttachMatch} />
      ) : pipelineIncomplete ? (
        <Slab>
          <EmptyState
            title="Simulation Requires a Completed Real Pipeline Run"
            message={`The simulator recomputes proxy metrics from persisted tracking rows. This match's processing job is currently "${job.data.status}", so there is nothing real to perturb yet.`}
            action={
              <button type="button" className="btn-ghost" onClick={() => goToTab('match')}>
                Back to Match Analysis
              </button>
            }
          />
        </Slab>
      ) : (
        <>
          <SubHeading
            action={
              <button
                type="button"
                className="btn-ghost dash-add-btn"
                disabled={interventions.length >= MAX_INTERVENTIONS}
                onClick={() => setInterventions((list) => [...list, blankIntervention()])}
              >
                + Add Intervention ({interventions.length}/{MAX_INTERVENTIONS})
              </button>
            }
          >
            Interventions
          </SubHeading>

          <form className="dash-sim-form" onSubmit={submit}>
            {interventions.map((item, index) => (
              <InterventionRow
                key={item.key}
                index={index}
                item={item}
                teamOptions={teamOptions}
                playerOptions={playerOptions}
                canRemove={interventions.length > 1}
                onChange={(patch) => update(item.key, patch)}
                onRemove={() =>
                  setInterventions((list) => list.filter((entry) => entry.key !== item.key))
                }
              />
            ))}

            <div className="dash-form-actions">
              <SubmitButton
                busy={simulation.status === 'loading'}
                busyLabel="Simulating"
                disabled={incomplete}
              >
                Run What-If
              </SubmitButton>
              {incomplete ? (
                <p className="dash-hint dash-absent">
                  Every remove/swap intervention needs a player ID before this can run.
                </p>
              ) : null}
            </div>
          </form>

          {simulation.status === 'loading' ? (
            <LoadingState label="Recomputing proxy metrics" rows={3} />
          ) : null}

          {simulation.status === 'error' ? (
            <ErrorState error={simulation.error} title="Simulation Rejected" />
          ) : null}

          {simulation.status === 'success' && simulation.data ? (
            <SimulationResult data={simulation.data} />
          ) : null}
        </>
      )}
    </TabSection>
  );
}

/** Rendered before any result exists, so the framing never arrives late. */
function MethodBanner() {
  return (
    <Slab className="dash-method-banner">
      <div className="dash-badge-row">
        <Chip tone="warn">method: heuristic_proxy</Chip>
        <Chip tone="warn">is_reinforcement_learning: false</Chip>
      </div>
      <p className="dash-hint">
        This tool perturbs persisted tracking data and re-runs the same deterministic scorers over
        the perturbed input. It does not learn, does not model an opponent&apos;s response, and does
        not predict a result. Read the deltas as “what this metric would have measured”, never as
        “what would have happened”.
      </p>
    </Slab>
  );
}

function InterventionRow({ index, item, teamOptions, playerOptions, canRemove, onChange, onRemove }) {
  const needs = KIND_MAP.get(item.kind)?.needs || [];

  return (
    <Slab className="dash-sim-row">
      <div className="dash-sim-row-head">
        <span className="feature-number">{String(index + 1).padStart(2, '0')}</span>
        {canRemove ? (
          <button type="button" className="dash-remove-btn" onClick={onRemove} aria-label="Remove intervention">
            ×
          </button>
        ) : null}
      </div>

      <div className="dash-field-grid dash-field-grid-2">
        <SelectField
          label="Kind"
          value={item.kind}
          onChange={(value) => onChange({ kind: value })}
          options={KINDS.map((kind) => ({ value: kind.value, label: kind.label }))}
        />

        {needs.includes('team') ? (
          <SelectField
            label="Team"
            value={item.team_id}
            onChange={(value) => onChange({ team_id: value })}
            options={teamOptions}
            hint="Sent as team_id."
          />
        ) : null}

        {needs.includes('pct') ? (
          <div className="dash-field">
            <label className="dash-label" htmlFor={`pct-${item.key}`}>
              Change (pct)
            </label>
            <div className="dash-scale">
              <input
                id={`pct-${item.key}`}
                className="dash-range"
                type="range"
                min={-50}
                max={50}
                step={1}
                value={item.pct}
                onChange={(event) => onChange({ pct: Number(event.target.value) })}
              />
              <output className="dash-scale-out">
                {item.pct > 0 ? '+' : ''}
                {item.pct}
                <span className="dash-scale-max">%</span>
              </output>
            </div>
          </div>
        ) : null}

        {needs.includes('player') ? (
          <PlayerPicker
            label="Player"
            value={item.player_id}
            options={playerOptions}
            onChange={(value) => onChange({ player_id: value })}
          />
        ) : null}

        {needs.includes('other') ? (
          <PlayerPicker
            label="Swap With"
            value={item.other_player_id}
            options={playerOptions}
            onChange={(value) => onChange({ other_player_id: value })}
          />
        ) : null}
      </div>
    </Slab>
  );
}

/**
 * A select when a roster exists, a number box when it does not. The roster
 * comes from PlayerMetric rows, which will not exist for a match whose
 * intelligence stage has not run -- but tracking rows (what the simulator
 * actually reads) may exist anyway, so typing an ID has to stay possible.
 */
function PlayerPicker({ label, value, options, onChange }) {
  const hasRoster = options.length > 1;
  if (hasRoster) {
    return (
      <SelectField
        label={label}
        value={value === null || value === undefined ? '' : String(value)}
        onChange={(next) => onChange(next === null ? null : Number(next))}
        options={options}
      />
    );
  }
  return (
    <NumberField
      label={label}
      value={value}
      min={0}
      step={1}
      nullable
      onChange={onChange}
      hint="No PlayerMetric roster for this match — enter a ByteTrack tracking ID."
    />
  );
}

/* ==================================================================== */

function SimulationResult({ data }) {
  return (
    <>
      <SubHeading>Result</SubHeading>

      <Slab className="dash-sim-provenance">
        <MethodStrip
          items={[
            { label: 'method', value: data.method, tone: 'warn' },
            { label: 'reinforcement learning', value: String(data.is_reinforcement_learning), tone: 'good' },
            {
              label: 'team assignment confidence',
              value:
                data.team_assignment_confidence === null || data.team_assignment_confidence === undefined
                  ? 'not persisted'
                  : data.team_assignment_confidence.toFixed(3),
              tone: data.team_assignment_confidence ? '' : 'warn',
            },
          ]}
        />

        {data.interventions?.length ? (
          <div className="dash-badge-row">
            {data.interventions.map((text, index) => (
              <Chip key={`${text}-${index}`} tone="accent">
                {text}
              </Chip>
            ))}
          </div>
        ) : null}
      </Slab>

      {data.caveats?.length ? (
        <Slab className="dash-caveats">
          <h4 className="dash-caveats-title">Caveats — read these first</h4>
          <ul>
            {data.caveats.map((caveat, index) => (
              <li key={index}>{caveat}</li>
            ))}
          </ul>
        </Slab>
      ) : null}

      {data.unavailable?.length ? (
        <Slab className="dash-unavailable">
          <h4 className="dash-caveats-title">Could not be simulated</h4>
          <ul>
            {data.unavailable.map((reason, index) => (
              <li key={index}>{reason}</li>
            ))}
          </ul>
        </Slab>
      ) : null}

      {data.metrics?.length ? (
        <PanelGrid columns={2}>
          {data.metrics.map((metric, index) => (
            <SimulatedMetricCard
              key={`${metric.metric_name}-${metric.team_id}-${index}`}
              number={String(index + 1).padStart(2, '0')}
              metric={metric}
            />
          ))}
        </PanelGrid>
      ) : (
        <EmptyState
          title="Simulation Requires a Completed Real Pipeline Run"
          message="The request succeeded but produced no simulated metrics — there were no persisted baseline metrics to perturb. The reasons above list exactly what was missing."
        />
      )}

      <SubHeading>Real Baseline Inputs</SubHeading>
      <RealInputs metrics={data.real_input_metrics} />
    </>
  );
}

function SimulatedMetricCard({ number, metric }) {
  const delta = metric.delta;
  const tone = delta === null || delta === undefined ? '' : delta > 0 ? 'good' : delta < 0 ? 'bad' : '';

  const label = String(metric.metric_name)
    .replace(/_score$/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase());

  return (
    <Panel number={number} title={label} subtitle={metric.team_id ? `team ${metric.team_id}` : undefined}>
      <div className="dash-delta-row">
        <div className="dash-delta-cell">
          <span className="dash-delta-key">baseline</span>
          <MetricValue value={metric.baseline_value} absentLabel="Not Available" />
        </div>
        <span className="dash-delta-arrow" aria-hidden="true">→</span>
        <div className="dash-delta-cell">
          <span className="dash-delta-key">simulated</span>
          <MetricValue value={metric.simulated_value} absentLabel="Not Available" />
        </div>
        <div className={`dash-delta-cell dash-delta-value ${tone ? `is-${tone}` : ''}`.trim()}>
          <span className="dash-delta-key">delta</span>
          {delta === null || delta === undefined ? (
            <span className="dash-absent">n/a</span>
          ) : (
            <span>
              {delta > 0 ? '+' : ''}
              {Number(delta).toFixed(3)}
            </span>
          )}
        </div>
      </div>

      <MethodStrip
        items={[
          { label: 'method', value: metric.method, tone: 'warn' },
          { label: 'confidence', value: metric.confidence },
          { label: 'parameter changed', value: metric.parameter_changed },
          { label: 'recomputed by', value: metric.recomputed_by },
          { label: 'derived from', value: metric.derived_from },
          { label: 'source method', value: metric.source_method },
          { label: 'source confidence', value: metric.source_confidence },
        ]}
      />

      <SubScores subScores={metric.baseline_sub_scores} label="Baseline sub-scores" />
      <SubScores subScores={metric.simulated_sub_scores} label="Simulated sub-scores" />
    </Panel>
  );
}

/**
 * The real, persisted metrics the simulation started from. Shown because a
 * delta is meaningless without the measurement it was computed against.
 */
function RealInputs({ metrics }) {
  if (!metrics || !metrics.length) {
    return (
      <Slab>
        <EmptyState
          title="No Baseline Metrics Persisted"
          message="No TeamMetric rows exist for this match, so the simulator had no measured starting point. Everything it could not compute is listed above."
        />
      </Slab>
    );
  }

  return (
    <Slab className="dash-table-wrap">
      <table className="dash-table">
        <thead>
          <tr>
            <th>metric</th>
            <th>team</th>
            <th>value</th>
            <th>method</th>
            <th>confidence</th>
            <th>n</th>
          </tr>
        </thead>
        <tbody>
          {metrics.map((metric) => (
            <tr key={metric.metric_id}>
              <td>{metric.metric_name}</td>
              <td>{metric.team_id}</td>
              <td>
                <MetricValue value={metric.value} absentLabel="Not Computed" />
              </td>
              <td>{metric.method}</td>
              <td>{metric.confidence}</td>
              <td>{metric.sample_size}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Slab>
  );
}
