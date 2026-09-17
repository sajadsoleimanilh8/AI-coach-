import { useState } from 'react';

import api from '../../api/client.js';
import { useAction, useAsync } from '../../hooks/useAsync.js';
import {
  Chip,
  Disclaimer,
  FactorList,
  MetaRow,
  Meter,
  MethodStrip,
  Panel,
  PanelGrid,
  RiskBadge,
  SectionHeader,
  Slab,
  StatNum,
  StatRow,
  SubHeading,
  TabSection,
} from '../ui/Primitives.jsx';
import { AsyncBlock, EmptyState, ErrorState, LoadingState } from '../ui/States.jsx';
import {
  FormSection,
  NumberField,
  ScaleField,
  SubmitButton,
  TextAreaField,
  TextField,
  ToggleField,
} from '../ui/Form.jsx';

/**
 * TAB 6 -- the pre-match physical readiness questionnaire.
 *
 * Every field of PreMatchQuestionnaireRequest is here, with the client-side
 * ranges matching the server's Field(ge=/le=) constraints exactly. The point
 * of matching them is that an out-of-range answer is a 422 by design (never
 * silently clamped), so the form should not be able to produce one by
 * accident -- and when the server does reject something, its `detail` is
 * rendered verbatim rather than replaced with a generic message.
 *
 * `disclaimer` from the response is always rendered. This is a
 * performance-readiness estimate from a self-report; it is not a medical
 * assessment.
 */

const INITIAL = {
  sleep_duration_hours: 8,
  sleep_quality: 7,
  bedtime: null,
  wake_time: null,
  night_awakenings: 0,
  trained_last_24h: false,
  trained_last_48h: false,
  training_duration_minutes: 0,
  training_intensity: 1,
  high_intensity_activity: false,
  hours_since_last_training: null,
  fatigue: 3,
  muscle_soreness: 3,
  pain_level: 1,
  perceived_readiness: 7,
  hydration_liters: 2,
  nutrition_quality: 7,
  hours_since_last_meal: 3,
  caffeine_or_supplement_notes: null,
};

export default function TabPreMatchHealth({ matchId }) {
  const [playerId, setPlayerId] = useState('');
  const [attachMatch, setAttachMatch] = useState(true);
  const [form, setForm] = useState(INITIAL);
  const [historyPlayer, setHistoryPlayer] = useState(null);

  const submit = useAction((payload, signal) =>
    api.submitPreMatchHealth(payload.playerId, payload.body, signal),
  );

  const set = (patch) => setForm((current) => ({ ...current, ...patch }));

  const onSubmit = async (event) => {
    event.preventDefault();
    const id = playerId.trim();
    if (!id) return;

    const body = { ...form };
    // Only sent when the user opted in and a match actually exists: the
    // endpoint refuses a dangling match reference with a 404, and a genuine
    // pre-match submission legitimately has no match yet.
    body.match_id = attachMatch && matchId ? matchId : null;

    const result = await submit.run({ playerId: id, body });
    if (result) setHistoryPlayer(id);
  };

  return (
    <TabSection animKey="health">
      <SectionHeader
        tag="Pre-Match Health"
        title={<>Physical<br />Readiness</>}
        meta="A self-reported questionnaire, scored server-side by a deterministic engine. Not a medical assessment."
      />

      <PanelGrid columns={2}>
        <div className="dash-form-col" data-anim>
          <form className="dash-form" onSubmit={onSubmit}>
            <FormSection title="Identity" columns={2}>
              <TextField
                label="Player ID"
                value={playerId}
                onChange={(value) => setPlayerId(value ?? '')}
                placeholder="e.g. 10 or jsmith"
                hint="Your own external identifier — sent in the path, not the body."
              />
              <div className="dash-field">
                <span className="dash-label">Attach to Match</span>
                {matchId ? (
                  <ToggleField
                    label={attachMatch ? `Linked to ${matchId}` : 'Not linked'}
                    value={attachMatch}
                    onChange={setAttachMatch}
                  />
                ) : (
                  <p className="dash-hint dash-absent">
                    No match in this session. A genuine pre-match submission has none yet — that is a
                    valid state, not a missing field.
                  </p>
                )}
              </div>
            </FormSection>

            <FormSection title="Sleep" columns={2}>
              <NumberField
                label="Sleep Duration"
                value={form.sleep_duration_hours}
                min={0.5}
                max={24}
                step={0.5}
                suffix="hrs"
                onChange={(value) => set({ sleep_duration_hours: value })}
                hint="Must be > 0 and ≤ 24."
              />
              <ScaleField
                label="Sleep Quality"
                value={form.sleep_quality}
                onChange={(value) => set({ sleep_quality: value })}
                lowLabel="Poor"
                highLabel="Excellent"
              />
              <TextField
                label="Bedtime (optional)"
                type="time"
                value={form.bedtime}
                onChange={(value) => set({ bedtime: value })}
              />
              <TextField
                label="Wake Time (optional)"
                type="time"
                value={form.wake_time}
                onChange={(value) => set({ wake_time: value })}
              />
              <NumberField
                label="Night Awakenings"
                value={form.night_awakenings}
                min={0}
                step={1}
                onChange={(value) => set({ night_awakenings: value })}
              />
            </FormSection>

            <FormSection title="Physical Activity" columns={2}>
              <ToggleField
                label="Trained in last 24h"
                value={form.trained_last_24h}
                onChange={(value) => set({ trained_last_24h: value })}
              />
              <ToggleField
                label="Trained in last 48h"
                value={form.trained_last_48h}
                onChange={(value) => set({ trained_last_48h: value })}
              />
              <NumberField
                label="Training Duration"
                value={form.training_duration_minutes}
                min={0}
                step={5}
                suffix="min"
                onChange={(value) => set({ training_duration_minutes: value })}
              />
              <ScaleField
                label="Training Intensity"
                value={form.training_intensity}
                onChange={(value) => set({ training_intensity: value })}
                lowLabel="Light"
                highLabel="Maximal"
              />
              <ToggleField
                label="High-intensity activity"
                value={form.high_intensity_activity}
                onChange={(value) => set({ high_intensity_activity: value })}
              />
              <NumberField
                label="Hours Since Last Training"
                value={form.hours_since_last_training}
                min={0}
                step={1}
                suffix="hrs"
                nullable
                placeholder="leave blank if none"
                onChange={(value) => set({ hours_since_last_training: value })}
                hint="Blank is a real answer — “no recent training to date from”. It is not stored as 0."
              />
            </FormSection>

            <FormSection title="Recovery" columns={2}>
              <ScaleField
                label="Fatigue"
                value={form.fatigue}
                onChange={(value) => set({ fatigue: value })}
                lowLabel="Fresh"
                highLabel="Exhausted"
              />
              <ScaleField
                label="Muscle Soreness"
                value={form.muscle_soreness}
                onChange={(value) => set({ muscle_soreness: value })}
                lowLabel="None"
                highLabel="Severe"
              />
              <ScaleField
                label="Pain Level"
                value={form.pain_level}
                onChange={(value) => set({ pain_level: value })}
                lowLabel="None"
                highLabel="Severe"
              />
              <ScaleField
                label="Perceived Readiness"
                value={form.perceived_readiness}
                onChange={(value) => set({ perceived_readiness: value })}
                lowLabel="Not ready"
                highLabel="Fully ready"
              />
            </FormSection>

            <FormSection title="Hydration & Nutrition" columns={2}>
              <NumberField
                label="Hydration"
                value={form.hydration_liters}
                min={0}
                step={0.1}
                suffix="L"
                onChange={(value) => set({ hydration_liters: value })}
              />
              <ScaleField
                label="Nutrition Quality"
                value={form.nutrition_quality}
                onChange={(value) => set({ nutrition_quality: value })}
                lowLabel="Poor"
                highLabel="Excellent"
              />
              <NumberField
                label="Hours Since Last Meal"
                value={form.hours_since_last_meal}
                min={0}
                step={0.5}
                suffix="hrs"
                onChange={(value) => set({ hours_since_last_meal: value })}
              />
              <TextAreaField
                label="Caffeine / Supplement Notes"
                value={form.caffeine_or_supplement_notes}
                onChange={(value) => set({ caffeine_or_supplement_notes: value })}
                placeholder="Free text — echoed back for a coach to read."
                hint="Never fed into any formula. It is attached to the assessment after scoring has finished."
              />
            </FormSection>

            <div className="dash-form-actions">
              <SubmitButton
                busy={submit.status === 'loading'}
                busyLabel="Scoring"
                disabled={!playerId.trim()}
              >
                Submit Questionnaire
              </SubmitButton>
              <button
                type="button"
                className="btn-ghost"
                onClick={() => {
                  setForm(INITIAL);
                  submit.reset();
                }}
              >
                Reset
              </button>
            </div>
          </form>
        </div>

        <div className="dash-result-col" data-anim>
          <ResultPanel state={submit} />
        </div>
      </PanelGrid>

      <SubHeading>Prior Assessments</SubHeading>
      <HistoryPanel playerId={historyPlayer || playerId.trim() || null} />
    </TabSection>
  );
}

/* ==================================================================== */

function ResultPanel({ state }) {
  if (state.status === 'idle') {
    return (
      <Panel number="—" title="Assessment" subtitle="Results appear here after submission.">
        <EmptyState
          title="Not Yet Submitted"
          message="Fill in the questionnaire and submit. Every score below is computed by the backend's deterministic engine — nothing is estimated in the browser."
        />
      </Panel>
    );
  }

  if (state.status === 'loading') {
    return (
      <Panel number="—" title="Assessment">
        <LoadingState label="Scoring questionnaire" rows={3} />
      </Panel>
    );
  }

  if (state.status === 'error') {
    return (
      <Panel number="—" title="Assessment" tone="bad">
        <ErrorState error={state.error} title="Submission Rejected" />
        {state.error?.status === 422 ? (
          <p className="dash-hint">
            The server validates every range independently and rejects out-of-range answers rather
            than clamping them, so a data-entry mistake can never become a real-looking reading.
          </p>
        ) : null}
      </Panel>
    );
  }

  return <AssessmentView assessment={state.data} />;
}

export function AssessmentView({ assessment }) {
  return (
    <Panel
      number="—"
      title="Physical Readiness Assessment"
      subtitle={`Submission #${assessment.submission_index} · player ${assessment.player_id}`}
    >
      <StatRow>
        <StatNum value={assessment.physical_readiness} label="Physical Readiness" />
        <StatNum value={assessment.fatigue_score} label="Fatigue Score" />
        <StatNum value={assessment.recovery_score} label="Recovery Score" />
      </StatRow>

      <div className="dash-badge-row">
        <RiskBadge label="performance" risk={assessment.performance_risk} />
        <RiskBadge label="workload" risk={assessment.workload_risk} />
        {assessment.match_id ? <Chip tone="accent">match {assessment.match_id}</Chip> : null}
      </div>

      <MethodStrip
        items={[
          { label: 'method', value: assessment.method },
          { label: 'data source', value: assessment.data_source },
          { label: 'schema', value: assessment.schema_version },
          { label: 'computed', value: formatTime(assessment.computed_at) },
        ]}
      />

      <div className="dash-factor-cols">
        <FactorList
          title="Key Positive Factors"
          items={assessment.key_positive_factors}
          tone="good"
          emptyLabel="No factor scored as a clear positive."
        />
        <FactorList
          title="Key Negative Factors"
          items={assessment.key_negative_factors}
          tone="bad"
          emptyLabel="No factor scored as a clear negative."
        />
      </div>

      {assessment.factors?.length ? (
        <>
          <h4 className="dash-card-title dash-factors-heading">Full Factor Breakdown</h4>
          <div className="dash-factor-table">
            {assessment.factors.map((factor) => (
              <div key={factor.dimension} className={`dash-factor-row is-${factor.label}`}>
                <span className="dash-factor-dim">{factor.dimension.replace(/_/g, ' ')}</span>
                <Chip tone={factor.label === 'positive' ? 'good' : factor.label === 'negative' ? 'bad' : 'neutral'}>
                  {factor.label}
                </Chip>
                <Meter value={factor.score} tone={factor.label === 'negative' ? 'pink' : 'blue'} label={factor.dimension} />
                <span className="dash-factor-detail">{factor.detail}</span>
              </div>
            ))}
          </div>
          <p className="dash-hint">
            Every factor score is normalised so higher is better, regardless of which way the raw
            input pointed.
          </p>
        </>
      ) : null}

      {assessment.notes ? (
        <div className="dash-meta-block">
          <MetaRow label="notes">{assessment.notes}</MetaRow>
        </div>
      ) : null}

      <Disclaimer>{assessment.disclaimer}</Disclaimer>
    </Panel>
  );
}

/* ==================================================================== */

function HistoryPanel({ playerId }) {
  const [limit, setLimit] = useState(10);

  const latest = useAsync(
    (signal) => api.getPreMatchHealthLatest(playerId, signal),
    [playerId],
    { enabled: Boolean(playerId) },
  );

  const history = useAsync(
    (signal) => api.getPreMatchHealthHistory(playerId, limit, signal),
    [playerId, limit],
    { enabled: Boolean(playerId) },
  );

  if (!playerId) {
    return (
      <Slab>
        <EmptyState
          title="No Player ID"
          message="Enter a player ID above to load that player's latest assessment and history."
        />
      </Slab>
    );
  }

  return (
    <PanelGrid columns={2}>
      <Panel number="01" title="Latest On Record" subtitle={`GET /api/prematch_health/${playerId}/latest`}>
        {latest.status === 'error' && latest.error?.notFound ? (
          <EmptyState
            title="No Assessment Yet"
            message={latest.error.message}
          />
        ) : (
          <AsyncBlock state={latest} loadingLabel="Loading latest assessment" loadingRows={2}>
            {(data) => (
              <>
                <StatRow>
                  <StatNum value={data.physical_readiness} label="Physical Readiness" />
                  <StatNum value={data.recovery_score} label="Recovery" />
                </StatRow>
                <div className="dash-badge-row">
                  <RiskBadge label="performance" risk={data.performance_risk} />
                  <RiskBadge label="workload" risk={data.workload_risk} />
                </div>
                <MetaRow label="computed">{formatTime(data.computed_at)}</MetaRow>
                <Disclaimer>{data.disclaimer}</Disclaimer>
              </>
            )}
          </AsyncBlock>
        )}
      </Panel>

      <Panel
        number="02"
        title="History"
        subtitle={`GET /api/prematch_health/${playerId}/history?limit=${limit}`}
      >
        <div className="dash-inline-controls">
          <NumberField label="Limit" value={limit} min={1} max={200} step={5} onChange={setLimit} />
        </div>
        <AsyncBlock
          state={history}
          loadingLabel="Loading history"
          loadingRows={3}
          isEmpty={(data) => !Array.isArray(data) || data.length === 0}
          emptyTitle="No History"
          emptyMessage="This player has submitted no questionnaires. An empty list is the complete, true answer here."
        >
          {(rows) => (
            <div className="dash-table-wrap">
              <table className="dash-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>readiness</th>
                    <th>fatigue</th>
                    <th>recovery</th>
                    <th>perf risk</th>
                    <th>computed</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={row.assessment_id}>
                      <td>{row.submission_index}</td>
                      <td>{row.physical_readiness?.toFixed(1)}</td>
                      <td>{row.fatigue_score?.toFixed(1)}</td>
                      <td>{row.recovery_score?.toFixed(1)}</td>
                      <td>
                        <RiskBadge risk={row.performance_risk} />
                      </td>
                      <td>{formatTime(row.computed_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </AsyncBlock>
      </Panel>
    </PanelGrid>
  );
}

function formatTime(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString();
}
