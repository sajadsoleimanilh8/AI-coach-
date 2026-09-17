import { useState } from 'react';

import api from '../../api/client.js';
import { useAction, useAsync } from '../../hooks/useAsync.js';
import {
  Chip,
  Disclaimer,
  FactorList,
  MetaRow,
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
  SelectField,
  SubmitButton,
  TextField,
  ToggleField,
} from '../ui/Form.jsx';

/**
 * TAB 8 -- the 13-item pre-match mental readiness questionnaire.
 *
 * Grouped exactly as the schema groups it (focus / stress / confidence /
 * motivation / pressure response), because those groupings are what the
 * scoring engine reads, not a layout convenience.
 *
 * `cv_player_id` is a separate, explicitly opt-in field. It is a DIFFERENT ID
 * space from player_id -- a ByteTrack tracking ID versus the caller's own
 * external identifier -- and it is never inferred from player_id here, for
 * the same reason the backend refuses to infer it: guessing a mapping would
 * attribute one player's on-pitch data to another player's answers.
 */

const INITIAL = {
  concentration_level: 7,
  focus_maintenance: 7,
  mental_clarity_raw: 7,
  pre_match_stress: 4,
  importance_pressure: 5,
  nervousness: 4,
  performance_confidence: 7,
  tactical_confidence_raw: 7,
  match_motivation: 8,
  competitive_motivation_raw: 8,
  mistake_recovery_speed: 6,
  pressure_performance_effect: 'no_change',
  post_error_calm: 6,
};

const PRESSURE_EFFECT_OPTIONS = [
  { value: 'improves', label: 'Improves my performance' },
  { value: 'no_change', label: 'No change' },
  { value: 'reduces', label: 'Reduces my performance' },
];

export default function TabPsychology({ matchId }) {
  const [playerId, setPlayerId] = useState('');
  const [attachMatch, setAttachMatch] = useState(true);
  const [linkCv, setLinkCv] = useState(false);
  const [cvPlayerId, setCvPlayerId] = useState(null);
  const [form, setForm] = useState(INITIAL);
  const [historyPlayer, setHistoryPlayer] = useState(null);

  const submit = useAction((payload, signal) =>
    api.submitPsychology(payload.playerId, payload.body, signal),
  );

  const set = (patch) => setForm((current) => ({ ...current, ...patch }));

  const onSubmit = async (event) => {
    event.preventDefault();
    const id = playerId.trim();
    if (!id) return;

    const body = { ...form };
    body.match_id = attachMatch && matchId ? matchId : null;
    body.cv_player_id = linkCv && cvPlayerId !== null && cvPlayerId !== '' ? Number(cvPlayerId) : null;

    const result = await submit.run({ playerId: id, body });
    if (result) setHistoryPlayer(id);
  };

  return (
    <TabSection animKey="psychology">
      <SectionHeader
        tag="Pre-Match Psychology"
        title={<>Mental<br />Readiness</>}
        meta="Thirteen self-reported items, scored server-side. A mental-readiness estimate — not emotion detection, not a clinical assessment."
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
                hint="Your own external identifier — sent in the path."
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
                    No match in this session — a genuine pre-match submission has none yet.
                  </p>
                )}
              </div>

              <div className="dash-field is-wide">
                <ToggleField
                  label="Link to CV tracking history (opt-in)"
                  value={linkCv}
                  onChange={setLinkCv}
                  hint="cv_player_id is a ByteTrack tracking ID — a different ID space from the player ID above. It is never inferred, only stated."
                />
                {linkCv ? (
                  <NumberField
                    label="CV Player ID"
                    value={cvPlayerId}
                    min={0}
                    step={1}
                    nullable
                    onChange={setCvPlayerId}
                    hint="Supplying this is what makes historical_context non-empty — and only when matching tracking rows actually exist."
                  />
                ) : null}
              </div>
            </FormSection>

            <FormSection title="Focus" columns={2}>
              <ScaleField
                label="Concentration Level"
                value={form.concentration_level}
                onChange={(value) => set({ concentration_level: value })}
                lowLabel="Scattered"
                highLabel="Locked in"
              />
              <ScaleField
                label="Focus Maintenance"
                value={form.focus_maintenance}
                onChange={(value) => set({ focus_maintenance: value })}
                lowLabel="Drifts fast"
                highLabel="Sustained"
              />
              <ScaleField
                label="Mental Clarity"
                value={form.mental_clarity_raw}
                onChange={(value) => set({ mental_clarity_raw: value })}
                lowLabel="Foggy"
                highLabel="Sharp"
              />
            </FormSection>

            <FormSection title="Stress" hint="Higher answers here mean more reported stress." columns={2}>
              <ScaleField
                label="Pre-Match Stress"
                value={form.pre_match_stress}
                onChange={(value) => set({ pre_match_stress: value })}
                lowLabel="Calm"
                highLabel="Very stressed"
              />
              <ScaleField
                label="Importance Pressure"
                value={form.importance_pressure}
                onChange={(value) => set({ importance_pressure: value })}
                lowLabel="Routine game"
                highLabel="Must win"
              />
              <ScaleField
                label="Nervousness"
                value={form.nervousness}
                onChange={(value) => set({ nervousness: value })}
                lowLabel="Settled"
                highLabel="Very nervous"
              />
            </FormSection>

            <FormSection title="Confidence" columns={2}>
              <ScaleField
                label="Performance Confidence"
                value={form.performance_confidence}
                onChange={(value) => set({ performance_confidence: value })}
                lowLabel="Doubtful"
                highLabel="Assured"
              />
              <ScaleField
                label="Tactical Confidence"
                value={form.tactical_confidence_raw}
                onChange={(value) => set({ tactical_confidence_raw: value })}
                lowLabel="Unclear on the plan"
                highLabel="Know the plan"
              />
            </FormSection>

            <FormSection title="Motivation" columns={2}>
              <ScaleField
                label="Match Motivation"
                value={form.match_motivation}
                onChange={(value) => set({ match_motivation: value })}
                lowLabel="Flat"
                highLabel="Fired up"
              />
              <ScaleField
                label="Competitive Motivation"
                value={form.competitive_motivation_raw}
                onChange={(value) => set({ competitive_motivation_raw: value })}
                lowLabel="Indifferent"
                highLabel="Driven to win"
              />
            </FormSection>

            <FormSection title="Pressure Response" columns={2}>
              <ScaleField
                label="Mistake Recovery Speed"
                value={form.mistake_recovery_speed}
                onChange={(value) => set({ mistake_recovery_speed: value })}
                lowLabel="Dwell on it"
                highLabel="Instant reset"
              />
              <ScaleField
                label="Post-Error Calm"
                value={form.post_error_calm}
                onChange={(value) => set({ post_error_calm: value })}
                lowLabel="Rattled"
                highLabel="Unshaken"
              />
              <SelectField
                label="Effect of Pressure"
                value={form.pressure_performance_effect}
                onChange={(value) => set({ pressure_performance_effect: value || 'no_change' })}
                options={PRESSURE_EFFECT_OPTIONS}
                wide
                hint="The one categorical item — an unrecognised value is rejected rather than interpreted."
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
      <HistoryPanel playerId={historyPlayer || playerId.trim() || null} matchId={matchId} />
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
          message="Answer the thirteen items and submit. Every score is computed by the backend's deterministic engine."
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
      </Panel>
    );
  }
  return <PsychAssessmentView assessment={state.data} />;
}

export function PsychAssessmentView({ assessment }) {
  const factorEntries = assessment.factors ? Object.entries(assessment.factors) : [];

  return (
    <Panel
      number="—"
      title="Mental Readiness Assessment"
      subtitle={`Submission #${assessment.submission_index} · player ${assessment.player_id}`}
    >
      <StatRow>
        <StatNum value={assessment.mental_readiness} label="Mental Readiness" />
        <StatNum value={assessment.focus} label="Focus" />
        <StatNum value={assessment.confidence} label="Confidence" />
        <StatNum value={assessment.stress} label="Stress · higher is worse" tone="warn" />
      </StatRow>

      <p className="dash-hint">
        All four are 0–100 integers. Mental readiness, focus and confidence are higher-is-better;
        stress runs the other way, which is why it is labelled rather than left to read as a fourth
        score of the same kind.
      </p>

      <div className="dash-badge-row">
        <RiskBadge label="pressure" risk={assessment.pressure_risk} />
        <RiskBadge label="mental performance" risk={assessment.mental_performance_risk} />
        {assessment.match_id ? <Chip tone="accent">match {assessment.match_id}</Chip> : null}
        {assessment.cv_player_id !== null && assessment.cv_player_id !== undefined ? (
          <Chip tone="accent">cv_player_id {assessment.cv_player_id}</Chip>
        ) : null}
      </div>

      <MethodStrip
        items={[
          { label: 'method', value: assessment.method },
          { label: 'confidence level', value: assessment.confidence_level },
          { label: 'n', value: assessment.sample_size },
          { label: 'data source', value: assessment.data_source },
          { label: 'computed', value: formatTime(assessment.computed_at) },
        ]}
      />

      <div className="dash-factor-cols">
        <FactorList
          title="Key Positive Factors"
          items={assessment.key_positive_factors}
          tone="good"
          emptyLabel="No item scored as a clear positive."
        />
        <FactorList
          title="Key Negative Factors"
          items={assessment.key_negative_factors}
          tone="bad"
          emptyLabel="No item scored as a clear negative."
        />
      </div>

      {factorEntries.length ? (
        <>
          <h4 className="dash-card-title dash-factors-heading">Dimension Classification</h4>
          <div className="dash-dimension-grid">
            {factorEntries.map(([dimension, label]) => (
              <span key={dimension} className={`dash-dimension is-${label}`}>
                <span className="dash-dimension-name">{dimension.replace(/_/g, ' ')}</span>
                <span className="dash-dimension-label">{label}</span>
              </span>
            ))}
          </div>
        </>
      ) : null}

      <div className="dash-historical">
        <h4 className="dash-card-title dash-factors-heading">CV Corroboration</h4>
        {assessment.historical_context?.length ? (
          <ul className="dash-list">
            {assessment.historical_context.map((line, index) => (
              <li key={index}>{line}</li>
            ))}
          </ul>
        ) : (
          <p className="dash-absent">
            {assessment.cv_player_id === null || assessment.cv_player_id === undefined
              ? 'No cv_player_id was supplied, so this self-report is not cross-referenced against any tracking data.'
              : 'A cv_player_id was supplied but no matching tracking metrics were found, so there is nothing to corroborate against.'}
          </p>
        )}
      </div>

      <Disclaimer>{assessment.disclaimer}</Disclaimer>
    </Panel>
  );
}

/* ==================================================================== */

function HistoryPanel({ playerId, matchId }) {
  const [limit, setLimit] = useState(10);
  const [scopeToMatch, setScopeToMatch] = useState(false);
  const scopedMatch = scopeToMatch && matchId ? matchId : undefined;

  const latest = useAsync(
    (signal) => api.getPsychologyLatest(playerId, scopedMatch, signal),
    [playerId, scopedMatch],
    { enabled: Boolean(playerId) },
  );

  const history = useAsync(
    (signal) => api.getPsychologyHistory(playerId, { limit, matchId: scopedMatch }, signal),
    [playerId, limit, scopedMatch],
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
      <Panel number="01" title="Latest On Record" subtitle={`GET /api/psychology/${playerId}/latest`}>
        {matchId ? (
          <ToggleField
            label={scopeToMatch ? `Filtered to match ${matchId}` : 'All matches'}
            value={scopeToMatch}
            onChange={setScopeToMatch}
          />
        ) : null}

        {latest.status === 'error' && latest.error?.notFound ? (
          <EmptyState title="No Assessment Yet" message={latest.error.message} />
        ) : (
          <AsyncBlock state={latest} loadingLabel="Loading latest assessment" loadingRows={2}>
            {(data) => (
              <>
                <StatRow>
                  <StatNum value={data.mental_readiness} label="Mental Readiness" />
                  <StatNum value={data.stress} label="Stress · higher is worse" tone="warn" />
                </StatRow>
                <div className="dash-badge-row">
                  <RiskBadge label="pressure" risk={data.pressure_risk} />
                  <RiskBadge label="mental performance" risk={data.mental_performance_risk} />
                </div>
                <MetaRow label="computed">{formatTime(data.computed_at)}</MetaRow>
                <Disclaimer>{data.disclaimer}</Disclaimer>
              </>
            )}
          </AsyncBlock>
        )}
      </Panel>

      <Panel number="02" title="History" subtitle={`GET /api/psychology/${playerId}/history?limit=${limit}`}>
        <div className="dash-inline-controls">
          <NumberField label="Limit" value={limit} min={1} max={200} step={5} onChange={setLimit} />
        </div>
        <AsyncBlock
          state={history}
          loadingLabel="Loading history"
          loadingRows={3}
          isEmpty={(data) => !Array.isArray(data) || data.length === 0}
          emptyTitle="No History"
          emptyMessage="This player has submitted no psychology questionnaires."
        >
          {(rows) => (
            <div className="dash-table-wrap">
              <table className="dash-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>readiness</th>
                    <th>focus</th>
                    <th>confidence</th>
                    <th>stress</th>
                    <th>pressure risk</th>
                    <th>computed</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={row.assessment_id}>
                      <td>{row.submission_index}</td>
                      <td>{row.mental_readiness}</td>
                      <td>{row.focus}</td>
                      <td>{row.confidence}</td>
                      <td>{row.stress}</td>
                      <td>
                        <RiskBadge risk={row.pressure_risk} />
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
