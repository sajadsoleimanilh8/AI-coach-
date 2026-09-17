import { useEffect, useMemo, useState } from 'react';

import api from '../../api/client.js';
import { useAsync } from '../../hooks/useAsync.js';
import {
  Chip,
  ConfidenceBadge,
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
import { AsyncBlock, EmptyState } from '../ui/States.jsx';
import { SelectMatchState } from '../ui/MatchPicker.jsx';

/**
 * TAB 3 -- team-level tactical output: detected formation, team shape over
 * time, and the general team-intelligence feed from ai/team_intelligence/*.
 *
 * All three sections stay on screen even when empty. The formation endpoint
 * in particular returns a 404 rather than a plausible-looking fallback
 * (backend/api/tactical.py deliberately stopped fabricating "4-3-3"), and
 * this tab renders that 404 as the stated empty state it is.
 */
export default function TabTeamIntelligence({ matchId, goToTab, onAttachMatch }) {
  const [teamId, setTeamId] = useState('');

  // The general feed is not team-scoped and returns every team's rows, which
  // makes it the honest source for "which team ids does this match actually
  // have?" -- far better than assuming a naming convention.
  const teamIntel = useAsync(
    (signal) => api.getTeamIntelligence(matchId, signal),
    [matchId],
    { enabled: Boolean(matchId) },
  );

  const availableTeams = useMemo(() => {
    if (!Array.isArray(teamIntel.data)) return [];
    return [...new Set(teamIntel.data.map((metric) => metric.team_id).filter(Boolean))];
  }, [teamIntel.data]);

  // Adopt the first real team id once it is known, instead of leaving the
  // formation call pointed at the endpoint's "unassigned" default.
  useEffect(() => {
    if (!teamId && availableTeams.length) setTeamId(availableTeams[0]);
  }, [availableTeams, teamId]);

  useEffect(() => {
    setTeamId('');
  }, [matchId]);

  const scopedReady = Boolean(matchId) && Boolean(teamId);

  const formation = useAsync(
    (signal) => api.getFormation(matchId, teamId, signal),
    [matchId, teamId],
    { enabled: scopedReady },
  );

  const teamShape = useAsync(
    (signal) => api.getTeamShape(matchId, teamId, signal),
    [matchId, teamId],
    { enabled: scopedReady },
  );

  return (
    <TabSection animKey={`team-${matchId}`}>
      <SectionHeader
        tag="Team Intelligence"
        title={<>Shape &amp;<br />Structure</>}
        meta="Formation, compactness and width over time, and the possession / pressing / transition / chemistry feed."
      />

      {!matchId ? (
        <SelectMatchState onGoToUpload={() => goToTab('match')} onAttachMatch={onAttachMatch} />
      ) : (
        <>
          {availableTeams.length ? (
            <Slab>
              <span className="dash-label">Team Scope</span>
              <div className="dash-quickpick-row">
                {availableTeams.map((id) => (
                  <button
                    key={id}
                    type="button"
                    className={`dash-pill ${teamId === id ? 'is-active' : ''}`.trim()}
                    onClick={() => setTeamId(id)}
                  >
                    {id}
                  </button>
                ))}
              </div>
              <p className="dash-hint">
                Formation and team shape are served per team. These are the team ids this match
                actually has, read from its own metric rows.
              </p>
            </Slab>
          ) : null}

          <SubHeading>Detected Formation</SubHeading>
          {scopedReady ? (
            <FormationPanel state={formation} />
          ) : (
            <Slab>
              <EmptyState
                title="No Team Scope Yet"
                message="No team ids have been read for this match, so there is nothing to scope a formation query to."
              />
            </Slab>
          )}

          <SubHeading>Team Shape Over Time</SubHeading>
          {scopedReady ? (
            <TeamShapePanel state={teamShape} />
          ) : (
            <Slab>
              <EmptyState
                title="No Team Scope Yet"
                message="Team shape is served per team id."
              />
            </Slab>
          )}

          <SubHeading>Team Intelligence Feed</SubHeading>
          <TeamIntelPanel state={teamIntel} />
        </>
      )}
    </TabSection>
  );
}

/* ==================================================================== */

function FormationPanel({ state }) {
  // 404 is the endpoint's honest "not computed yet", not a failure.
  if (state.status === 'error' && state.error?.notFound) {
    return (
      <Slab>
        <EmptyState
          title="Formation Data Not Detected"
          message={state.error.message}
          action={
            <button type="button" className="btn-ghost" onClick={state.reload}>
              Check Again
            </button>
          }
        />
      </Slab>
    );
  }

  return (
    <AsyncBlock state={state} loadingLabel="Reading detected formation" loadingRows={2}>
      {(metric) => {
        const hasValue = metric.value !== null && metric.value !== undefined && metric.value !== '';
        return (
          <PanelGrid columns={2}>
            <Panel number="01" title="Formation" subtitle="GET /api/tactical/formation/{match_id}">
              {hasValue ? (
                <span className="stat-num dash-formation-value">{String(metric.value)}</span>
              ) : (
                <p className="dash-absent dash-metric-absent">Formation Data Not Detected</p>
              )}
              <div className="dash-badge-row">
                <ConfidenceBadge
                  confidence={metric.confidence}
                  score={metric.confidence_score}
                />
                <Chip>team {metric.team_id}</Chip>
              </div>
              <MethodStrip
                items={[
                  { label: 'method', value: metric.method },
                  { label: 'n', value: metric.sample_size },
                  { label: 'schema', value: metric.schema_version },
                ]}
              />
            </Panel>

            <Panel number="02" title="Detection Record" subtitle="The reasoning behind the label above.">
              <SubScores subScores={metric.sub_scores} />
              <div className="dash-meta-block">
                <MetaRow label="metric_id">
                  <code className="dash-code-sm">{metric.metric_id}</code>
                </MetaRow>
                <MetaRow label="computed">{formatTime(metric.computed_at)}</MetaRow>
              </div>
            </Panel>
          </PanelGrid>
        );
      }}
    </AsyncBlock>
  );
}

/* ==================================================================== */

function TeamShapePanel({ state }) {
  return (
    <AsyncBlock
      state={state}
      loadingLabel="Reading team shape series"
      isEmpty={(data) => !Array.isArray(data) || data.length === 0}
      emptyTitle="No Team Shape Metrics"
      emptyMessage="No compactness or width rows were written for this match. Team shape is computed during a processing run; until one completes there is nothing to plot."
    >
      {(metrics) => <TeamMetricGrid metrics={metrics} startNumber={1} />}
    </AsyncBlock>
  );
}

function TeamIntelPanel({ state }) {
  return (
    <AsyncBlock
      state={state}
      loadingLabel="Reading team intelligence feed"
      isEmpty={(data) => !Array.isArray(data) || data.length === 0}
      emptyTitle="No Team Intelligence Metrics"
      emptyMessage="No TeamMetric rows exist for this match yet — possession, pressing structure, transition and chemistry all appear here once a processing job completes."
    >
      {(metrics) => <GroupedTeamMetrics metrics={metrics} />}
    </AsyncBlock>
  );
}

/**
 * The team-intelligence feed mixes several teams into one flat list, so it is
 * grouped by team_id rather than shown as an undifferentiated pile of cards
 * where "compactness_score" appears twice with no way to tell whose it is.
 */
function GroupedTeamMetrics({ metrics }) {
  const groups = useMemo(() => {
    const map = new Map();
    metrics.forEach((metric) => {
      const key = metric.team_id ?? 'unassigned';
      if (!map.has(key)) map.set(key, []);
      map.get(key).push(metric);
    });
    return [...map.entries()];
  }, [metrics]);

  return (
    <>
      {groups.map(([teamId, teamMetrics]) => (
        <div key={teamId} className="dash-team-group">
          <div className="dash-team-group-head" data-anim>
            <Chip tone="accent">team {teamId}</Chip>
            <span className="dash-hint">{teamMetrics.length} metrics</span>
          </div>
          <TeamMetricGrid metrics={teamMetrics} startNumber={1} />
        </div>
      ))}
    </>
  );
}

function TeamMetricGrid({ metrics, startNumber = 1 }) {
  return (
    <PanelGrid columns={3}>
      {metrics.map((metric, index) => (
        <TeamMetricCard
          key={metric.metric_id}
          number={String(startNumber + index).padStart(2, '0')}
          metric={metric}
        />
      ))}
    </PanelGrid>
  );
}

function TeamMetricCard({ number, metric }) {
  const label = String(metric.metric_name)
    .replace(/_score$/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase());

  const hasValue = metric.value !== null && metric.value !== undefined && metric.value !== '';
  // TeamMetric.value resolves to either a number or a label string
  // (value_numeric / value_label), so both shapes have to render sensibly.
  const isNumeric = typeof metric.value === 'number';

  return (
    <Panel number={number} title={label} tone={hasValue ? '' : 'absent'}>
      {hasValue ? (
        <span className={`stat-num ${isNumeric ? 'dash-metric-headline' : 'dash-formation-value'}`}>
          {isNumeric ? metric.value.toFixed(1) : String(metric.value)}
        </span>
      ) : (
        <p className="dash-absent dash-metric-absent">Not Yet Computed</p>
      )}

      <div className="dash-badge-row">
        <ConfidenceBadge confidence={metric.confidence} score={metric.confidence_score} />
        <Chip>team {metric.team_id}</Chip>
      </div>

      <MethodStrip
        items={[
          { label: 'method', value: metric.method },
          { label: 'n', value: metric.sample_size },
        ]}
      />

      <SubScores subScores={metric.sub_scores} />

      <div className="dash-meta-block">
        <MetaRow label="computed">
          <MetricValue value={formatTime(metric.computed_at)} absentLabel="Not Recorded" />
        </MetaRow>
      </div>
    </Panel>
  );
}

function formatTime(value) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.toLocaleString();
}
