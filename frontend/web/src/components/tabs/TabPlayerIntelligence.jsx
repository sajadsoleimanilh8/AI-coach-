import { useState } from 'react';

import api from '../../api/client.js';
import { useAsync } from '../../hooks/useAsync.js';
import {
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
 * TAB 2 -- per-player scores from ai/player_intelligence/*.
 *
 * The nine categories below are the full set the engine can produce. All nine
 * are ALWAYS listed: a category the backend did not return is rendered as
 * "Not Yet Computed" rather than being hidden, because a silently shorter
 * list is indistinguishable from a player who happens to score badly on the
 * missing ones.
 */
const CATEGORIES = [
  { key: 'first_touch_score', label: 'First Touch' },
  { key: 'passing_vision_score', label: 'Passing Vision' },
  { key: 'press_resistance_score', label: 'Press Resistance' },
  { key: 'defensive_positioning_score', label: 'Defensive Positioning' },
  { key: 'off_ball_movement_score', label: 'Off-Ball Movement' },
  { key: 'decision_making_score', label: 'Decision Making' },
  { key: 'finishing_efficiency_score', label: 'Finishing Efficiency' },
  { key: 'body_orientation_score', label: 'Body Orientation' },
  { key: 'scanning_behavior_score', label: 'Scanning Behavior' },
];

const LABELS = new Map(CATEGORIES.map((category) => [category.key, category.label]));

function prettify(metricName) {
  return (
    LABELS.get(metricName)
    || String(metricName)
      .replace(/_score$/, '')
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (character) => character.toUpperCase())
  );
}

export default function TabPlayerIntelligence({ matchId, goToTab, onAttachMatch }) {
  const [pickedPlayer, setPickedPlayer] = useState(null);

  const roster = useAsync(
    (signal) => api.getPlayerIntelligence(matchId, signal),
    [matchId],
    { enabled: Boolean(matchId) },
  );

  // A tracking ID from a previous match can never be shown under a new
  // match's header: DashboardApp keys this tab by matchId, so a new match is a
  // fresh component. Until the user picks someone, the first roster entry is
  // the selection -- derived here rather than copied into state by an effect.
  const selectedPlayer =
    pickedPlayer ?? (Array.isArray(roster.data) && roster.data.length ? roster.data[0].player_id : null);

  return (
    <TabSection animKey={`player-${matchId}`}>
      <SectionHeader
        tag="Player Intelligence"
        title={<>Per-Player<br />Breakdown</>}
        meta="Nine score categories computed from tracking, each carrying the engine's own method, confidence, and sub-score record."
      />

      {!matchId ? (
        <SelectMatchState onGoToUpload={() => goToTab('match')} onAttachMatch={onAttachMatch} />
      ) : (
        <AsyncBlock
          state={roster}
          loadingLabel="Loading player roster"
          isEmpty={(data) => !Array.isArray(data) || data.length === 0}
          emptyTitle="No Player Metrics Computed"
          emptyMessage="The pipeline has not written any PlayerMetric rows for this match. Once a processing job completes, every tracked player appears here."
        >
          {(players) => (
            <>
              <RosterStrip
                players={players}
                selected={selectedPlayer}
                onSelect={setPickedPlayer}
              />
              <PlayerDetail matchId={matchId} playerId={selectedPlayer} roster={players} />
            </>
          )}
        </AsyncBlock>
      )}
    </TabSection>
  );
}

function RosterStrip({ players, selected, onSelect }) {
  return (
    <Slab className="dash-roster">
      <span className="dash-label">Tracked Players ({players.length})</span>
      <div className="dash-roster-row">
        {players.map((player) => {
          const computed = (player.metrics || []).filter(
            (metric) => metric.value !== null && metric.value !== undefined,
          ).length;
          return (
            <button
              key={player.player_id}
              type="button"
              className={`dash-pill dash-pill-lg ${selected === player.player_id ? 'is-active' : ''}`.trim()}
              onClick={() => onSelect(player.player_id)}
            >
              <span className="dash-pill-id">{player.player_id}</span>
              {player.player_name ? (
                <span className="dash-pill-name">{player.player_name}</span>
              ) : null}
              <span className="dash-pill-meta">
                {computed}/{CATEGORIES.length}
              </span>
            </button>
          );
        })}
      </div>
      <p className="dash-hint">
        The count is how many of the nine categories carry a real value for that player.
      </p>
    </Slab>
  );
}

function PlayerDetail({ matchId, playerId, roster }) {
  const detail = useAsync(
    (signal) => api.getPlayerMetrics(matchId, playerId, signal),
    [matchId, playerId],
    { enabled: Boolean(matchId) && playerId !== null && playerId !== undefined },
  );

  if (playerId === null || playerId === undefined) {
    return (
      <EmptyState
        title="No Player Selected"
        message="Pick a tracking ID above to see its nine score categories."
      />
    );
  }

  const entry = roster.find((player) => player.player_id === playerId);

  return (
    <>
      <SubHeading>
        Player {playerId}
        {entry?.player_name ? ` — ${entry.player_name}` : ''}
      </SubHeading>

      <AsyncBlock
        state={detail}
        loadingLabel={`Loading metrics for player ${playerId}`}
        loadingRows={4}
      >
        {(metrics) => {
          const byName = new Map(
            (Array.isArray(metrics) ? metrics : []).map((metric) => [metric.metric_name, metric]),
          );

          // Anything the engine produced that is not one of the nine known
          // categories still gets a card -- dropping it would hide real data.
          const extras = (Array.isArray(metrics) ? metrics : []).filter(
            (metric) => !LABELS.has(metric.metric_name),
          );

          return (
            <>
              <PanelGrid columns={3}>
                {CATEGORIES.map((category, index) => (
                  <MetricCard
                    key={category.key}
                    number={String(index + 1).padStart(2, '0')}
                    label={category.label}
                    metric={byName.get(category.key)}
                  />
                ))}
              </PanelGrid>

              {extras.length ? (
                <>
                  <SubHeading>Additional Metrics</SubHeading>
                  <PanelGrid columns={3}>
                    {extras.map((metric, index) => (
                      <MetricCard
                        key={metric.metric_id}
                        number={String(CATEGORIES.length + index + 1).padStart(2, '0')}
                        label={prettify(metric.metric_name)}
                        metric={metric}
                      />
                    ))}
                  </PanelGrid>
                </>
              ) : null}
            </>
          );
        }}
      </AsyncBlock>
    </>
  );
}

function MetricCard({ number, label, metric }) {
  if (!metric) {
    return (
      <Panel number={number} title={label} tone="absent">
        <p className="dash-absent dash-metric-absent">Not Yet Computed</p>
        <p className="dash-hint">
          No PlayerMetric row exists for this category. That is not a score of zero — the engine has
          not produced one.
        </p>
      </Panel>
    );
  }

  const hasValue = metric.value !== null && metric.value !== undefined;

  return (
    <Panel number={number} title={label}>
      <div className="dash-metric-head">
        {hasValue ? (
          <span className="stat-num dash-metric-headline">{Number(metric.value).toFixed(1)}</span>
        ) : (
          <span className="dash-absent dash-metric-absent">Not Yet Computed</span>
        )}
        <ConfidenceBadge confidence={metric.confidence} />
      </div>

      <MethodStrip
        items={[
          { label: 'method', value: metric.method },
          { label: 'n', value: metric.sample_size },
          { label: 'schema', value: metric.schema_version },
        ]}
      />

      <SubScores subScores={metric.sub_scores} />

      <div className="dash-meta-block">
        <MetaRow label="computed">
          <MetricValue value={formatTime(metric.computed_at)} absentLabel="Not Recorded" />
        </MetaRow>
        <MetaRow label="metric_id">
          <code className="dash-code-sm">{metric.metric_id}</code>
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
