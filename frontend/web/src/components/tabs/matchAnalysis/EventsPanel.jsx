import { useState } from 'react';
import api from '../../../api/client.js';
import { useAsync } from '../../../hooks/useAsync.js';
import { Chip, MethodStrip, Panel, Slab } from '../../ui/Primitives.jsx';
import { AsyncBlock, EmptyState } from '../../ui/States.jsx';

/* ==================================================================== */
/* Detected events                                                      */
/* ==================================================================== */

const EVENT_TONE = {
  shot: 'bad',
  pass: 'good',
  turnover: 'warn',
  first_touch: 'neutral',
};

/**
 * The match's detected events, as a timeline that seeks the video.
 *
 * ADDED: the pipeline has written pass / shot / turnover / first-touch rows
 * since Phase 3 and there was no endpoint to read them and no UI to show
 * them. The `events` table was invisible to the product.
 *
 * The `space` chip is not decoration. Possession -- which every event here is
 * derived from -- is measured in pitch metres when calibration validates and
 * in image pixels when it does not, and the second is a weaker instrument.
 * The reading is labelled with which one produced it rather than presented
 * flat.
 */
export function EventsPanel({ matchId, dataEpoch = 0, onSeek }) {
  const [filter, setFilter] = useState(null);

  const events = useAsync(
    (signal) => api.getEvents(matchId, { eventType: filter }, signal),
    [matchId, filter, dataEpoch],
    { enabled: Boolean(matchId) },
  );

  if (!matchId) {
    return (
      <Slab>
        <EmptyState
          title="Select a Match First"
          message="Events are detected per match by the pipeline's pass, shot, turnover and first-touch heuristics."
        />
      </Slab>
    );
  }

  const byType = events.data?.by_type || {};
  const types = Object.keys(byType).sort();

  return (
    <Panel
      number="05"
      title="Detected Events"
      subtitle="GET /api/matches/{match_id}/events — pass, shot, turnover and first-touch heuristics."
    >
      {types.length ? (
        <div className="dash-quickpick">
          <span className="dash-label">Filter</span>
          <div className="dash-quickpick-row">
            <button
              type="button"
              className={`dash-pill ${filter === null ? 'is-active' : ''}`.trim()}
              onClick={() => setFilter(null)}
            >
              all ({events.data?.total ?? 0})
            </button>
            {types.map((type) => (
              <button
                key={type}
                type="button"
                className={`dash-pill ${filter === type ? 'is-active' : ''}`.trim()}
                onClick={() => setFilter(type)}
              >
                {type} ({byType[type]})
              </button>
            ))}
          </div>
        </div>
      ) : null}

      <AsyncBlock
        state={events}
        loadingLabel="Reading detected events"
        isEmpty={(data) => !data?.events?.length}
        emptyTitle="No Events Detected"
        emptyMessage={
          'The pipeline found no possession sequence to derive events from. That happens when the ball '
          + 'model detected the ball on too few frames, or when no tracked player was ever within the '
          + 'control radius of it. Events are never invented to fill this panel.'
        }
      >
        {(data) => (
          <>
            <MethodStrip
              items={[
                {
                  label: 'space',
                  value: data.space === 'pitch' ? 'pitch / metres' : 'image / pixels',
                  tone: data.space === 'pitch' ? 'good' : 'warn',
                },
                { label: 'events', value: `${data.returned} of ${data.total}` },
              ]}
            />

            {data.space === 'image' ? (
              <p className="dash-inline-error">
                <span className="dash-inline-error-key">Estimated</span>
                This match has no valid calibration, so possession was resolved in image space using
                each player&apos;s own bounding-box height as the local scale. The events are real
                detections; any distance attached to one is an estimate, and none of them carry a
                pitch position.
              </p>
            ) : null}

            <Slab className="dash-stage-table">
              {data.events.map((event) => (
                <button
                  key={event.event_id}
                  type="button"
                  className="dash-event-row"
                  onClick={() => onSeek?.(event.timestamp)}
                  title="Jump the player to this moment"
                >
                  <span className="dash-event-time">{event.timestamp.toFixed(2)}s</span>
                  <Chip tone={EVENT_TONE[event.event_type] || 'neutral'}>{event.event_type}</Chip>
                  <span className="dash-event-players">
                    {event.player_id !== null && event.player_id !== undefined ? (
                      <>#{event.player_id}</>
                    ) : (
                      <span className="dash-absent">unattributed</span>
                    )}
                    {event.related_player_id !== null && event.related_player_id !== undefined ? (
                      <> → #{event.related_player_id}</>
                    ) : null}
                  </span>
                  <span className="dash-event-meta">
                    {event.metadata?.pass_distance_m
                      ? `${event.metadata.pass_distance_m}m${
                          event.metadata.pass_distance_m_is_estimate ? ' (est)' : ''
                        }`
                      : event.team_id || ''}
                  </span>
                </button>
              ))}
            </Slab>
          </>
        )}
      </AsyncBlock>
    </Panel>
  );
}
