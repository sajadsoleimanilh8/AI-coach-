import { useEffect, useMemo, useState } from 'react';
import api from '../../../api/client.js';
import { useAsync } from '../../../hooks/useAsync.js';
import { MetaRow, MetricValue, Panel, PanelGrid, Slab } from '../../ui/Primitives.jsx';
import { AsyncBlock, EmptyState } from '../../ui/States.jsx';
import { NumberField } from '../../ui/Form.jsx';
import { HeatmapPitch } from '../../ui/Pitch.jsx';

/* ==================================================================== */
/* Heatmap                                                              */
/* ==================================================================== */

export function HeatmapPanel({ matchId, goToTab, dataEpoch = 0 }) {
  const [playerId, setPlayerId] = useState(1);
  const [requested, setRequested] = useState(null);

  // Player ids come from the authoritative list where one exists; this is a
  // discovery aid, not a requirement, so its failure is non-blocking.
  const roster = useAsync(
    (signal) => api.getPlayerIntelligence(matchId, signal),
    [matchId, dataEpoch],
    { enabled: Boolean(matchId) },
  );

  const knownIds = useMemo(
    () => (Array.isArray(roster.data) ? roster.data.map((entry) => entry.player_id) : []),
    [roster.data],
  );

  // Pitch space is the real reading and stays the default. Image space is
  // offered because on footage this project's calibration cannot solve, the
  // pitch grid is correctly but permanently empty -- and "where in frame was
  // this player" is a question the stored pixels can actually answer.
  const [space, setSpace] = useState('pitch');
  // Set when the pitch grid came back with no usable samples and this panel
  // fell back to image space on its own. Distinguishes "the user chose image
  // space" from "pitch space had nothing", which the note below states.
  const [autoFellBack, setAutoFellBack] = useState(false);

  // AUTO-SELECT. This panel used to render nothing at all until the user
  // typed a tracking id and pressed a button -- so the heatmap appeared
  // broken on a match that had perfectly good data, because nothing told the
  // user an id was required and there was no way to know a valid one. The
  // first id the backend reports is selected as soon as the roster arrives.
  useEffect(() => {
    if (requested === null && knownIds.length) {
      setPlayerId(knownIds[0]);
      setRequested(knownIds[0]);
    }
  }, [knownIds, requested]);

  // A new run invalidates the previous selection's data, and may change which
  // space has anything in it. Re-arm the fallback so it is judged again.
  useEffect(() => {
    setAutoFellBack(false);
  }, [dataEpoch, requested]);

  const heatmap = useAsync(
    (signal) => api.getHeatmap(matchId, requested, space, signal),
    [matchId, requested, space, dataEpoch],
    { enabled: Boolean(matchId) && requested !== null },
  );

  // AUTOMATIC FALLBACK. On every clip this project's calibration model cannot
  // solve -- currently all real broadcast footage -- the pitch grid is
  // correctly empty, and a user who does not already know that reads an empty
  // panel as a broken heatmap. When the response says the player HAS tracking
  // rows but none of them carry pitch coordinates, switch to the reading that
  // those same rows can support, and say so.
  //
  // Only when there is genuinely something to fall back TO: a player with no
  // tracking rows at all (sample_size 0) has nothing in either space, and
  // switching would swap one honest empty state for another.
  useEffect(() => {
    const data = heatmap.data;
    if (!data || autoFellBack) return;
    if (data.space === 'pitch' && data.usable_sample_size === 0 && data.sample_size > 0) {
      setAutoFellBack(true);
      setSpace('image');
    }
  }, [heatmap.data, autoFellBack]);

  if (!matchId) {
    return (
      <Slab>
        <EmptyState
          title="Select a Match First"
          message="Heatmaps are served per match and player."
          action={
            <button type="button" className="btn-ghost" onClick={() => goToTab('match')}>
              Upload a clip above
            </button>
          }
        />
      </Slab>
    );
  }

  return (
    <PanelGrid columns={2}>
      <Panel
        number="03"
        title="Player Heatmap"
        subtitle="GET /api/matches/{match_id}/heatmap/{player_id}"
      >
        <form
          className="dash-form dash-form-inline"
          onSubmit={(event) => {
            event.preventDefault();
            setRequested(Number(playerId));
          }}
        >
          <NumberField
            label="Player ID"
            value={playerId}
            min={0}
            step={1}
            onChange={setPlayerId}
            hint="ByteTrack tracking ID, scoped to this processed video."
          />
          <button type="submit" className="btn-primary dash-submit">
            Render Heatmap
          </button>
        </form>

        <div className="dash-space-toggle">
          <span className="dash-label">Coordinate space</span>
          <div className="dash-source-toggle" role="group" aria-label="Heatmap coordinate space">
            <button
              type="button"
              className={`dash-source-btn ${space === 'pitch' ? 'is-active' : ''}`.trim()}
              onClick={() => {
                setAutoFellBack(true); // an explicit choice is not overridden
                setSpace('pitch');
              }}
              aria-pressed={space === 'pitch'}
            >
              Pitch (metres)
            </button>
            <button
              type="button"
              className={`dash-source-btn ${space === 'image' ? 'is-active' : ''}`.trim()}
              onClick={() => {
                setAutoFellBack(true);
                setSpace('image');
              }}
              aria-pressed={space === 'image'}
            >
              Image (pixels)
            </button>
          </div>
          <p className="dash-hint">
            {space === 'pitch'
              ? 'Metres on a 105×68 model, via a validated homography. Comparable between matches — and empty when calibration never validated.'
              : 'Where the player appeared in the video frame. Needs no calibration, but it is not a pitch position: it moves with the camera and cannot be compared across matches.'}
          </p>
        </div>

        {knownIds.length ? (
          <div className="dash-quickpick">
            <span className="dash-label">Detected players</span>
            <div className="dash-quickpick-row">
              {knownIds.slice(0, 24).map((id) => (
                <button
                  key={id}
                  type="button"
                  className={`dash-pill ${requested === id ? 'is-active' : ''}`.trim()}
                  onClick={() => {
                    setPlayerId(id);
                    setRequested(id);
                  }}
                >
                  {id}
                </button>
              ))}
            </div>
          </div>
        ) : roster.status === 'success' ? (
          <p className="dash-hint dash-absent">
            No player-intelligence rows for this match yet, so there is no roster to pick from — enter a tracking ID directly.
          </p>
        ) : null}
      </Panel>

      <Panel number="04" title="Density Grid" subtitle="Density is the backend's own count / max_count — never rescaled here.">
        <AsyncBlock
          state={heatmap}
          idle={
            <EmptyState
              title="No Player Selected"
              message="Pick a tracking ID and render to see where that player actually spent the match."
            />
          }
          loadingLabel="Building heatmap"
          isEmpty={(data) => !data?.cells?.length}
          emptyTitle="No Usable Samples"
          emptyMessage={
            space === 'pitch'
              ? 'This player has tracking rows, but none of them carry valid pitch coordinates — so there is no position to plot. Switch to image space to plot the pixel positions that do exist.'
              : 'This player has no tracking rows carrying a pixel position, so there is nothing to bin.'
          }
        >
          {(data) => (
            <>
              <HeatmapPitch heatmap={data} />
              <div className="dash-meta-block">
                {/* The space the RESPONSE reports, not the one requested --
                    these two readings are not interchangeable and the label
                    has to come from the data. */}
                <MetaRow label="space">
                  {data.space === 'image' ? (
                    <span>
                      image / pixels
                      {data.frame_width_px
                        ? ` · ${data.frame_width_px}×${data.frame_height_px} frame`
                        : ''}
                    </span>
                  ) : (
                    <span>pitch / metres · {data.pitch_length_m}×{data.pitch_width_m}m</span>
                  )}
                </MetaRow>
                <MetaRow label="confidence">
                  <MetricValue value={data.confidence} absentLabel="Not Reported" />
                </MetaRow>
                <MetaRow label="sample size">
                  <MetricValue value={data.sample_size} />
                </MetaRow>
                <MetaRow label="usable samples">
                  <MetricValue value={data.usable_sample_size} />
                </MetaRow>
              </div>

              {data.space === 'image' ? (
                <p className="dash-inline-error">
                  <span className="dash-inline-error-key">Image space</span>
                  {autoFellBack
                    ? 'Shown automatically because this match has no valid calibration, so the pitch grid has nothing in it. '
                    : ''}
                  These cells are frame pixels, not pitch positions. They shift with the camera,
                  carry no distance in metres, and cannot be compared against another match.
                </p>
              ) : null}

              {data.space === 'pitch' && data.usable_sample_size === 0 && data.sample_size > 0 ? (
                <p className="dash-inline-error">
                  <span className="dash-inline-error-key">Note</span>
                  {data.sample_size} tracking rows exist for this player, but none were captured under valid calibration.
                </p>
              ) : null}
            </>
          )}
        </AsyncBlock>
      </Panel>
    </PanelGrid>
  );
}
