import { useEffect, useMemo, useRef, useState } from 'react';
import api from '../../../api/client.js';
import { useAsync } from '../../../hooks/useAsync.js';
import { Chip, MetaRow, MetricValue, Slab } from '../../ui/Primitives.jsx';
import { AsyncBlock, EmptyState } from '../../ui/States.jsx';
import { NumberField } from '../../ui/Form.jsx';
import { TrackingMinimap, TrackingOverlaySvg } from '../../ui/Pitch.jsx';

/* ==================================================================== */
/* Playback + tracking overlay                                          */
/* ==================================================================== */

/** Frames per tracking request. Mirrors the server's own default window. */
const WINDOW_FRAMES = 150;

/**
 * How close to the end of the loaded window playback may get before the next
 * one is fetched. 40 frames is ~1.6s at 25fps -- enough time for the request
 * to land before the overlay would otherwise run out of data.
 */
const PREFETCH_MARGIN_FRAMES = 40;

/**
 * Tracks the <video> element's currentTime and reports it on every animation
 * frame while playing.
 *
 * WHY NOT `timeupdate`. That event fires roughly 4x a second, so the overlay
 * lagged the picture by up to 250ms and visibly stepped. rAF is the browser's
 * own paint clock, so the overlay is recomputed exactly as often as the frame
 * it is drawn over -- and because the value read is always
 * `video.currentTime`, it cannot drift from the video the way an independent
 * timer would. Seeking, pausing, and playbackRate changes need no special
 * handling: currentTime is already the truth in all three cases. `seeked` and
 * `pause` are still subscribed so a seek made while paused updates the
 * overlay immediately rather than waiting for the next play.
 */
function useVideoTime(videoRef, ready) {
  const [time, setTime] = useState(0);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !ready) return undefined;

    let raf = null;
    let cancelled = false;

    const sample = () => {
      if (cancelled) return;
      setTime(video.currentTime);
      raf = requestAnimationFrame(sample);
    };

    const start = () => {
      if (raf === null) raf = requestAnimationFrame(sample);
    };
    const stop = () => {
      if (raf !== null) cancelAnimationFrame(raf);
      raf = null;
      setTime(video.currentTime);
    };

    video.addEventListener('play', start);
    video.addEventListener('playing', start);
    video.addEventListener('pause', stop);
    video.addEventListener('ended', stop);
    video.addEventListener('seeked', stop);
    video.addEventListener('loadedmetadata', stop);

    if (!video.paused) start();
    else setTime(video.currentTime);

    return () => {
      cancelled = true;
      if (raf !== null) cancelAnimationFrame(raf);
      video.removeEventListener('play', start);
      video.removeEventListener('playing', start);
      video.removeEventListener('pause', stop);
      video.removeEventListener('ended', stop);
      video.removeEventListener('seeked', stop);
      video.removeEventListener('loadedmetadata', stop);
    };
  }, [videoRef, ready]);

  return time;
}

export function PlaybackPanel({ videoId, matchId, dataEpoch = 0, seekRequest = null }) {
  const videoRef = useRef(null);
  const [videoReady, setVideoReady] = useState(false);
  const [windowStart, setWindowStart] = useState(0);
  const [manualFrame, setManualFrame] = useState(0);
  const [follow, setFollow] = useState(true);
  const [showOverlay, setShowOverlay] = useState(true);
  const [scrubIndex, setScrubIndex] = useState(0);

  const tracking = useAsync(
    (signal) => api.getTracking(matchId, { startFrame: windowStart }, signal),
    [matchId, windowStart, dataEpoch],
    { enabled: Boolean(matchId) },
  );

  // Does an annotated render exist for this match? Re-asked on dataEpoch,
  // because the render is written by the LAST pipeline stage -- the answer
  // changes from no to yes exactly when a run completes.
  const summary = useAsync(
    (signal) => api.getMatchSummary(matchId, signal),
    [matchId, dataEpoch],
    { enabled: Boolean(matchId) },
  );
  const hasProcessed = summary.data?.processed_video_exists === true;
  const processedError = summary.data?.processed_video_error || null;

  // Default to the processed clip once it exists. `source` is only the
  // user's explicit override, so the default can follow availability rather
  // than being frozen at whatever was true on first render -- this is what
  // makes the player switch to the processed video by itself the moment a
  // run completes, with no refresh.
  const [sourceOverride, setSourceOverride] = useState(null);
  const source = sourceOverride ?? (hasProcessed ? 'processed' : 'raw');
  const showingProcessed = source === 'processed' && hasProcessed;

  const frames = useMemo(() => tracking.data?.frames || [], [tracking.data]);
  const fps = tracking.data?.fps || 25;
  const calibration = tracking.data?.calibration;
  const maxFrame = tracking.data?.max_frame ?? null;

  // The SOURCE clip's extent. The overlay's coordinate system, and the one
  // thing that lets the same overlay sit correctly over the raw clip and the
  // downscaled render alike -- see TrackingOverlaySvg's docstring. Falls back
  // to the played element's own size only when the backend could not read the
  // file, which is right for the raw clip and merely imprecise for the render.
  const [elementSize, setElementSize] = useState(null);
  const sourceWidth = tracking.data?.frame_width_px
    ?? summary.data?.frame_width_px
    ?? elementSize?.width
    ?? null;
  const sourceHeight = tracking.data?.frame_height_px
    ?? summary.data?.frame_height_px
    ?? elementSize?.height
    ?? null;

  const videoTime = useVideoTime(videoRef, videoReady);

  // The frame to draw. Derived from the video clock while following, from the
  // scrubber when not -- one value, so the two cannot disagree.
  const followedIndex = useMemo(() => {
    if (!frames.length) return 0;
    // Nearest, not interpolated: there is no tracking row between two
    // persisted frames and inventing one would be fabrication.
    let best = 0;
    let bestDelta = Infinity;
    for (let i = 0; i < frames.length; i += 1) {
      const delta = Math.abs(frames[i].timestamp - videoTime);
      if (delta < bestDelta) {
        bestDelta = delta;
        best = i;
      }
    }
    return best;
  }, [frames, videoTime]);

  const frameIndex = follow
    ? followedIndex
    : Math.min(scrubIndex, Math.max(0, frames.length - 1));
  const currentFrame = frames[frameIndex] || null;

  // AUTO-PAGING. The window is 150 frames -- six seconds. Before this, the
  // only way to move it was to type a start frame into a form and press a
  // button, so the overlay went blank six seconds into every clip and stayed
  // blank. Playback position now drives the window: crossing near either edge
  // loads the neighbouring one.
  useEffect(() => {
    if (!follow || !videoReady || !fps) return;
    const playhead = Math.floor(videoTime * fps);
    const lowerBound = windowStart;
    const upperBound = windowStart + WINDOW_FRAMES;

    if (playhead >= upperBound - PREFETCH_MARGIN_FRAMES) {
      const next = windowStart + WINDOW_FRAMES;
      if (maxFrame === null || next <= maxFrame) setWindowStart(next);
    } else if (playhead < lowerBound) {
      // A backwards seek past the start of the window. Jump straight to the
      // window containing the playhead rather than stepping back one at a
      // time, which would need several round trips after a long rewind.
      setWindowStart(Math.max(0, Math.floor(playhead / WINDOW_FRAMES) * WINDOW_FRAMES));
    }
  }, [videoTime, fps, follow, videoReady, windowStart, maxFrame]);

  // A seek asked for from elsewhere on the page (the events timeline).
  // Keyed on the request's nonce so clicking the same event twice seeks
  // twice.
  const seekNonce = seekRequest?.nonce ?? null;
  const seekTime = seekRequest?.timestamp ?? null;
  useEffect(() => {
    if (seekNonce === null || seekTime === null) return;
    const video = videoRef.current;
    if (!video) return;
    setFollow(true);
    video.currentTime = seekTime;
    // Move the window with it, so the overlay has data at the destination
    // without waiting for the auto-pager's next tick.
    if (fps) {
      const frame = Math.floor(seekTime * fps);
      setWindowStart(Math.max(0, Math.floor(frame / WINDOW_FRAMES) * WINDOW_FRAMES));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seekNonce]);

  const seekToFrame = (index) => {
    setScrubIndex(index);
    const video = videoRef.current;
    const frame = frames[index];
    if (video && frame && Number.isFinite(frame.timestamp)) {
      video.currentTime = frame.timestamp;
    }
  };

  if (!videoId) {
    return (
      <Slab>
        <EmptyState
          title="No Clip Loaded"
          message="Upload a match video above. The raw file streams back from GET /api/videos/{video_id}/file as soon as the upload finishes, whether or not processing has completed."
        />
      </Slab>
    );
  }

  return (
    <div className="dash-playback">
      <div className="dash-playback-video" data-anim>
        <div className="dash-video-stack">
          <video
            ref={videoRef}
            // Remounts on source change so the element reloads instead of
            // keeping the previous clip's decoded state.
            key={showingProcessed ? 'processed' : 'raw'}
            className="dash-video"
            src={showingProcessed ? api.processedVideoUrl(videoId) : api.videoFileUrl(videoId)}
            controls
            preload="metadata"
            onLoadedMetadata={(event) => {
              setElementSize({
                width: event.currentTarget.videoWidth,
                height: event.currentTarget.videoHeight,
              });
              setVideoReady(true);
            }}
          />
          {/* The overlay sits over WHICHEVER clip is playing, including the
              processed one, and is drawn in the source clip's coordinate
              space so it lines up over both (see TrackingOverlaySvg).

              Over the annotated render it deliberately draws LESS: rings and
              the ball, but no track-id text, because the render already has
              those ids burned in. Two copies of one id on one player would
              read as two independent identifications when they are the same
              measurement drawn twice. The ring is a different shape from the
              burned rectangle for the same reason. */}
          {showOverlay && currentFrame && sourceWidth && sourceHeight ? (
            <TrackingOverlaySvg
              frame={currentFrame}
              sourceWidth={sourceWidth}
              sourceHeight={sourceHeight}
              showPlayers
              showBall
              showLabels={!showingProcessed}
            />
          ) : null}
        </div>

        <div className="dash-playback-controls">
          <div className="dash-source-toggle" role="group" aria-label="Video source">
            <button
              type="button"
              className={`dash-source-btn ${showingProcessed ? 'is-active' : ''}`.trim()}
              onClick={() => setSourceOverride('processed')}
              disabled={!hasProcessed}
              aria-pressed={showingProcessed}
            >
              Processed
            </button>
            <button
              type="button"
              className={`dash-source-btn ${!showingProcessed ? 'is-active' : ''}`.trim()}
              onClick={() => setSourceOverride('raw')}
              aria-pressed={!showingProcessed}
            >
              Original
            </button>
          </div>

          <p className="dash-hint">
            {showingProcessed
              ? 'The pipeline’s own render — boxes, track ids and team colours burned into the pixels — with the live tactical overlay synchronised on top. The overlay’s rings track the same detections as the burned-in boxes, so they confirm alignment rather than adding a second reading.'
              : hasProcessed
                ? 'The raw upload, with the full tracking overlay drawn live.'
                : 'The raw upload. The annotated render is written by the last pipeline stage — it appears here on its own once a run completes.'}
          </p>

          {/* A render that failed must say so here rather than leaving the
              Processed button greyed out with no explanation. */}
          {!hasProcessed && processedError ? (
            <div className="dash-inline-error" role="alert">
              <span className="dash-inline-error-key">No annotated render</span>
              {processedError}
            </div>
          ) : null}

          <label className="dash-check">
            <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
            <span>Follow playback</span>
          </label>

          <label className="dash-check">
            <input
              type="checkbox"
              checked={showOverlay}
              onChange={(e) => setShowOverlay(e.target.checked)}
            />
            <span>Tactical overlay</span>
          </label>

          <div className="dash-frame-scrub">
            <input
              type="range"
              className="dash-range"
              min={0}
              max={Math.max(0, frames.length - 1)}
              value={frameIndex}
              disabled={!frames.length}
              onChange={(event) => {
                setFollow(false);
                seekToFrame(Number(event.target.value));
              }}
            />
            <span className="dash-frame-readout">
              {currentFrame ? (
                <>
                  frame <strong>{currentFrame.frame_number}</strong> · {currentFrame.timestamp.toFixed(2)}s
                  {maxFrame !== null ? <> · window {windowStart}–{windowStart + WINDOW_FRAMES} of {maxFrame}</> : null}
                </>
              ) : (
                <span className="dash-absent">no frames in window</span>
              )}
            </span>
          </div>

          <form
            className="dash-window-form"
            onSubmit={(event) => {
              event.preventDefault();
              const target = Math.max(0, Number(manualFrame) || 0);
              setWindowStart(target);
              const video = videoRef.current;
              if (video && fps) video.currentTime = target / fps;
            }}
          >
            <NumberField
              label="Jump to frame"
              value={manualFrame}
              min={0}
              step={50}
              onChange={setManualFrame}
              hint="The window follows playback on its own; this is for jumping somewhere specific."
            />
            <button type="submit" className="btn-ghost dash-window-btn">
              Jump
            </button>
          </form>
        </div>
      </div>

      <div className="dash-playback-side" data-anim>
        <h4 className="dash-card-title">2D Pitch Minimap</h4>
        {!matchId ? (
          <EmptyState
            title="No Match Linked"
            message="Tracking is served per match. Upload a clip to get a match_id."
          />
        ) : (
          <AsyncBlock
            state={tracking}
            loadingLabel="Loading tracking window"
            loadingRows={2}
            isEmpty={(data) => !data?.frames?.length}
            emptyTitle="No Tracking Rows"
            emptyMessage={`No tracking data was persisted for frames ${windowStart}–${windowStart + WINDOW_FRAMES}. Either the pipeline has not reached this window yet, or it produced no detections here.`}
          >
            {(data) => (
              <>
                <TrackingMinimap frame={currentFrame} calibration={data.calibration} />
                <CalibrationSummary calibration={calibration} fps={fps} />
              </>
            )}
          </AsyncBlock>
        )}
      </div>
    </div>
  );
}

/**
 * The calibration block for a tracking window. This is the part that must not
 * be quietly skipped: on current broadcast footage pitch coordinates are
 * frequently null, and a minimap that silently draws nothing looks identical
 * to a minimap of an empty pitch.
 */
function CalibrationSummary({ calibration, fps }) {
  if (!calibration) {
    return (
      <p className="dash-absent dash-pitch-note">
        Calibration status not reported for this window.
      </p>
    );
  }

  const valid = Boolean(calibration.valid_in_window);

  return (
    <div className={`dash-calib-summary ${valid ? 'is-valid' : 'is-invalid'}`}>
      <Chip tone={valid ? 'good' : 'warn'}>
        {valid ? 'Calibration valid' : 'Calibration unavailable'}
      </Chip>
      <div className="dash-meta-block">
        <MetaRow label="fps">
          <MetricValue value={fps} precision={2} />
        </MetaRow>
        <MetaRow label="rows in window">
          <MetricValue value={calibration.tracking_rows_in_window} />
        </MetaRow>
        <MetaRow label="rows with pitch coords">
          <MetricValue value={calibration.rows_with_pitch_coordinates} />
        </MetaRow>
        <MetaRow label="status rows">
          {calibration.has_status_rows ? 'present' : <span className="dash-absent">absent</span>}
        </MetaRow>
        <MetaRow label="valid frame ranges">
          {calibration.valid_frame_ranges?.length ? (
            <code>
              {calibration.valid_frame_ranges.map(([lo, hi]) => `${lo}–${hi}`).join(', ')}
            </code>
          ) : (
            <span className="dash-absent">none</span>
          )}
        </MetaRow>
      </div>
      {calibration.note ? <p className="dash-hint">{calibration.note}</p> : null}
    </div>
  );
}
