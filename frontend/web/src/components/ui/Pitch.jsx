import { EmptyState } from './States.jsx';

/**
 * Pitch visuals, drawn with the exact same vectors as the #logo-draw SVG on
 * the landing page (viewBox 0 0 380 260, the same rectangle / halfway line /
 * penalty boxes / centre circle, stroked with the same blue->purple
 * gradient). The dashboard's pitch and the marketing pitch are literally the
 * same drawing.
 */

/**
 * Colour for a team id. One definition, used by both the minimap and the
 * video overlay.
 *
 * FIXED: each of those two had its own copy and BOTH matched by substring --
 * the minimap on `includes('a')`, the overlay on `includes('b')`. Every team
 * id in this system is "team-home" or "team-away": both contain an "a" (in
 * "team") and neither contains a "b", so the minimap painted both teams blue
 * and so did the overlay. The team distinction they appeared to draw was
 * decorative in both.
 *
 * Matched exactly now, with anything unrecognised reading as unassigned --
 * which is true -- rather than being bucketed by how its id is spelled. Kept
 * identical to backend/pipeline/overlay_video.py's TEAM_COLOURS_BGR, so the
 * live overlay and the burned-in annotated video cannot disagree about which
 * team a player is on.
 */
export function teamColour(teamId) {
  const key = teamId === null || teamId === undefined ? '' : String(teamId).trim().toLowerCase();
  if (key === 'team-home' || key === 'home' || key === '0') return '#00d4ff';
  if (key === 'team-away' || key === 'away' || key === '1') return '#f472b6';
  return '#64748b';
}

const VIEW_W = 380;
const VIEW_H = 260;
const PAD = 10;
const INNER_W = VIEW_W - PAD * 2; // 360
const INNER_H = VIEW_H - PAD * 2; // 240

/** metres -> SVG units, for a pitch of the length/width the backend reports. */
export function projectToPitch(xMetres, yMetres, lengthM, widthM) {
  return {
    x: PAD + (xMetres / lengthM) * INNER_W,
    y: PAD + (yMetres / widthM) * INNER_H,
  };
}

export function PitchOutline({ gradientId = 'pitch-grad' }) {
  return (
    <>
      <defs>
        <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#00d4ff" />
          <stop offset="100%" stopColor="#8b5cf6" />
        </linearGradient>
      </defs>
      <path
        d="M10 10 L370 10 L370 250 L10 250 Z M190 10 L190 250 M10 60 L60 60 L60 200 L10 200 M370 60 L320 60 L320 200 L370 200"
        stroke={`url(#${gradientId})`}
        strokeWidth="1.5"
        fill="rgba(0, 212, 255, 0.02)"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="190" cy="130" r="30" stroke={`url(#${gradientId})`} strokeWidth="1.5" fill="none" />
      <circle cx="190" cy="130" r="2" fill="#00d4ff" opacity="0.6" />
    </>
  );
}

/**
 * GET /api/matches/{match_id}/heatmap/{player_id} rendered as a density grid
 * over the pitch outline.
 *
 * `density` is already normalised 0..1 by the backend (count / max_count),
 * so it maps straight to opacity -- nothing is rescaled here, which would
 * make one player's heatmap incomparable to another's.
 */
export function HeatmapPitch({ heatmap }) {
  const {
    cells = [],
    grid_cols: cols,
    grid_rows: rows,
    pitch_length_m: lengthM,
    pitch_width_m: widthM,
    space = 'pitch',
    frame_width_px: frameW,
    frame_height_px: frameH,
  } = heatmap || {};

  const isImageSpace = space === 'image';

  if (!cells.length) {
    return (
      <EmptyState
        title="No Positional Samples"
        message={
          isImageSpace
            ? 'This player has no tracking rows carrying a pixel position, so there is nothing to bin.'
            : 'This player has no tracking rows with valid pitch coordinates, so there is nothing to plot. A heatmap drawn from zero samples would be a picture of nothing.'
        }
      />
    );
  }

  const cellW = INNER_W / cols;
  const cellH = INNER_H / rows;

  return (
    <div className="dash-pitch-wrap">
      <svg
        className="dash-pitch"
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        role="img"
        aria-label={isImageSpace ? 'Player image-space density grid' : 'Player position heatmap'}
      >
        {/* No pitch markings in image space. The cells are frame pixels, so
            drawing a halfway line and penalty boxes under them would assert
            a correspondence between the two that does not exist -- the
            camera pans, and cell (10, 6) is not the centre circle. */}
        {isImageSpace ? (
          <rect
            x={PAD} y={PAD} width={INNER_W} height={INNER_H}
            fill="rgba(0, 212, 255, 0.02)" stroke="rgba(148, 163, 184, 0.25)" strokeWidth="1"
          />
        ) : (
          <PitchOutline gradientId="heatmap-grad" />
        )}
        <g>
          {cells.map((cell) => {
            const density = Math.max(0, Math.min(1, Number(cell.density) || 0));
            if (density <= 0) return null;
            return (
              <rect
                key={`${cell.grid_x}-${cell.grid_y}`}
                x={PAD + cell.grid_x * cellW}
                y={PAD + cell.grid_y * cellH}
                width={cellW}
                height={cellH}
                fill={density > 0.66 ? '#f472b6' : density > 0.33 ? '#8b5cf6' : '#00d4ff'}
                opacity={0.12 + density * 0.62}
              >
                <title>{`cell (${cell.grid_x}, ${cell.grid_y}) — ${cell.count} samples, density ${density.toFixed(2)}`}</title>
              </rect>
            );
          })}
        </g>
      </svg>
      <div className="dash-pitch-legend">
        <span className="dash-pitch-legend-label">Density</span>
        <span className="dash-pitch-swatch" style={{ background: '#00d4ff' }} /> low
        <span className="dash-pitch-swatch" style={{ background: '#8b5cf6' }} /> mid
        <span className="dash-pitch-swatch" style={{ background: '#f472b6' }} /> high
        <span className="dash-pitch-legend-meta">
          {/* The extent that was actually binned. Reporting the pitch model
              here in image space would have labelled a grid of frame pixels
              "105m × 68m", which is the one claim this view must never make. */}
          {isImageSpace
            ? `${cols}×${rows} grid · ${frameW ?? '?'}×${frameH ?? '?'} px frame`
            : `${cols}×${rows} grid · ${lengthM}m × ${widthM}m`}
        </span>
      </div>
    </div>
  );
}

/**
 * A single tracking frame as a 2D pitch minimap.
 *
 * Only players with non-null pitch_x_m/pitch_y_m are drawn. When calibration
 * is invalid every player in the frame has null pitch coordinates, and the
 * honest result is an empty pitch with a stated reason -- pixel coordinates
 * are NOT substituted in, because a pixel position plotted on a metric pitch
 * is a fabricated location that looks exactly like a real one.
 */
export function TrackingMinimap({ frame, calibration, pitchLengthM = 105, pitchWidthM = 68 }) {
  const players = frame?.players || [];
  const positioned = players.filter(
    (player) => player.pitch_x_m !== null && player.pitch_x_m !== undefined
      && player.pitch_y_m !== null && player.pitch_y_m !== undefined,
  );


  return (
    <div className="dash-pitch-wrap">
      <svg className="dash-pitch" viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} role="img" aria-label="Tracking minimap">
        <PitchOutline gradientId="minimap-grad" />
        {positioned.map((player) => {
          const { x, y } = projectToPitch(player.pitch_x_m, player.pitch_y_m, pitchLengthM, pitchWidthM);
          return (
            <g key={`${player.player_id}-${player.team_id ?? 'na'}`}>
              <circle cx={x} cy={y} r="5" fill={teamColour(player.team_id)} opacity="0.85">
                <title>
                  {`player ${player.player_id}${player.team_id ? ` · team ${player.team_id}` : ''} — `
                    + `${player.pitch_x_m.toFixed(1)}m, ${player.pitch_y_m.toFixed(1)}m`}
                </title>
              </circle>
              <text x={x} y={y - 8} className="dash-pitch-label" textAnchor="middle">
                {player.player_id}
              </text>
            </g>
          );
        })}
      </svg>

      {!positioned.length ? (
        <p className="dash-pitch-note dash-absent">
          {players.length
            ? `Calibration unavailable — ${players.length} player${players.length === 1 ? '' : 's'} detected in this frame, but none have pitch coordinates. `
              + 'Pixel positions are not projected onto the pitch, because an unprojected position drawn here would be indistinguishable from a real one.'
            : 'No tracked players in this frame.'}
        </p>
      ) : null}

      {calibration?.note ? <p className="dash-pitch-note">{calibration.note}</p> : null}
    </div>
  );
}

/**
 * The tactical overlay stacked over the <video>.
 *
 * COORDINATE SPACE -- the thing this component gets right and used not to.
 *
 * Every pixel_x/pixel_y the backend serves is in the SOURCE clip's pixel
 * space. The viewBox therefore has to be the source frame's extent, which is
 * `sourceWidth`/`sourceHeight` -- NOT the played <video> element's intrinsic
 * size.
 *
 * Those two are the same number only for the raw upload. The processed clip
 * is re-encoded at OVERLAY_MAX_WIDTH (960 by default), so a 1280x720 source
 * plays back as 960x540: sizing the viewBox from the video element made every
 * marker land at 1.33x its true position and pushed most of them off the
 * right-hand edge. That is precisely the "overlay is attached to the wrong
 * video" failure -- the overlay only ever lined up because it was pinned to
 * the raw clip, and it was pinned to the raw clip because it could not
 * survive being moved.
 *
 * With the source extent as the viewBox and preserveAspectRatio="none", the
 * same overlay is correct over EITHER source, because both are the same
 * framing at different scales and the SVG maps its viewBox onto whatever box
 * CSS gives it.
 */
export function TrackingOverlaySvg({
  frame,
  sourceWidth,
  sourceHeight,
  showPlayers = true,
  showBall = true,
  // Track-id text. Suppressed over the annotated render, where the pipeline
  // has already burned the same ids in: drawing them twice would put two
  // copies of the same number on one player and invite reading them as two
  // independent identifications. The ring is still drawn, because it is the
  // live layer and it is what stays synchronised with currentTime -- it is
  // visually a ring against the render's rectangle, so the two are not
  // mistakable for each other.
  showLabels = true,
  focusPlayerId = null,
}) {
  const players = frame?.players || [];
  const ball = frame?.ball;

  if (!sourceWidth || !sourceHeight) return null;

  // Stroke and glyph sizes are expressed in viewBox units, so they scale with
  // the source rather than with however large the player happens to be
  // rendered. A 4K source and a 720p source get proportionally equal marks.
  const unit = sourceWidth / 100;

  return (
    <svg
      className="dash-video-overlay"
      viewBox={`0 0 ${sourceWidth} ${sourceHeight}`}
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      {showPlayers
        ? players.map((player) => {
            const focused = focusPlayerId !== null && player.player_id === focusPlayerId;
            return (
              <g key={player.player_id}>
                <circle
                  cx={player.pixel_x}
                  cy={player.pixel_y}
                  r={focused ? unit * 1.5 : unit * 0.9}
                  fill="none"
                  stroke={focused ? '#ffffff' : teamColour(player.team_id)}
                  strokeWidth={focused ? unit * 0.22 : unit * 0.14}
                  opacity={focused ? 1 : 0.9}
                />
                {showLabels ? (
                  <text
                    x={player.pixel_x}
                    y={player.pixel_y - unit * 1.3}
                    className="dash-overlay-label"
                    fontSize={unit * 1.5}
                    textAnchor="middle"
                  >
                    {player.player_id}
                  </text>
                ) : null}
              </g>
            );
          })
        : null}

      {showBall && ball && ball.pixel_x !== null && ball.pixel_x !== undefined ? (
        <g>
          <circle
            cx={ball.pixel_x}
            cy={ball.pixel_y}
            r={unit * 0.85}
            fill="none"
            stroke="#fbbf24"
            strokeWidth={unit * 0.18}
            opacity="0.95"
          />
          <circle cx={ball.pixel_x} cy={ball.pixel_y} r={unit * 0.25} fill="#fbbf24" />
        </g>
      ) : null}
    </svg>
  );
}
