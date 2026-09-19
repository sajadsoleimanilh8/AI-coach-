// Tracking-window paging for the playback overlay (PlaybackPanel.jsx).

/** Frames per tracking request. Mirrors the server's own default window. */
export const WINDOW_FRAMES = 150;

/**
 * How close to the end of the loaded window playback may get before the next
 * one is fetched. 40 frames is ~1.6s at 25fps -- enough time for the request
 * to land before the overlay would otherwise run out of data.
 */
export const PREFETCH_MARGIN_FRAMES = 40;

/**
 * The tracking window to load for a playhead: it always CONTAINS the
 * playhead, with PREFETCH_MARGIN_FRAMES of look-back and the rest ahead.
 *
 * The previous paging rule stepped to the next aligned window as soon as the
 * playhead was within the margin of the end -- which moved the window past
 * the playhead (0 -> 150 at frame 140), after which the backward-seek rule
 * pulled it straight back (150 -> 0). For the last 40 frames of every window
 * the two rules flipped it on every render, re-requesting tracking each time.
 * A window that contains the playhead satisfies both rules, so it settles.
 */
export function windowContaining(playhead, maxFrame) {
  const start = Math.max(0, playhead - PREFETCH_MARGIN_FRAMES);
  return maxFrame === null ? start : Math.min(start, maxFrame);
}

/** True when the loaded window no longer serves this playhead. */
export function needsNewWindow(playhead, windowStart) {
  return playhead < windowStart || playhead >= windowStart + WINDOW_FRAMES - PREFETCH_MARGIN_FRAMES;
}
