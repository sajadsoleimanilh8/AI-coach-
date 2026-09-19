import { describe, expect, it } from 'vitest';

import { PREFETCH_MARGIN_FRAMES, WINDOW_FRAMES, needsNewWindow, windowContaining } from './paging.js';

/** Apply the paging rule until it stops moving; return every window visited. */
function settle(playhead, maxFrame = null, start = 0) {
  const visited = [start];
  let windowStart = start;
  for (let i = 0; i < 10; i += 1) {
    if (!needsNewWindow(playhead, windowStart)) return visited;
    const next = windowContaining(playhead, maxFrame);
    if (next === windowStart) return visited;
    windowStart = next;
    visited.push(windowStart);
  }
  throw new Error(`never settled for playhead ${playhead}: ${visited.join(' -> ')}`);
}

describe('tracking window paging', () => {
  it('settles for every playhead in a long clip, from any starting window', () => {
    // Regression: playheads in the last PREFETCH_MARGIN_FRAMES of a window
    // (110-149, 260-299, ...) used to flip 0 <-> 150 forever.
    for (let playhead = 0; playhead < 2000; playhead += 1) {
      for (const start of [0, 150, 300, 1500]) {
        expect(() => settle(playhead, null, start)).not.toThrow();
      }
    }
  });

  it('moves at most once per playhead change', () => {
    for (let playhead = 0; playhead < 2000; playhead += 7) {
      expect(settle(playhead).length).toBeLessThanOrEqual(2);
    }
  });

  it('always loads a window that contains the playhead', () => {
    for (let playhead = 0; playhead < 2000; playhead += 1) {
      const start = windowContaining(playhead, null);
      expect(start).toBeLessThanOrEqual(playhead);
      expect(playhead).toBeLessThan(start + WINDOW_FRAMES);
    }
  });

  it('keeps look-back behind the playhead and the rest ahead', () => {
    expect(windowContaining(500, null)).toBe(500 - PREFETCH_MARGIN_FRAMES);
    expect(windowContaining(10, null)).toBe(0); // never negative
  });

  it('pages before the loaded data runs out, not after', () => {
    const lastSafe = WINDOW_FRAMES - PREFETCH_MARGIN_FRAMES - 1;
    expect(needsNewWindow(lastSafe, 0)).toBe(false);
    expect(needsNewWindow(lastSafe + 1, 0)).toBe(true);
  });

  it('pages after a backwards seek out of the window', () => {
    expect(needsNewWindow(100, 300)).toBe(true);
    expect(settle(100, null, 300).at(-1)).toBe(windowContaining(100, null));
  });

  it('never starts a window past the last frame', () => {
    expect(windowContaining(5000, 374)).toBe(374);
  });
});
