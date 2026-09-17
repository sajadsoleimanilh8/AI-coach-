/**
 * Bridge to the animation system that already exists in index.html.
 *
 * There is exactly ONE animation system on this page. index.html loads GSAP +
 * ScrollTrigger from a CDN and exposes window.SSCMotion; everything here
 * forwards to it rather than importing a second animation library or
 * hand-rolling a parallel set of transitions.
 *
 * Motion runs at full strength in every view — the dashboard animates exactly
 * as the landing page does. The only fallback path is the one index.html
 * already had: if GSAP itself failed to load (offline, blocked CDN), elements
 * are made visible statically so content is never stuck at opacity 0.
 */

function motion() {
  return typeof window !== 'undefined' ? window.SSCMotion : undefined;
}

/**
 * Fade + translateY entrance with a 0.1s stagger over the [data-anim]
 * children of `root` — the same shape as animateFeatures()/animateProcess()
 * on the landing page.
 */
export function animateIn(root) {
  motion()?.animateIn?.(root);
}

/** Re-measure ScrollTrigger after a view toggle changes the page height. */
export function refreshScroll() {
  motion()?.refresh?.();
}

/**
 * The landing page's stat count-up, reused verbatim (60 steps, 16ms tick,
 * one decimal place for non-integers). Returns a cancel function.
 */
export function countUp(target, onValue) {
  const bridge = motion();
  if (bridge?.countUp) return bridge.countUp(target, onValue);

  // index.html has not run yet (or was edited to drop the bridge). Show the
  // real value rather than leaving a stat reading 0.
  const value = Number(target);
  onValue(Number.isFinite(value) ? String(value) : null);
  return () => {};
}
