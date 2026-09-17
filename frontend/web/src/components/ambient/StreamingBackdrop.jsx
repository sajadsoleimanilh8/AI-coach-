import { useEffect, useState } from 'react';

/* ──────────────────────────────────────────────────────────────
 * STREAMING BACKDROP
 *
 * Adapted from the `streaming-text` component. The mechanic is kept
 * verbatim — a token array, a word-interval timer, a count that walks it,
 * a hold at the end, and a trailing caret — but everything that made the
 * original a *content* component has been removed, because this is a
 * decorative layer behind the hero:
 *
 *   - no citations / source chips  ─┐  all three imply the text is a real
 *   - no copy/retry/vote actions   ─┤  answer with real provenance. It is
 *   - no follow-up prompts         ─┘  not. It is set dressing.
 *
 * The copy is deliberately capability description ("associating identities
 * across occlusions"), never a measurement. No number appears here that a
 * reader could mistake for a reading off the pipeline — an ambient layer is
 * the last place a figure should be invented.
 *
 * Restyled to the page's own language: JetBrains Mono, --neon-blue, and the
 * "// LABEL" mono idiom. It is aria-hidden and pointer-events:none, so it is
 * invisible to assistive tech and never intercepts a click.
 * ────────────────────────────────────────────────────────────── */

const WORD_MS = 85;
const HOLD_MS = 2800;
const CLEAR_MS = 620;

const LINES = [
  'Detecting players, ball, and referees frame by frame',
  'Associating identities across occlusions and crowded boxes',
  'Projecting image pixels onto metric pitch coordinates',
  'Reading shape, spacing, and pressing structure over time',
  'Turning geometry into tactical explanation',
];

const TOKEN_LINES = LINES.map((line) => line.split(' '));

/**
 * Runs only while the landing view is on screen and the tab is focused.
 *
 * Two independent reasons, both real: index.html hides the hero outright in
 * dashboard view (so this would be animating into a `display:none` subtree),
 * and a backgrounded tab still fires setTimeout. Neither is visible to
 * anyone, and both keep the main thread busy.
 */
function useBackdropActive() {
  const [active, setActive] = useState(true);

  useEffect(() => {
    const read = () =>
      setActive(document.body.dataset.view !== 'dashboard' && !document.hidden);

    read();
    window.addEventListener('ssc:view', read);
    document.addEventListener('visibilitychange', read);
    return () => {
      window.removeEventListener('ssc:view', read);
      document.removeEventListener('visibilitychange', read);
    };
  }, []);

  return active;
}

export default function StreamingBackdrop() {
  const [lineIndex, setLineIndex] = useState(0);
  const [count, setCount] = useState(0);
  const [clearing, setClearing] = useState(false);
  const active = useBackdropActive();

  const tokens = TOKEN_LINES[lineIndex];
  const done = count >= tokens.length;

  useEffect(() => {
    if (!active) return undefined;

    // Three phases: type the line, hold it, then fade it out and advance.
    if (!done) {
      const timer = setTimeout(() => setCount((c) => c + 1), WORD_MS);
      return () => clearTimeout(timer);
    }

    if (!clearing) {
      const timer = setTimeout(() => setClearing(true), HOLD_MS);
      return () => clearTimeout(timer);
    }

    const timer = setTimeout(() => {
      setClearing(false);
      setCount(0);
      setLineIndex((index) => (index + 1) % TOKEN_LINES.length);
    }, CLEAR_MS);
    return () => clearTimeout(timer);
  }, [count, done, clearing, lineIndex, active]);

  return (
    <div className="ssc-backdrop" aria-hidden="true">
      <div className="ssc-backdrop-tag">// Live Pipeline</div>

      <p className={`ssc-backdrop-line ${clearing ? 'is-clearing' : ''}`.trim()}>
        {tokens.slice(0, count).map((word, index) => (
          <span
            // Index keying is correct here and nowhere else: the list only
            // ever grows from the front, and remounting a word is exactly
            // what re-triggers its resolve animation on a new line.
            key={`${lineIndex}-${index}`}
            className="ssc-backdrop-word"
          >
            {word}{' '}
          </span>
        ))}
        {!done && !clearing ? <span className="ssc-backdrop-caret" /> : null}
      </p>

      <div className="ssc-backdrop-rail">
        {TOKEN_LINES.map((_, index) => (
          <span
            key={index}
            className={`ssc-backdrop-tick ${index === lineIndex ? 'is-active' : ''}`.trim()}
          />
        ))}
      </div>
    </div>
  );
}
