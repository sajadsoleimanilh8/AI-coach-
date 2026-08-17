import { useEffect, useState } from 'react';

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

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
