import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { animateIn, countUp } from '../../lib/motion.js';

/**
 * The dashboard's building blocks, all of them restatements of patterns that
 * already exist in index.html -- .section-tag / .section-headline,
 * .feature-card's hairline grid, .stat-item / .stat-num, and the angular
 * clip-path buttons. Nothing here invents a new visual idiom.
 */

/* ------------------------------------------------------------------ */
/* Section scaffolding                                                 */
/* ------------------------------------------------------------------ */

/**
 * Wraps a tab's contents and runs the landing page's entrance animation over
 * its [data-anim] children on mount, using the same GSAP timeline shape as
 * animateFeatures() / animateProcess().
 */
export function TabSection({ children, animKey }) {
  const ref = useRef(null);

  useLayoutEffect(() => {
    animateIn(ref.current);
  }, [animKey]);

  return (
    <div className="dash-section" ref={ref}>
      {children}
    </div>
  );
}

/** "// LABEL" + Orbitron headline -- .section-tag / .section-headline. */
export function SectionHeader({ tag, title, meta }) {
  return (
    <header className="dash-section-head" data-anim>
      <div className="section-tag dash-tag">{tag}</div>
      <h2 className="section-headline dash-headline">{title}</h2>
      {meta ? <p className="dash-section-meta">{meta}</p> : null}
    </header>
  );
}

/** Sub-heading inside a tab, for grouping panels under one section header. */
export function SubHeading({ children, action }) {
  return (
    <div className="dash-subhead" data-anim>
      <h3>{children}</h3>
      {action ? <div className="dash-subhead-action">{action}</div> : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Cards                                                               */
/* ------------------------------------------------------------------ */

/**
 * The .features-grid shape: 1.5px hairline gaps over a neon-tinted
 * background, so the seams between cards read as glowing rules rather than
 * as padding.
 */
export function PanelGrid({ children, columns = 3, className = '' }) {
  return (
    <div className={`dash-grid dash-grid-${columns} ${className}`.trim()}>{children}</div>
  );
}

/** One cell of a PanelGrid -- .feature-card, hover gradient line included. */
export function Panel({ number, title, subtitle, children, tone = '', className = '' }) {
  return (
    <section
      className={`dash-card ${tone ? `dash-card-${tone}` : ''} ${className}`.trim()}
      data-anim
    >
      {number ? <div className="feature-number dash-card-number">{number}</div> : null}
      {title ? <h3 className="dash-card-title">{title}</h3> : null}
      {subtitle ? <p className="dash-card-subtitle">{subtitle}</p> : null}
      <div className="dash-card-body">{children}</div>
      <div className="feature-hover-line" />
    </section>
  );
}

/** A standalone bordered block that is not part of a hairline grid. */
export function Slab({ children, className = '', anim = true }) {
  return (
    <div className={`dash-slab ${className}`.trim()} {...(anim ? { 'data-anim': '' } : {})}>
      {children}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Numbers                                                             */
/* ------------------------------------------------------------------ */

/**
 * The .stat-item / .stat-num headline number, animated with the landing
 * page's exact count-up.
 *
 * `value === null | undefined` is NOT rendered as 0. A readiness score of
 * zero and a readiness score that was never computed look identical as "0",
 * and only one of them is a real reading -- so the absent case gets words.
 */
export function StatNum({ value, label, absentLabel = 'Not Yet Computed', suffix = '', tone }) {
  const [display, setDisplay] = useState('0');
  const absent = value === null || value === undefined || Number.isNaN(Number(value));

  useEffect(() => {
    if (absent) return undefined;
    return countUp(value, setDisplay);
  }, [value, absent]);

  return (
    <div className={`dash-stat ${tone ? `dash-stat-${tone}` : ''}`.trim()}>
      {absent ? (
        <span className="dash-stat-absent">{absentLabel}</span>
      ) : (
        <span className="stat-num dash-stat-num">
          {display}
          {suffix ? <span className="dash-stat-suffix">{suffix}</span> : null}
        </span>
      )}
      <span className="stat-label dash-stat-label">{label}</span>
    </div>
  );
}

/** A row of StatNums, bordered like #stats on the landing page. */
export function StatRow({ children }) {
  return (
    <div className="dash-stat-row" data-anim>
      {children}
    </div>
  );
}

/**
 * An inline metric value. Same absent-vs-zero rule as StatNum: a missing
 * number says so in words.
 */
export function MetricValue({ value, absentLabel = 'Not Yet Computed', precision = 2, suffix = '' }) {
  if (value === null || value === undefined) {
    return <span className="dash-absent">{absentLabel}</span>;
  }
  if (typeof value === 'number') {
    const text = Number.isInteger(value) ? String(value) : value.toFixed(precision);
    return (
      <span className="dash-metric-value">
        {text}
        {suffix}
      </span>
    );
  }
  if (typeof value === 'boolean') {
    return <span className="dash-metric-value">{value ? 'Yes' : 'No'}</span>;
  }
  if (typeof value === 'object') {
    return <span className="dash-metric-value">{JSON.stringify(value)}</span>;
  }
  const text = String(value);
  if (!text.trim()) return <span className="dash-absent">{absentLabel}</span>;
  return <span className="dash-metric-value">{text}</span>;
}

/* ------------------------------------------------------------------ */
/* Badges & chips                                                      */
/* ------------------------------------------------------------------ */

const RISK_TONE = { low: 'good', moderate: 'warn', high: 'bad' };

/** performance_risk / workload_risk / pressure_risk / mental_performance_risk. */
export function RiskBadge({ risk, label }) {
  if (!risk) {
    return <span className="dash-badge dash-badge-absent">{label ? `${label}: ` : ''}Not Available</span>;
  }
  const tone = RISK_TONE[String(risk).toLowerCase()] || 'neutral';
  return (
    <span className={`dash-badge dash-badge-${tone}`}>
      {label ? <span className="dash-badge-key">{label}</span> : null}
      {String(risk)}
    </span>
  );
}

/**
 * The backend's `confidence` field is a categorical string
 * ("normal", "low_upstream_confidence", "low_sample", ...). It is shown as
 * written -- never mapped onto a percentage the engine did not produce.
 */
export function ConfidenceBadge({ confidence, score }) {
  if (!confidence) {
    return <span className="dash-badge dash-badge-absent">Confidence Not Reported</span>;
  }
  const raw = String(confidence).toLowerCase();
  const tone = raw === 'normal' || raw === 'high' ? 'good' : raw.includes('low') ? 'warn' : 'neutral';
  return (
    <span className={`dash-badge dash-badge-${tone}`} title={`confidence = ${confidence}`}>
      <span className="dash-badge-key">conf</span>
      {String(confidence).replace(/_/g, ' ')}
      {typeof score === 'number' ? ` · ${score.toFixed(2)}` : ''}
    </span>
  );
}

export function Chip({ children, tone = 'neutral', title }) {
  return (
    <span className={`dash-chip dash-chip-${tone}`} title={title}>
      {children}
    </span>
  );
}

/** A key/value line in the mono meta language used across the landing page. */
export function MetaRow({ label, children }) {
  return (
    <div className="dash-meta-row">
      <span className="dash-meta-key">{label}</span>
      <span className="dash-meta-val">{children}</span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Provenance & disclaimers                                            */
/* ------------------------------------------------------------------ */

/**
 * Always-rendered disclaimer text. The health and psychology endpoints both
 * return one and both are explicitly not medical/clinical assessments, so it
 * is never conditional on space or layout.
 */
export function Disclaimer({ children }) {
  if (!children) return null;
  return (
    <p className="dash-disclaimer" role="note">
      <span className="dash-disclaimer-key">Disclaimer</span>
      {children}
    </p>
  );
}

/**
 * A visible method/provenance strip. Used wherever the backend labels its own
 * output -- method: "heuristic_proxy", source: "measured",
 * keypoint_detection_method: "ml_trained" -- so the label travels with the
 * number instead of being dropped on the way to the screen.
 */
export function MethodStrip({ items = [] }) {
  const shown = items.filter((item) => item && item.value !== null && item.value !== undefined && item.value !== '');
  if (!shown.length) return null;
  return (
    <div className="dash-method-strip">
      {shown.map((item) => (
        <span key={item.label} className="dash-method-item">
          <span className="dash-method-key">{item.label}</span>
          <span className={`dash-method-val ${item.tone ? `is-${item.tone}` : ''}`.trim()}>
            {String(item.value)}
          </span>
        </span>
      ))}
    </div>
  );
}

/** The positive / negative factor lists shared by the two questionnaire tabs. */
export function FactorList({ title, items, tone = 'neutral', emptyLabel = 'None reported' }) {
  return (
    <div className={`dash-factors dash-factors-${tone}`}>
      <h4 className="dash-factors-title">{title}</h4>
      {items && items.length ? (
        <ul>
          {items.map((item, index) => (
            <li key={`${item}-${index}`}>{item}</li>
          ))}
        </ul>
      ) : (
        <p className="dash-absent">{emptyLabel}</p>
      )}
    </div>
  );
}

/** A 0-100 bar used for factor scores and progress. */
export function Meter({ value, max = 100, tone = 'blue', label }) {
  const absent = value === null || value === undefined || Number.isNaN(Number(value));
  const pct = absent ? 0 : Math.max(0, Math.min(100, (Number(value) / max) * 100));
  return (
    <div className="dash-meter" role="group" aria-label={label}>
      <div className="dash-meter-track">
        <div className={`dash-meter-fill is-${tone}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="dash-meter-value">
        {absent ? <span className="dash-absent">n/a</span> : Math.round(Number(value))}
      </span>
    </div>
  );
}

/**
 * A collapsible sub_scores breakdown. Rendered as a real disclosure rather
 * than a tooltip because sub_scores is the explainability record behind every
 * headline number and needs to be readable, copyable, and printable.
 */
export function SubScores({ subScores, label = 'Sub-scores' }) {
  const entries = subScores && typeof subScores === 'object' ? Object.entries(subScores) : [];
  if (!entries.length) {
    return <p className="dash-absent dash-subscores-empty">No sub-score breakdown recorded.</p>;
  }
  return (
    <details className="dash-subscores">
      <summary>
        {label}: <span className="dash-subscores-count">{entries.length}</span>
      </summary>
      <dl>
        {entries.map(([key, value]) => (
          <div key={key} className="dash-subscore-row">
            <dt>{key.replace(/_/g, ' ')}</dt>
            <dd>
              <MetricValue value={value} absentLabel="Not Available" />
            </dd>
          </div>
        ))}
      </dl>
    </details>
  );
}
