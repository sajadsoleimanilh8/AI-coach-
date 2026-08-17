import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { animateIn, countUp } from '../../lib/motion.js';

                                                                                                                                                                                                                                                                                                  

                                                                        
                                                                         
                                                                        

                                                                                                                                                                                                          
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

                                                                          
export function SectionHeader({ tag, title, meta }) {
  return (
    <header className="dash-section-head" data-anim>
      <div className="section-tag dash-tag">{tag}</div>
      <h2 className="section-headline dash-headline">{title}</h2>
      {meta ? <p className="dash-section-meta">{meta}</p> : null}
    </header>
  );
}

                                                                              
export function SubHeading({ children, action }) {
  return (
    <div className="dash-subhead" data-anim>
      <h3>{children}</h3>
      {action ? <div className="dash-subhead-action">{action}</div> : null}
    </div>
  );
}

                                                                        
                                                                         
                                                                        

                                                                                                                                                                      
export function PanelGrid({ children, columns = 3, className = '' }) {
  return (
    <div className={`dash-grid dash-grid-${columns} ${className}`.trim()}>{children}</div>
  );
}

                                                                              
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

                                                                       
export function Slab({ children, className = '', anim = true }) {
  return (
    <div className={`dash-slab ${className}`.trim()} {...(anim ? { 'data-anim': '' } : {})}>
      {children}
    </div>
  );
}

                                                                        
                                                                         
                                                                        

                                                                                                                                                                                                                                                                                                                                                 
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

                                                                   
export function StatRow({ children }) {
  return (
    <div className="dash-stat-row" data-anim>
      {children}
    </div>
  );
}

                                                                                                             
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

                                                                        
                                                                         
                                                                        

const RISK_TONE = { low: 'good', moderate: 'warn', high: 'bad' };

                                                                                  
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

                                                                               
export function MetaRow({ label, children }) {
  return (
    <div className="dash-meta-row">
      <span className="dash-meta-key">{label}</span>
      <span className="dash-meta-val">{children}</span>
    </div>
  );
}

                                                                        
                                                                         
                                                                        

                                                                                                                                                                                                              
export function Disclaimer({ children }) {
  if (!children) return null;
  return (
    <p className="dash-disclaimer" role="note">
      <span className="dash-disclaimer-key">Disclaimer</span>
      {children}
    </p>
  );
}

                                                                                                                                                                                                                                                                                           
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
