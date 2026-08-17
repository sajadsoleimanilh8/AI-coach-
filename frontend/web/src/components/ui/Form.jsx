import { useId } from 'react';

                                                                                                                                                                                                                            

export function FormSection({ title, hint, children, columns = 2 }) {
  return (
    <fieldset className="dash-fieldset" data-anim>
      <legend className="dash-legend">{title}</legend>
      {hint ? <p className="dash-fieldset-hint">{hint}</p> : null}
      <div className={`dash-field-grid dash-field-grid-${columns}`}>{children}</div>
    </fieldset>
  );
}

function FieldShell({ id, label, hint, error, children, wide }) {
  return (
    <div className={`dash-field ${wide ? 'is-wide' : ''}`.trim()}>
      <label className="dash-label" htmlFor={id}>
        {label}
      </label>
      {children}
      {hint ? <p className="dash-hint">{hint}</p> : null}
      {error ? <p className="dash-field-error">{error}</p> : null}
    </div>
  );
}

                                                                                                                                                                                                                                            
export function ScaleField({ label, value, onChange, min = 1, max = 10, hint, lowLabel, highLabel, error }) {
  const id = useId();
  return (
    <FieldShell id={id} label={label} hint={hint} error={error}>
      <div className="dash-scale">
        <input
          id={id}
          className="dash-range"
          type="range"
          min={min}
          max={max}
          step={1}
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
        />
        <output className="dash-scale-out" htmlFor={id}>
          {value}
          <span className="dash-scale-max">/{max}</span>
        </output>
      </div>
      {lowLabel || highLabel ? (
        <div className="dash-scale-ends">
          <span>{lowLabel}</span>
          <span>{highLabel}</span>
        </div>
      ) : null}
    </FieldShell>
  );
}

export function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  hint,
  error,
  placeholder,
  nullable = false,
  suffix,
}) {
  const id = useId();
  return (
    <FieldShell id={id} label={label} hint={hint} error={error}>
      <div className="dash-input-wrap">
        <input
          id={id}
          className="dash-input"
          type="number"
          inputMode="decimal"
          min={min}
          max={max}
          step={step}
          placeholder={placeholder}
          value={value === null || value === undefined ? '' : value}
          onChange={(event) => {
            const raw = event.target.value;
            if (raw === '') {
                                                                   
                                                                           
              onChange(nullable ? null : '');
              return;
            }
            onChange(Number(raw));
          }}
        />
        {suffix ? <span className="dash-input-suffix">{suffix}</span> : null}
      </div>
    </FieldShell>
  );
}

export function TextField({ label, value, onChange, hint, error, placeholder, type = 'text', wide }) {
  const id = useId();
  return (
    <FieldShell id={id} label={label} hint={hint} error={error} wide={wide}>
      <input
        id={id}
        className="dash-input"
        type={type}
        placeholder={placeholder}
        value={value ?? ''}
        onChange={(event) => onChange(event.target.value === '' ? null : event.target.value)}
      />
    </FieldShell>
  );
}

export function TextAreaField({ label, value, onChange, hint, placeholder, rows = 3 }) {
  const id = useId();
  return (
    <FieldShell id={id} label={label} hint={hint} wide>
      <textarea
        id={id}
        className="dash-input dash-textarea"
        rows={rows}
        placeholder={placeholder}
        value={value ?? ''}
        onChange={(event) => onChange(event.target.value === '' ? null : event.target.value)}
      />
    </FieldShell>
  );
}

export function SelectField({ label, value, onChange, options, hint, error, wide }) {
  const id = useId();
  return (
    <FieldShell id={id} label={label} hint={hint} error={error} wide={wide}>
      <select
        id={id}
        className="dash-input dash-select"
        value={value ?? ''}
        onChange={(event) => onChange(event.target.value === '' ? null : event.target.value)}
      >
        {options.map((option) => (
          <option key={String(option.value)} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </FieldShell>
  );
}

export function ToggleField({ label, value, onChange, hint }) {
  const id = useId();
  return (
    <div className="dash-field dash-field-toggle">
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={Boolean(value)}
        className={`dash-toggle ${value ? 'is-on' : ''}`.trim()}
        onClick={() => onChange(!value)}
      >
        <span className="dash-toggle-track">
          <span className="dash-toggle-thumb" />
        </span>
        <span className="dash-toggle-label">{label}</span>
      </button>
      {hint ? <p className="dash-hint">{hint}</p> : null}
    </div>
  );
}

                                                                        
export function SubmitButton({ children, busy, busyLabel = 'Submitting', disabled, onClick, type = 'submit' }) {
  return (
    <button
      type={type}
      className="btn-primary dash-submit"
      disabled={busy || disabled}
      onClick={onClick}
    >
      {busy ? (
        <>
          <span className="dash-submit-pulse" aria-hidden="true" />
          {busyLabel}…
        </>
      ) : (
        children
      )}
    </button>
  );
}
