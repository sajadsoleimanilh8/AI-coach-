/**
 * The three states every async view in this dashboard is required to have.
 * A view that renders nothing while it waits, or renders an empty grid when
 * a request failed, is a bug -- these components exist so that outcome is not
 * reachable by accident.
 *
 * The loading language matches the landing page's own "Processing..." idiom:
 * a neon scanning bar and mono uppercase status text, not a generic spinner.
 */

/** Skeleton + scanning bar, in the page's own progress language. */
export function LoadingState({ label = 'Loading', detail, rows = 3 }) {
  return (
    <div className="dash-state dash-state-loading" role="status" aria-live="polite">
      <div className="dash-scanbar">
        <span className="dash-scanbar-fill" />
      </div>
      <div className="dash-state-label">{label}…</div>
      {detail ? <p className="dash-state-detail">{detail}</p> : null}
      <div className="dash-skeleton-stack" aria-hidden="true">
        {Array.from({ length: rows }, (_, index) => (
          <div key={index} className="dash-skeleton" style={{ width: `${100 - index * 12}%` }} />
        ))}
      </div>
    </div>
  );
}

/**
 * An error the user can act on. `error.unreachable` (status 0) means the
 * request never reached the service at all, which is a different instruction
 * ("start the process") from a 4xx ("this data does not exist yet") -- so the
 * two are not collapsed into one message.
 */
export function ErrorState({ error, onRetry, title }) {
  const message =
    (error && (error.message || String(error))) || 'Something went wrong.';
  const unreachable = Boolean(error?.unreachable);
  const heading = title || (unreachable ? 'Service Unreachable' : 'Request Failed');

  return (
    <div className="dash-state dash-state-error" role="alert">
      <div className="dash-state-label is-error">{heading}</div>
      <p className="dash-state-message">{message}</p>
      {error?.status ? (
        <p className="dash-state-detail">HTTP {error.status}</p>
      ) : null}
      {onRetry ? (
        <button type="button" className="btn-ghost dash-retry" onClick={onRetry}>
          Retry
        </button>
      ) : null}
    </div>
  );
}

/**
 * "There is genuinely nothing here." Distinct from an error: the request
 * succeeded and the honest answer is that this metric has not been computed.
 */
export function EmptyState({ title = 'No Data', message, action }) {
  return (
    <div className="dash-state dash-state-empty">
      <div className="dash-state-label is-empty">{title}</div>
      {message ? <p className="dash-state-message">{message}</p> : null}
      {action ? <div className="dash-state-action">{action}</div> : null}
    </div>
  );
}

// SelectMatchState used to live here. It moved to ui/MatchPicker.jsx when it
// grew from a sign-post into a component that fetches the match list: this
// module is the leaf every view imports, and having it import a data-fetching
// component would have made the dependency circular
// (States -> MatchPicker -> States).

/**
 * Renders a useAsync result through the three required states in one place,
 * so no tab can forget one.
 *
 * `isEmpty` lets a caller declare that a 200 response is still nothing to
 * show (an empty metrics list), which is an empty state and not a success.
 */
export function AsyncBlock({
  state,
  children,
  loadingLabel,
  loadingDetail,
  loadingRows,
  errorTitle,
  isEmpty,
  emptyTitle,
  emptyMessage,
  emptyAction,
  idle = null,
  onRetry,
}) {
  if (state.status === 'idle') return idle;

  if (state.status === 'loading') {
    return <LoadingState label={loadingLabel} detail={loadingDetail} rows={loadingRows} />;
  }

  if (state.status === 'error') {
    return (
      <ErrorState error={state.error} title={errorTitle} onRetry={onRetry || state.reload} />
    );
  }

  const empty = typeof isEmpty === 'function' ? isEmpty(state.data) : Boolean(isEmpty);
  if (empty) {
    return <EmptyState title={emptyTitle} message={emptyMessage} action={emptyAction} />;
  }

  return typeof children === 'function' ? children(state.data) : children;
}
