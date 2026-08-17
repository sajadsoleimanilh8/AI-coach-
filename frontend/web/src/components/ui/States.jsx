                                                                                                                                                                                                                                                                                                                                                                                                                                        

                                                                    
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

                                                                                                                                                               
export function EmptyState({ title = 'No Data', message, action }) {
  return (
    <div className="dash-state dash-state-empty">
      <div className="dash-state-label is-empty">{title}</div>
      {message ? <p className="dash-state-message">{message}</p> : null}
      {action ? <div className="dash-state-action">{action}</div> : null}
    </div>
  );
}

                                                                             
                                                                           
                                                                              
                                                    
                                     

                                                                                                                                                                                                                                                                           
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
