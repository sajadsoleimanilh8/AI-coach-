import { Component } from 'react';

import { ErrorState } from './States.jsx';

/**
 * Contains a render crash to the tab that caused it.
 *
 * Without this, an exception thrown while rendering any one tab unmounts the
 * whole dashboard root: the navigation disappears with it, and the only way
 * back is a full page reload. Wrapped around the active tab (keyed by tab id in
 * DashboardApp), a crash replaces just that tab's content, the other tabs stay
 * reachable, and switching tabs clears the error.
 *
 * Only render-time errors are caught here. Failed requests are not exceptions
 * at this level: useAsync turns them into an `error` state that each panel
 * renders itself.
 */
export class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
    this.reset = this.reset.bind(this);
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // Kept visible in the console so the component stack is not lost.
    console.error('Dashboard tab crashed:', error, info?.componentStack);
  }

  reset() {
    this.setState({ error: null });
  }

  render() {
    const { error } = this.state;
    if (error) {
      return (
        <ErrorState
          title={this.props.title || 'This Tab Hit an Error'}
          error={error}
          onRetry={this.reset}
        />
      );
    }
    return this.props.children;
  }
}
