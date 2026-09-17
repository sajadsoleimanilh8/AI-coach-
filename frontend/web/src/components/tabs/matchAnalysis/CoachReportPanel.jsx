import { useAsync } from '../../../hooks/useAsync.js';
import { Slab } from '../../ui/Primitives.jsx';
import { EmptyState, ErrorState, LoadingState } from '../../ui/States.jsx';
import { CoachReport } from '../../ui/CoachReport.jsx';
import nexus, { NEXUS_UNREACHABLE } from '../../../api/nexus.js';

/**
 * The coach report is served by NEXUS (:8100), not the core backend
 * (:8000), and it is built only after processing completes -- so this panel
 * is reloaded on `dataEpoch` and degrades on its own when that service is
 * down, rather than taking the rest of the tab with it.
 */
export function CoachReportPanel({ matchId, dataEpoch }) {
  const report = useAsync(
    (signal) => nexus.getCoachReport(matchId, signal),
    [matchId, dataEpoch],
    { enabled: Boolean(matchId) },
  );

  if (!matchId) {
    return (
      <EmptyState
        title="No Match Linked"
        message="The coach report is served per match. Upload a clip to get a match_id."
      />
    );
  }

  if (report.status === 'loading') {
    return <LoadingState label="Building coach report" rows={4} />;
  }

  if (report.status === 'error') {
    const status = report.error?.status;
    // 503 is the documented "no provider reachable" answer, and the coach
    // is downstream of processing -- neither is a broken analysis view.
    if (status === 503 || status === 0) {
      return (
        <EmptyState
          title="Coach Report Unavailable"
          message={
            status === 0
              ? NEXUS_UNREACHABLE
              : 'NEXUS is running but no model provider was reachable, so the narrative could not be generated. The measured analysis above is unaffected.'
          }
          action={<button type="button" onClick={report.reload}>Retry</button>}
        />
      );
    }
    return <ErrorState error={report.error} onRetry={report.reload} title="Coach Report Failed" />;
  }

  return (
    <Slab>
      <CoachReport report={report.data} />
    </Slab>
  );
}
