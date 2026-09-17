import { useCallback, useRef, useState } from 'react';
import { PanelGrid, SectionHeader, SubHeading, TabSection } from '../ui/Primitives.jsx';
import { AttachExistingMatch } from './matchAnalysis/AttachExistingMatch.jsx';
import { CoachReportPanel } from './matchAnalysis/CoachReportPanel.jsx';
import { EventsPanel } from './matchAnalysis/EventsPanel.jsx';
import { HeatmapPanel } from './matchAnalysis/HeatmapPanel.jsx';
import { LatencyPanel } from './matchAnalysis/LatencyPanel.jsx';
import { PlaybackPanel } from './matchAnalysis/PlaybackPanel.jsx';
import { ProcessingPanel } from './matchAnalysis/ProcessingPanel.jsx';
import { UploadPanel } from './matchAnalysis/UploadPanel.jsx';
import { useCompletionEpoch, useJobStatus } from './matchAnalysis/jobStatus.js';

/**
 * TAB 1 -- the entry point. Upload -> processing pipeline -> playback with a
 * tracking overlay, plus heatmaps and the measured per-stage latency report.
 *
 * This is the only tab that can set `matchId`; it lifts the match_id out of
 * the upload response so every other match-scoped tab has something to read.
 */
export default function TabMatchAnalysis({ matchId, videoId, jobId, filename, onUploadComplete, onAttachMatch, goToTab }) {
  // FIXED: the job poller used to live inside ProcessingPanel, so nothing
  // outside that panel ever learned that the pipeline had finished. An upload
  // sets matchId/jobId the moment the job is *queued*, which is when the three
  // panels below fire their one and only fetch -- against a match that has no
  // tracking rows, no player metrics, and no latency report yet. They
  // correctly rendered "No Tracking Rows" / "Latency Report Not Ready", then
  // kept rendering it after the worker wrote all of it, because their deps
  // never changed again. The result read as a broken pipeline when the
  // pipeline had in fact succeeded.
  //
  // The poller now lives here and publishes `dataEpoch`, which increments on
  // each transition INTO `completed`. It is a dep of every panel that reads
  // pipeline output, so those panels refetch exactly once, when there is
  // finally something new to read.
  const job = useJobStatus(jobId);
  const dataEpoch = useCompletionEpoch(jobId, job.data?.status);

  // The seek channel between the events timeline and the player. A bare
  // number would not re-fire when the same event is clicked twice -- the
  // state would be unchanged and the effect would not run -- so each request
  // carries its own nonce and the player depends on that.
  const [seekRequest, setSeekRequest] = useState(null);
  const seekNonce = useRef(0);
  const requestSeek = useCallback((timestamp) => {
    if (!Number.isFinite(timestamp)) return;
    seekNonce.current += 1;
    setSeekRequest({ timestamp, nonce: seekNonce.current });
  }, []);

  return (
    <TabSection animKey="match">
      <SectionHeader
        tag="Match Analysis"
        title={<>Ingest &amp;<br />Track</>}
        meta="Upload a clip, watch the real pipeline run, then read the tracking, heatmaps, and measured stage timings it produced."
      />

      <PanelGrid columns={2}>
        <UploadPanel onUploadComplete={onUploadComplete} />
        <ProcessingPanel job={job} jobId={jobId} filename={filename} />
      </PanelGrid>

      <AttachExistingMatch
        matchId={matchId}
        attachedWithoutVideo={Boolean(matchId) && !videoId}
        onAttachMatch={onAttachMatch}
      />

      <SubHeading>Processed Video &amp; Synchronised Overlay</SubHeading>
      <PlaybackPanel
        videoId={videoId}
        matchId={matchId}
        dataEpoch={dataEpoch}
        seekRequest={seekRequest}
      />

      <SubHeading>Detected Events</SubHeading>
      <EventsPanel matchId={matchId} dataEpoch={dataEpoch} onSeek={requestSeek} />

      <SubHeading>Positional Heatmap</SubHeading>
      <HeatmapPanel matchId={matchId} goToTab={goToTab} dataEpoch={dataEpoch} />

      <SubHeading>Measured Pipeline Latency</SubHeading>
      <LatencyPanel jobId={jobId} dataEpoch={dataEpoch} />

      <SubHeading>LLM Coach Report</SubHeading>
      <CoachReportPanel matchId={matchId} dataEpoch={dataEpoch} />
    </TabSection>
  );
}
