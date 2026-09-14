import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import api from '../../api/client.js';
import { useAction, useAsync } from '../../hooks/useAsync.js';
import {
  Chip,
  MetaRow,
  MethodStrip,
  MetricValue,
  Panel,
  PanelGrid,
  SectionHeader,
  Slab,
  StatNum,
  StatRow,
  SubHeading,
  TabSection,
} from '../ui/Primitives.jsx';
import { AsyncBlock, EmptyState, ErrorState, LoadingState } from '../ui/States.jsx';
import { NumberField, SubmitButton, TextField } from '../ui/Form.jsx';
import { CoachReport } from '../ui/CoachReport.jsx';
import nexus, { NEXUS_UNREACHABLE } from '../../api/nexus.js';
import { HeatmapPitch, TrackingMinimap, TrackingOverlaySvg } from '../ui/Pitch.jsx';

const TERMINAL = new Set(['completed', 'failed']);
const POLL_MS = 1500;

                                                                                                                                                                                                                                                                                                                                 
export default function TabMatchAnalysis({ matchId, videoId, jobId, filename, onUploadComplete, onAttachMatch, goToTab }) {
                                                                          
                                                                              
                                                                               
                                                                              
                                                                      
                                                                             
                                                                           
                                                                       
                                    
    
                                                                             
                                                                            
                                                                         
                                   
  const job = useJobStatus(jobId);
  const dataEpoch = useCompletionEpoch(jobId, job.data?.status);

                                                                        
                                                                         
                                                                             
                                                          
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

                                                                          
                                                                          
                                                                          

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
/**
 * The coach report is served by NEXUS (:8100), not the core backend
 * (:8000), and it is built only after processing completes -- so this panel
 * is reloaded on `dataEpoch` and degrades on its own when that service is
 * down, rather than taking the rest of the tab with it.
 */
function CoachReportPanel({ matchId, dataEpoch }) {
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


function AttachExistingMatch({ matchId, attachedWithoutVideo, onAttachMatch }) {
  const [draft, setDraft] = useState('');

                                                                             
                                                                          
                                                                            
                                                                  
  const check = useAction((id, signal) => api.getMatchSummary(id, signal));

  const submit = async (event) => {
    event.preventDefault();
    const id = draft.trim();
    if (!id) return;
    const summary = await check.run(id);
    if (summary) onAttachMatch(summary);
  };

  return (
    <Slab>
      <div className="dash-subhead-inline">
        <h4 className="dash-card-title">Or Attach an Already-Processed Match</h4>
        <p className="dash-hint">
          Paste a <code>match_id</code> that has already been through the pipeline. Its video_id and
          job_id are resolved from the backend, so playback, calibration, and the latency report
          work too — nothing here is guessed.
        </p>
      </div>

      <form className="dash-form dash-form-inline" onSubmit={submit}>
        <TextField
          label="Existing match_id"
          value={draft}
          onChange={(value) => setDraft(value ?? '')}
          placeholder="e.g. bb58ad9f-e02a-47f3-9aa3-6ced09d009d0"
        />
        <SubmitButton busy={check.status === 'loading'} busyLabel="Checking" disabled={!draft.trim()}>
          Attach Match
        </SubmitButton>
      </form>

      {check.status === 'error' ? (
        <ErrorState
          error={check.error}
          title={check.error?.notFound ? 'No Such Match' : 'Could Not Attach'}
        />
      ) : null}

      {check.status === 'success' && check.data ? (
        <div className="dash-meta-block">
          <div className="dash-badge-row">
            <Chip tone="good">attached</Chip>
            <code className="dash-code-sm">{check.data.match_id}</code>
          </div>
          <MetaRow label="teams">
            {check.data.home_team || '—'} vs {check.data.away_team || '—'}
          </MetaRow>
          <MetaRow label="video">
            {check.data.video_id ? (
              check.data.video_file_exists ? (
                <Chip tone="good">playable</Chip>
              ) : (
                                                                             
                                                                     
                <Chip tone="warn">row exists, file missing from disk</Chip>
              )
            ) : (
              <span className="dash-absent">no video row</span>
            )}
          </MetaRow>
          <MetaRow label="job">
            {check.data.job_id ? (
              <Chip tone={check.data.job_status === 'completed' ? 'good' : 'neutral'}>
                {check.data.job_status}
              </Chip>
            ) : (
              <span className="dash-absent">no processing job</span>
            )}
          </MetaRow>
        </div>
      ) : null}
    </Slab>
  );
}

                                                                          
                                                                          
                                                                          

function UploadPanel({ onUploadComplete }) {
  const [file, setFile] = useState(null);
  const [team, setTeam] = useState('');
  const [opponent, setOpponent] = useState('');
  const inputRef = useRef(null);

  const upload = useAction((payload, signal) =>
    api.uploadVideo(payload.file, payload.metadata, signal),
  );

  const submit = async (event) => {
    event.preventDefault();
    if (!file) return;

    const metadata = {};
    if (team.trim()) metadata.team = team.trim();
    if (opponent.trim()) metadata.opponent = opponent.trim();

    const result = await upload.run({ file, metadata });
                                                                         
                                                                     
    if (result?.match_id) onUploadComplete(result);
  };

  return (
    <Panel
      number="01"
      title="Upload Match Video"
      subtitle="POST /api/videos/upload — multipart. The clip is written to disk before the pipeline runs, so playback is available immediately."
    >
      <form className="dash-form" onSubmit={submit}>
        <div className="dash-field is-wide">
          <span className="dash-label">Match Clip</span>
          <button
            type="button"
            className="dash-filedrop"
            onClick={() => inputRef.current?.click()}
          >
            {file ? (
              <>
                <span className="dash-filedrop-name">{file.name}</span>
                <span className="dash-filedrop-meta">
                  {(file.size / (1024 * 1024)).toFixed(1)} MB · {file.type || 'unknown type'}
                </span>
              </>
            ) : (
              <>
                <span className="dash-filedrop-name">Select a video file</span>
                <span className="dash-filedrop-meta">mp4 · mpeg · mov · avi</span>
              </>
            )}
          </button>
          <input
            ref={inputRef}
            className="dash-file-input"
            type="file"
            accept="video/mp4,video/mpeg,video/quicktime,video/x-msvideo"
            onChange={(event) => setFile(event.target.files?.[0] || null)}
          />
        </div>

        <div className="dash-field-grid dash-field-grid-2">
          <TextField
            label="Team (optional)"
            value={team}
            onChange={(value) => setTeam(value ?? '')}
            placeholder="Home"
            hint="Stored as Match.home_team. Left blank it stays the honest placeholder “Home”."
          />
          <TextField
            label="Opponent (optional)"
            value={opponent}
            onChange={(value) => setOpponent(value ?? '')}
            placeholder="Away"
            hint="Stored as Match.away_team."
          />
        </div>

        <SubmitButton busy={upload.status === 'loading'} busyLabel="Uploading" disabled={!file}>
          Upload &amp; Queue
        </SubmitButton>
      </form>

      {upload.status === 'error' ? (
        <ErrorState error={upload.error} title="Upload Failed" onRetry={upload.reset} />
      ) : null}

      {upload.status === 'success' && upload.data ? (
        <Slab className="dash-upload-receipt" anim={false}>
          <MetaRow label="match_id">
            <code>{upload.data.match_id}</code>
          </MetaRow>
          <MetaRow label="video_id">
            <code>{upload.data.video_id}</code>
          </MetaRow>
          <MetaRow label="job_id">
            <code>{upload.data.job_id}</code>
          </MetaRow>
          <MetaRow label="status">
            <Chip tone={upload.data.status === 'failed' ? 'bad' : 'good'}>{upload.data.status}</Chip>
          </MetaRow>
          <p className="dash-hint">{upload.data.message}</p>
        </Slab>
      ) : null}
    </Panel>
  );
}

                                                                          
                                                                          
                                                                          

                                                                                                                                                                                                                                                                                                                                                                                                                                                               
function useJobStatus(jobId) {
  const [state, setState] = useState({ status: 'idle', data: null, error: null });
  const [nonce, setNonce] = useState(0);
  const retry = useCallback(() => setNonce((value) => value + 1), []);

  useEffect(() => {
    if (!jobId) {
      setState({ status: 'idle', data: null, error: null });
      return undefined;
    }

    const controller = new AbortController();
    let active = true;
    let timer = null;

    setState({ status: 'loading', data: null, error: null });

    const tick = async () => {
      try {
        const data = await api.getProcessingStatus(jobId, controller.signal);
        if (!active) return;
        setState({ status: 'success', data, error: null });
        if (!TERMINAL.has(data.status)) {
          timer = setTimeout(tick, POLL_MS);
        }
      } catch (error) {
        if (!active || error?.name === 'AbortError') return;
        setState({ status: 'error', data: null, error });
      }
    };

    tick();

    return () => {
      active = false;
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [jobId, nonce]);

  return { ...state, retry };
}

                                                                                                                                                                                                                                                                                                                                                                     
function useCompletionEpoch(jobId, jobStatus) {
  const [epoch, setEpoch] = useState(0);
  const seen = useRef({ jobId: null, status: null });

  useEffect(() => {
    const previous = seen.current;
    seen.current = { jobId, status: jobStatus ?? null };

                                                                             
    if (previous.jobId !== jobId) return;

    if (previous.status && previous.status !== 'completed' && jobStatus === 'completed') {
      setEpoch((value) => value + 1);
    }
  }, [jobId, jobStatus]);

  return epoch;
}

function ProcessingPanel({ job, jobId, filename }) {
  if (!jobId) {
    return (
      <Panel number="02" title="Processing Pipeline" subtitle="GET /api/processing/{job_id}">
        <EmptyState
          title="No Job Queued"
          message="Upload a clip to start a processing job. This panel then polls the real job status until it completes or fails."
        />
      </Panel>
    );
  }

  if (job.status === 'loading' && !job.data) {
    return (
      <Panel number="02" title="Processing Pipeline" subtitle="GET /api/processing/{job_id}">
        <LoadingState label="Reading job status" rows={2} />
      </Panel>
    );
  }

  if (job.status === 'error') {
    return (
      <Panel number="02" title="Processing Pipeline" subtitle="GET /api/processing/{job_id}">
        <ErrorState error={job.error} onRetry={job.retry} />
      </Panel>
    );
  }

  const data = job.data;
  const progress = Math.max(0, Math.min(100, Number(data?.progress) || 0));
  const failed = data?.status === 'failed';
  const done = data?.status === 'completed';

  return (
    <Panel
      number="02"
      title="Processing Pipeline"
      subtitle={filename ? `Job for ${filename}` : 'GET /api/processing/{job_id}'}
      tone={failed ? 'bad' : done ? 'good' : ''}
    >
      <div className="dash-progress" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
        <div className={`dash-progress-fill ${failed ? 'is-failed' : ''} ${!TERMINAL.has(data?.status) ? 'is-live' : ''}`.trim()} style={{ width: `${progress}%` }} />
      </div>

      <div className="dash-progress-head">
        <Chip tone={failed ? 'bad' : done ? 'good' : 'neutral'}>{data?.status}</Chip>
        <span className="dash-progress-pct">{progress}%</span>
      </div>

      {data?.message ? <p className="dash-hint">{data.message}</p> : null}

      {failed && data?.error ? (
        <div className="dash-inline-error" role="alert">
          <span className="dash-inline-error-key">Pipeline error</span>
          {data.error}
        </div>
      ) : null}

      <div className="dash-meta-block">
        <MetaRow label="job_id">
          <code>{data?.job_id}</code>
        </MetaRow>
        <MetaRow label="match_id">
          {data?.match_id ? <code>{data.match_id}</code> : <span className="dash-absent">Not linked</span>}
        </MetaRow>
        <MetaRow label="created">{formatTime(data?.created_at)}</MetaRow>
        <MetaRow label="started">{formatTime(data?.started_at)}</MetaRow>
        <MetaRow label="completed">{formatTime(data?.completed_at)}</MetaRow>
      </div>

      {!TERMINAL.has(data?.status) ? (
        <p className="dash-hint dash-polling">
          <span className="dash-polling-dot" aria-hidden="true" />
          Polling every {POLL_MS / 1000}s until this job reaches a terminal state.
        </p>
      ) : null}
    </Panel>
  );
}

                                                                          
                                                                          
                                                                          

                                                                            
const WINDOW_FRAMES = 150;

                                                                                                                                                                                                                                  
const PREFETCH_MARGIN_FRAMES = 40;

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
function useVideoTime(videoRef, ready) {
  const [time, setTime] = useState(0);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !ready) return undefined;

    let raf = null;
    let cancelled = false;

    const sample = () => {
      if (cancelled) return;
      setTime(video.currentTime);
      raf = requestAnimationFrame(sample);
    };

    const start = () => {
      if (raf === null) raf = requestAnimationFrame(sample);
    };
    const stop = () => {
      if (raf !== null) cancelAnimationFrame(raf);
      raf = null;
      setTime(video.currentTime);
    };

    video.addEventListener('play', start);
    video.addEventListener('playing', start);
    video.addEventListener('pause', stop);
    video.addEventListener('ended', stop);
    video.addEventListener('seeked', stop);
    video.addEventListener('loadedmetadata', stop);

    if (!video.paused) start();
    else setTime(video.currentTime);

    return () => {
      cancelled = true;
      if (raf !== null) cancelAnimationFrame(raf);
      video.removeEventListener('play', start);
      video.removeEventListener('playing', start);
      video.removeEventListener('pause', stop);
      video.removeEventListener('ended', stop);
      video.removeEventListener('seeked', stop);
      video.removeEventListener('loadedmetadata', stop);
    };
  }, [videoRef, ready]);

  return time;
}

function PlaybackPanel({ videoId, matchId, dataEpoch = 0, seekRequest = null }) {
  const videoRef = useRef(null);
  const [videoReady, setVideoReady] = useState(false);
  const [windowStart, setWindowStart] = useState(0);
  const [manualFrame, setManualFrame] = useState(0);
  const [follow, setFollow] = useState(true);
  const [showOverlay, setShowOverlay] = useState(true);
  const [scrubIndex, setScrubIndex] = useState(0);

  const tracking = useAsync(
    (signal) => api.getTracking(matchId, { startFrame: windowStart }, signal),
    [matchId, windowStart, dataEpoch],
    { enabled: Boolean(matchId) },
  );

                                                                          
                                                                           
                                                         
  const summary = useAsync(
    (signal) => api.getMatchSummary(matchId, signal),
    [matchId, dataEpoch],
    { enabled: Boolean(matchId) },
  );
  const hasProcessed = summary.data?.processed_video_exists === true;
  const processedError = summary.data?.processed_video_error || null;

                                                                       
                                                                            
                                                                           
                                                                          
                                    
  const [sourceOverride, setSourceOverride] = useState(null);
  const source = sourceOverride ?? (hasProcessed ? 'processed' : 'raw');
  const showingProcessed = source === 'processed' && hasProcessed;

  const frames = useMemo(() => tracking.data?.frames || [], [tracking.data]);
  const fps = tracking.data?.fps || 25;
  const calibration = tracking.data?.calibration;
  const maxFrame = tracking.data?.max_frame ?? null;

                                                                           
                                                                             
                                                                              
                                                                              
                                                                               
  const [elementSize, setElementSize] = useState(null);
  const sourceWidth = tracking.data?.frame_width_px
    ?? summary.data?.frame_width_px
    ?? elementSize?.width
    ?? null;
  const sourceHeight = tracking.data?.frame_height_px
    ?? summary.data?.frame_height_px
    ?? elementSize?.height
    ?? null;

  const videoTime = useVideoTime(videoRef, videoReady);

                                                                              
                                                                
  const followedIndex = useMemo(() => {
    if (!frames.length) return 0;
                                                                      
                                                               
    let best = 0;
    let bestDelta = Infinity;
    for (let i = 0; i < frames.length; i += 1) {
      const delta = Math.abs(frames[i].timestamp - videoTime);
      if (delta < bestDelta) {
        bestDelta = delta;
        best = i;
      }
    }
    return best;
  }, [frames, videoTime]);

  const frameIndex = follow
    ? followedIndex
    : Math.min(scrubIndex, Math.max(0, frames.length - 1));
  const currentFrame = frames[frameIndex] || null;

                                                                           
                                                                          
                                                                             
                                                                              
                                
  useEffect(() => {
    if (!follow || !videoReady || !fps) return;
    const playhead = Math.floor(videoTime * fps);
    const lowerBound = windowStart;
    const upperBound = windowStart + WINDOW_FRAMES;

    if (playhead >= upperBound - PREFETCH_MARGIN_FRAMES) {
      const next = windowStart + WINDOW_FRAMES;
      if (maxFrame === null || next <= maxFrame) setWindowStart(next);
    } else if (playhead < lowerBound) {
                                                                            
                                                                          
                                                                        
      setWindowStart(Math.max(0, Math.floor(playhead / WINDOW_FRAMES) * WINDOW_FRAMES));
    }
  }, [videoTime, fps, follow, videoReady, windowStart, maxFrame]);

                                                                       
                                                                        
           
  const seekNonce = seekRequest?.nonce ?? null;
  const seekTime = seekRequest?.timestamp ?? null;
  useEffect(() => {
    if (seekNonce === null || seekTime === null) return;
    const video = videoRef.current;
    if (!video) return;
    setFollow(true);
    video.currentTime = seekTime;
                                                                          
                                                      
    if (fps) {
      const frame = Math.floor(seekTime * fps);
      setWindowStart(Math.max(0, Math.floor(frame / WINDOW_FRAMES) * WINDOW_FRAMES));
    }
                                                           
  }, [seekNonce]);

  const seekToFrame = (index) => {
    setScrubIndex(index);
    const video = videoRef.current;
    const frame = frames[index];
    if (video && frame && Number.isFinite(frame.timestamp)) {
      video.currentTime = frame.timestamp;
    }
  };

  if (!videoId) {
    return (
      <Slab>
        <EmptyState
          title="No Clip Loaded"
          message="Upload a match video above. The raw file streams back from GET /api/videos/{video_id}/file as soon as the upload finishes, whether or not processing has completed."
        />
      </Slab>
    );
  }

  return (
    <div className="dash-playback">
      <div className="dash-playback-video" data-anim>
        <div className="dash-video-stack">
          <video
            ref={videoRef}
                                                                          
                                                         
            key={showingProcessed ? 'processed' : 'raw'}
            className="dash-video"
            src={showingProcessed ? api.processedVideoUrl(videoId) : api.videoFileUrl(videoId)}
            controls
            preload="metadata"
            onLoadedMetadata={(event) => {
              setElementSize({
                width: event.currentTarget.videoWidth,
                height: event.currentTarget.videoHeight,
              });
              setVideoReady(true);
            }}
          />
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
          {showOverlay && currentFrame && sourceWidth && sourceHeight ? (
            <TrackingOverlaySvg
              frame={currentFrame}
              sourceWidth={sourceWidth}
              sourceHeight={sourceHeight}
              showPlayers
              showBall
              showLabels={!showingProcessed}
            />
          ) : null}
        </div>

        <div className="dash-playback-controls">
          <div className="dash-source-toggle" role="group" aria-label="Video source">
            <button
              type="button"
              className={`dash-source-btn ${showingProcessed ? 'is-active' : ''}`.trim()}
              onClick={() => setSourceOverride('processed')}
              disabled={!hasProcessed}
              aria-pressed={showingProcessed}
            >
              Processed
            </button>
            <button
              type="button"
              className={`dash-source-btn ${!showingProcessed ? 'is-active' : ''}`.trim()}
              onClick={() => setSourceOverride('raw')}
              aria-pressed={!showingProcessed}
            >
              Original
            </button>
          </div>

          <p className="dash-hint">
            {showingProcessed
              ? 'The pipeline’s own render — boxes, track ids and team colours burned into the pixels — with the live tactical overlay synchronised on top. The overlay’s rings track the same detections as the burned-in boxes, so they confirm alignment rather than adding a second reading.'
              : hasProcessed
                ? 'The raw upload, with the full tracking overlay drawn live.'
                : 'The raw upload. The annotated render is written by the last pipeline stage — it appears here on its own once a run completes.'}
          </p>

                                                                                                                                              
          {!hasProcessed && processedError ? (
            <div className="dash-inline-error" role="alert">
              <span className="dash-inline-error-key">No annotated render</span>
              {processedError}
            </div>
          ) : null}

          <label className="dash-check">
            <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
            <span>Follow playback</span>
          </label>

          <label className="dash-check">
            <input
              type="checkbox"
              checked={showOverlay}
              onChange={(e) => setShowOverlay(e.target.checked)}
            />
            <span>Tactical overlay</span>
          </label>

          <div className="dash-frame-scrub">
            <input
              type="range"
              className="dash-range"
              min={0}
              max={Math.max(0, frames.length - 1)}
              value={frameIndex}
              disabled={!frames.length}
              onChange={(event) => {
                setFollow(false);
                seekToFrame(Number(event.target.value));
              }}
            />
            <span className="dash-frame-readout">
              {currentFrame ? (
                <>
                  frame <strong>{currentFrame.frame_number}</strong> · {currentFrame.timestamp.toFixed(2)}s
                  {maxFrame !== null ? <> · window {windowStart}–{windowStart + WINDOW_FRAMES} of {maxFrame}</> : null}
                </>
              ) : (
                <span className="dash-absent">no frames in window</span>
              )}
            </span>
          </div>

          <form
            className="dash-window-form"
            onSubmit={(event) => {
              event.preventDefault();
              const target = Math.max(0, Number(manualFrame) || 0);
              setWindowStart(target);
              const video = videoRef.current;
              if (video && fps) video.currentTime = target / fps;
            }}
          >
            <NumberField
              label="Jump to frame"
              value={manualFrame}
              min={0}
              step={50}
              onChange={setManualFrame}
              hint="The window follows playback on its own; this is for jumping somewhere specific."
            />
            <button type="submit" className="btn-ghost dash-window-btn">
              Jump
            </button>
          </form>
        </div>
      </div>

      <div className="dash-playback-side" data-anim>
        <h4 className="dash-card-title">2D Pitch Minimap</h4>
        {!matchId ? (
          <EmptyState
            title="No Match Linked"
            message="Tracking is served per match. Upload a clip to get a match_id."
          />
        ) : (
          <AsyncBlock
            state={tracking}
            loadingLabel="Loading tracking window"
            loadingRows={2}
            isEmpty={(data) => !data?.frames?.length}
            emptyTitle="No Tracking Rows"
            emptyMessage={`No tracking data was persisted for frames ${windowStart}–${windowStart + WINDOW_FRAMES}. Either the pipeline has not reached this window yet, or it produced no detections here.`}
          >
            {(data) => (
              <>
                <TrackingMinimap frame={currentFrame} calibration={data.calibration} />
                <CalibrationSummary calibration={calibration} fps={fps} />
              </>
            )}
          </AsyncBlock>
        )}
      </div>
    </div>
  );
}

                                                                                                                                                                                                                                                                                 
function CalibrationSummary({ calibration, fps }) {
  if (!calibration) {
    return (
      <p className="dash-absent dash-pitch-note">
        Calibration status not reported for this window.
      </p>
    );
  }

  const valid = Boolean(calibration.valid_in_window);

  return (
    <div className={`dash-calib-summary ${valid ? 'is-valid' : 'is-invalid'}`}>
      <Chip tone={valid ? 'good' : 'warn'}>
        {valid ? 'Calibration valid' : 'Calibration unavailable'}
      </Chip>
      <div className="dash-meta-block">
        <MetaRow label="fps">
          <MetricValue value={fps} precision={2} />
        </MetaRow>
        <MetaRow label="rows in window">
          <MetricValue value={calibration.tracking_rows_in_window} />
        </MetaRow>
        <MetaRow label="rows with pitch coords">
          <MetricValue value={calibration.rows_with_pitch_coordinates} />
        </MetaRow>
        <MetaRow label="status rows">
          {calibration.has_status_rows ? 'present' : <span className="dash-absent">absent</span>}
        </MetaRow>
        <MetaRow label="valid frame ranges">
          {calibration.valid_frame_ranges?.length ? (
            <code>
              {calibration.valid_frame_ranges.map(([lo, hi]) => `${lo}–${hi}`).join(', ')}
            </code>
          ) : (
            <span className="dash-absent">none</span>
          )}
        </MetaRow>
      </div>
      {calibration.note ? <p className="dash-hint">{calibration.note}</p> : null}
    </div>
  );
}

                                                                          
                                                                          
                                                                          

function HeatmapPanel({ matchId, goToTab, dataEpoch = 0 }) {
  const [playerId, setPlayerId] = useState(1);
  const [requested, setRequested] = useState(null);

                                                                            
                                                                      
  const roster = useAsync(
    (signal) => api.getPlayerIntelligence(matchId, signal),
    [matchId, dataEpoch],
    { enabled: Boolean(matchId) },
  );

  const knownIds = useMemo(
    () => (Array.isArray(roster.data) ? roster.data.map((entry) => entry.player_id) : []),
    [roster.data],
  );

                                                                          
                                                                            
                                                                             
                                                                      
  const [space, setSpace] = useState('pitch');
                                                                            
                                                                             
                                                                        
  const [autoFellBack, setAutoFellBack] = useState(false);

                                                                         
                                                                        
                                                                             
                                                                          
                                                                            
  useEffect(() => {
    if (requested === null && knownIds.length) {
      setPlayerId(knownIds[0]);
      setRequested(knownIds[0]);
    }
  }, [knownIds, requested]);

                                                                              
                                                                         
  useEffect(() => {
    setAutoFellBack(false);
  }, [dataEpoch, requested]);

  const heatmap = useAsync(
    (signal) => api.getHeatmap(matchId, requested, space, signal),
    [matchId, requested, space, dataEpoch],
    { enabled: Boolean(matchId) && requested !== null },
  );

                                                                              
                                                                       
                                                                              
                                                                              
                                                                              
                                             
    
                                                                             
                                                                          
                                                             
  useEffect(() => {
    const data = heatmap.data;
    if (!data || autoFellBack) return;
    if (data.space === 'pitch' && data.usable_sample_size === 0 && data.sample_size > 0) {
      setAutoFellBack(true);
      setSpace('image');
    }
  }, [heatmap.data, autoFellBack]);

  if (!matchId) {
    return (
      <Slab>
        <EmptyState
          title="Select a Match First"
          message="Heatmaps are served per match and player."
          action={
            <button type="button" className="btn-ghost" onClick={() => goToTab('match')}>
              Upload a clip above
            </button>
          }
        />
      </Slab>
    );
  }

  return (
    <PanelGrid columns={2}>
      <Panel
        number="03"
        title="Player Heatmap"
        subtitle="GET /api/matches/{match_id}/heatmap/{player_id}"
      >
        <form
          className="dash-form dash-form-inline"
          onSubmit={(event) => {
            event.preventDefault();
            setRequested(Number(playerId));
          }}
        >
          <NumberField
            label="Player ID"
            value={playerId}
            min={0}
            step={1}
            onChange={setPlayerId}
            hint="ByteTrack tracking ID, scoped to this processed video."
          />
          <button type="submit" className="btn-primary dash-submit">
            Render Heatmap
          </button>
        </form>

        <div className="dash-space-toggle">
          <span className="dash-label">Coordinate space</span>
          <div className="dash-source-toggle" role="group" aria-label="Heatmap coordinate space">
            <button
              type="button"
              className={`dash-source-btn ${space === 'pitch' ? 'is-active' : ''}`.trim()}
              onClick={() => {
                setAutoFellBack(true);                                        
                setSpace('pitch');
              }}
              aria-pressed={space === 'pitch'}
            >
              Pitch (metres)
            </button>
            <button
              type="button"
              className={`dash-source-btn ${space === 'image' ? 'is-active' : ''}`.trim()}
              onClick={() => {
                setAutoFellBack(true);
                setSpace('image');
              }}
              aria-pressed={space === 'image'}
            >
              Image (pixels)
            </button>
          </div>
          <p className="dash-hint">
            {space === 'pitch'
              ? 'Metres on a 105×68 model, via a validated homography. Comparable between matches — and empty when calibration never validated.'
              : 'Where the player appeared in the video frame. Needs no calibration, but it is not a pitch position: it moves with the camera and cannot be compared across matches.'}
          </p>
        </div>

        {knownIds.length ? (
          <div className="dash-quickpick">
            <span className="dash-label">Detected players</span>
            <div className="dash-quickpick-row">
              {knownIds.slice(0, 24).map((id) => (
                <button
                  key={id}
                  type="button"
                  className={`dash-pill ${requested === id ? 'is-active' : ''}`.trim()}
                  onClick={() => {
                    setPlayerId(id);
                    setRequested(id);
                  }}
                >
                  {id}
                </button>
              ))}
            </div>
          </div>
        ) : roster.status === 'success' ? (
          <p className="dash-hint dash-absent">
            No player-intelligence rows for this match yet, so there is no roster to pick from — enter a tracking ID directly.
          </p>
        ) : null}
      </Panel>

      <Panel number="04" title="Density Grid" subtitle="Density is the backend's own count / max_count — never rescaled here.">
        <AsyncBlock
          state={heatmap}
          idle={
            <EmptyState
              title="No Player Selected"
              message="Pick a tracking ID and render to see where that player actually spent the match."
            />
          }
          loadingLabel="Building heatmap"
          isEmpty={(data) => !data?.cells?.length}
          emptyTitle="No Usable Samples"
          emptyMessage={
            space === 'pitch'
              ? 'This player has tracking rows, but none of them carry valid pitch coordinates — so there is no position to plot. Switch to image space to plot the pixel positions that do exist.'
              : 'This player has no tracking rows carrying a pixel position, so there is nothing to bin.'
          }
        >
          {(data) => (
            <>
              <HeatmapPitch heatmap={data} />
              <div className="dash-meta-block">
                                                                                                                                                                                                            
                <MetaRow label="space">
                  {data.space === 'image' ? (
                    <span>
                      image / pixels
                      {data.frame_width_px
                        ? ` · ${data.frame_width_px}×${data.frame_height_px} frame`
                        : ''}
                    </span>
                  ) : (
                    <span>pitch / metres · {data.pitch_length_m}×{data.pitch_width_m}m</span>
                  )}
                </MetaRow>
                <MetaRow label="confidence">
                  <MetricValue value={data.confidence} absentLabel="Not Reported" />
                </MetaRow>
                <MetaRow label="sample size">
                  <MetricValue value={data.sample_size} />
                </MetaRow>
                <MetaRow label="usable samples">
                  <MetricValue value={data.usable_sample_size} />
                </MetaRow>
              </div>

              {data.space === 'image' ? (
                <p className="dash-inline-error">
                  <span className="dash-inline-error-key">Image space</span>
                  {autoFellBack
                    ? 'Shown automatically because this match has no valid calibration, so the pitch grid has nothing in it. '
                    : ''}
                  These cells are frame pixels, not pitch positions. They shift with the camera,
                  carry no distance in metres, and cannot be compared against another match.
                </p>
              ) : null}

              {data.space === 'pitch' && data.usable_sample_size === 0 && data.sample_size > 0 ? (
                <p className="dash-inline-error">
                  <span className="dash-inline-error-key">Note</span>
                  {data.sample_size} tracking rows exist for this player, but none were captured under valid calibration.
                </p>
              ) : null}
            </>
          )}
        </AsyncBlock>
      </Panel>
    </PanelGrid>
  );
}

                                                                          
                                                                          
                                                                          

const EVENT_TONE = {
  shot: 'bad',
  pass: 'good',
  turnover: 'warn',
  first_touch: 'neutral',
};

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
function EventsPanel({ matchId, dataEpoch = 0, onSeek }) {
  const [filter, setFilter] = useState(null);

  const events = useAsync(
    (signal) => api.getEvents(matchId, { eventType: filter }, signal),
    [matchId, filter, dataEpoch],
    { enabled: Boolean(matchId) },
  );

  if (!matchId) {
    return (
      <Slab>
        <EmptyState
          title="Select a Match First"
          message="Events are detected per match by the pipeline's pass, shot, turnover and first-touch heuristics."
        />
      </Slab>
    );
  }

  const byType = events.data?.by_type || {};
  const types = Object.keys(byType).sort();

  return (
    <Panel
      number="05"
      title="Detected Events"
      subtitle="GET /api/matches/{match_id}/events — pass, shot, turnover and first-touch heuristics."
    >
      {types.length ? (
        <div className="dash-quickpick">
          <span className="dash-label">Filter</span>
          <div className="dash-quickpick-row">
            <button
              type="button"
              className={`dash-pill ${filter === null ? 'is-active' : ''}`.trim()}
              onClick={() => setFilter(null)}
            >
              all ({events.data?.total ?? 0})
            </button>
            {types.map((type) => (
              <button
                key={type}
                type="button"
                className={`dash-pill ${filter === type ? 'is-active' : ''}`.trim()}
                onClick={() => setFilter(type)}
              >
                {type} ({byType[type]})
              </button>
            ))}
          </div>
        </div>
      ) : null}

      <AsyncBlock
        state={events}
        loadingLabel="Reading detected events"
        isEmpty={(data) => !data?.events?.length}
        emptyTitle="No Events Detected"
        emptyMessage={
          'The pipeline found no possession sequence to derive events from. That happens when the ball '
          + 'model detected the ball on too few frames, or when no tracked player was ever within the '
          + 'control radius of it. Events are never invented to fill this panel.'
        }
      >
        {(data) => (
          <>
            <MethodStrip
              items={[
                {
                  label: 'space',
                  value: data.space === 'pitch' ? 'pitch / metres' : 'image / pixels',
                  tone: data.space === 'pitch' ? 'good' : 'warn',
                },
                { label: 'events', value: `${data.returned} of ${data.total}` },
              ]}
            />

            {data.space === 'image' ? (
              <p className="dash-inline-error">
                <span className="dash-inline-error-key">Estimated</span>
                This match has no valid calibration, so possession was resolved in image space using
                each player&apos;s own bounding-box height as the local scale. The events are real
                detections; any distance attached to one is an estimate, and none of them carry a
                pitch position.
              </p>
            ) : null}

            <Slab className="dash-stage-table">
              {data.events.map((event) => (
                <button
                  key={event.event_id}
                  type="button"
                  className="dash-event-row"
                  onClick={() => onSeek?.(event.timestamp)}
                  title="Jump the player to this moment"
                >
                  <span className="dash-event-time">{event.timestamp.toFixed(2)}s</span>
                  <Chip tone={EVENT_TONE[event.event_type] || 'neutral'}>{event.event_type}</Chip>
                  <span className="dash-event-players">
                    {event.player_id !== null && event.player_id !== undefined ? (
                      <>#{event.player_id}</>
                    ) : (
                      <span className="dash-absent">unattributed</span>
                    )}
                    {event.related_player_id !== null && event.related_player_id !== undefined ? (
                      <> → #{event.related_player_id}</>
                    ) : null}
                  </span>
                  <span className="dash-event-meta">
                    {event.metadata?.pass_distance_m
                      ? `${event.metadata.pass_distance_m}m${
                          event.metadata.pass_distance_m_is_estimate ? ' (est)' : ''
                        }`
                      : event.team_id || ''}
                  </span>
                </button>
              ))}
            </Slab>
          </>
        )}
      </AsyncBlock>
    </Panel>
  );
}

                                                                          
                                                                          
                                                                          

function LatencyPanel({ jobId, dataEpoch = 0 }) {
  const latency = useAsync(
    (signal) => api.getPipelineLatency(jobId, signal),
    [jobId, dataEpoch],
    { enabled: Boolean(jobId) },
  );

  if (!jobId) {
    return (
      <Slab>
        <EmptyState
          title="No Job Yet"
          message="Stage timings are measured during a real run and keyed by job_id."
        />
      </Slab>
    );
  }

                                                                              
                                                                    
  if (latency.status === 'error' && latency.error?.notFound) {
    return (
      <Slab>
        <EmptyState
          title="Latency Report Not Ready"
          message="No stage of this job has completed a measured run yet. The report appears here as soon as one does."
          action={
            <button type="button" className="btn-ghost" onClick={latency.reload}>
              Check Again
            </button>
          }
        />
      </Slab>
    );
  }

  return (
    <AsyncBlock
      state={latency}
      loadingLabel="Reading measured stage timings"
      isEmpty={(data) => !data?.stages?.length}
      emptyTitle="No Stages Recorded"
      emptyMessage="The report exists but contains no stage rows."
    >
      {(data) => {
        const max = Math.max(...data.stages.map((stage) => stage.seconds || 0), 0.0001);
        return (
          <>
            <StatRow>
              <StatNum value={data.total_seconds} label="Seconds Total" />
              <StatNum value={data.stages.length} label="Measured Stages" />
            </StatRow>

            <MethodStrip
              items={[
                { label: 'source', value: data.source, tone: 'good' },
                { label: 'job_id', value: data.job_id },
                { label: 'generated', value: formatTime(data.generated_at) },
              ]}
            />

            <Slab className="dash-stage-table">
              {data.stages.map((stage) => (
                <div key={stage.stage} className="dash-stage-row">
                  <span className="dash-stage-name">{stage.stage}</span>
                  <span className="dash-stage-bar">
                    <span
                      className="dash-stage-bar-fill"
                      style={{ width: `${((stage.seconds || 0) / max) * 100}%` }}
                    />
                  </span>
                  <span className="dash-stage-secs">{(stage.seconds ?? 0).toFixed(3)}s</span>
                  {stage.detail ? <span className="dash-stage-detail">{stage.detail}</span> : null}
                </div>
              ))}
            </Slab>
          </>
        );
      }}
    </AsyncBlock>
  );
}

                                                                          

function formatTime(value) {
  if (!value) return <span className="dash-absent">—</span>;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return <span className="dash-absent">—</span>;
  return date.toLocaleString();
}
