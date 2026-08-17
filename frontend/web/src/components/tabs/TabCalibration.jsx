import { useEffect, useState } from 'react';

import api from '../../api/client.js';
import { useAction } from '../../hooks/useAsync.js';
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
import { EmptyState, ErrorState, LoadingState } from '../ui/States.jsx';
import { SelectMatchState } from '../ui/MatchPicker.jsx';
import { NumberField, SubmitButton } from '../ui/Form.jsx';

                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
export default function TabCalibration({ matchId, goToTab, onAttachMatch }) {
  const [frameNumber, setFrameNumber] = useState(0);
  const debug = useAction((frame, signal) => api.getCalibrationDebug(matchId, frame, signal));

                                                                            
  useEffect(() => {
    debug.reset();
                                                           
  }, [matchId]);

  return (
    <TabSection animKey={`calibration-${matchId}`}>
      <SectionHeader
        tag="Calibration Debug"
        title={<>Homography<br />Diagnostics</>}
        meta="Render any frame with the detected field, keypoints, and projected players drawn on it — then read the pipeline's own verdict on whether that projection can be trusted."
      />

      {!matchId ? (
        <SelectMatchState onGoToUpload={() => goToTab('match')} onAttachMatch={onAttachMatch} />
      ) : (
        <>
          <Slab>
            <form
              className="dash-form dash-form-inline"
              onSubmit={(event) => {
                event.preventDefault();
                debug.run(Math.max(0, Number(frameNumber) || 0));
              }}
            >
              <NumberField
                label="Frame Number"
                value={frameNumber}
                min={0}
                step={1}
                onChange={setFrameNumber}
                hint="Decoded live from the stored video — any frame index in the clip."
              />
              <SubmitButton busy={debug.status === 'loading'} busyLabel="Rendering">
                Render Debug Frame
              </SubmitButton>
            </form>
          </Slab>

          {debug.status === 'idle' ? (
            <EmptyState
              title="No Frame Rendered"
              message="Choose a frame number and render it. Nothing is fetched until you do — the endpoint decodes the video frame on demand."
            />
          ) : null}

          {debug.status === 'loading' ? (
            <LoadingState
              label="Decoding frame and running detectors"
              detail="Field detection, keypoint detection, and homography estimation all run server-side for this single frame."
              rows={3}
            />
          ) : null}

          {debug.status === 'error' ? (
            <ErrorState error={debug.error} onRetry={() => debug.run(Number(frameNumber) || 0)} />
          ) : null}

          {debug.status === 'success' && debug.data ? <DebugResult data={debug.data} /> : null}
        </>
      )}
    </TabSection>
  );
}

function DebugResult({ data }) {
  const valid = Boolean(data.calibration_valid);

  return (
    <>
      <SubHeading>
        Frame {data.frame_number}
        <span className="dash-subhead-chip">
          <Chip tone={valid ? 'good' : 'warn'}>
            {valid ? 'Calibration valid' : 'Calibration unavailable'}
          </Chip>
        </span>
      </SubHeading>

      {!valid && data.invalid_reason ? (
        <div className="dash-inline-error" role="status">
          <span className="dash-inline-error-key">invalid_reason</span>
          {data.invalid_reason}
        </div>
      ) : null}

      <PanelGrid columns={2}>
        <Panel number="01" title="Overlay" subtitle="Server-rendered PNG, returned base64-encoded.">
          {data.overlay_png_base64 ? (
            <div className="dash-overlay-frame">
              <img
                className="dash-overlay-img"
                src={`data:image/png;base64,${data.overlay_png_base64}`}
                alt={`Calibration debug overlay for frame ${data.frame_number}`}
              />
            </div>
          ) : (
            <p className="dash-absent">No overlay image returned for this frame.</p>
          )}
        </Panel>

        <Panel number="02" title="Detection Confidence" subtitle="Two independent detectors, each reporting its own method.">
          <StatRow>
            <StatNum
              value={data.calibration_confidence}
              label="Calibration Confidence"
              absentLabel="Not Reported"
            />
            <StatNum
              value={data.keypoint_detection_confidence}
              label="Keypoint Confidence"
              absentLabel="Not Reported"
            />
            <StatNum
              value={data.field_detection_confidence}
              label="Field Confidence"
              absentLabel="Not Reported"
            />
          </StatRow>

          <MethodStrip
            items={[
              { label: 'calibration method', value: data.calibration_metric_method },
              { label: 'calibration confidence class', value: data.calibration_metric_confidence,
                tone: data.calibration_metric_confidence === 'normal' ? 'good' : 'warn' },
              { label: 'keypoint method', value: data.keypoint_detection_method },
            ]}
          />

          <div className="dash-meta-block">
            <MetaRow label="keypoints found">
              <MetricValue value={data.n_keypoints} absentLabel="Not Reported" />
            </MetaRow>
            <MetaRow label="field detected">
              <Chip tone={data.field_detected ? 'good' : 'warn'}>
                {data.field_detected ? 'yes' : 'no'}
              </Chip>
            </MetaRow>
            <MetaRow label="match_id">
              <code className="dash-code-sm">{data.match_id}</code>
            </MetaRow>
          </div>
        </Panel>
      </PanelGrid>

      <SubHeading>Projected Players</SubHeading>
      <ProjectedPlayers
        players={data.projected_players}
        suppressed={data.projection_suppressed}
      />
    </>
  );
}

                                                                                                                                                                                                                                                    
function ProjectedPlayers({ players, suppressed }) {
  if (suppressed) {
    return (
      <Slab>
        <EmptyState
          title="Projection Suppressed"
          message="The pipeline withheld pitch projections for this frame because the homography did not meet its validity threshold. Positions are not shown rather than shown wrong — a projected point drawn from an invalid homography looks exactly like a correct one."
        />
      </Slab>
    );
  }

  if (!players || !players.length) {
    return (
      <Slab>
        <EmptyState
          title="No Players Projected"
          message="Projection was not suppressed, but no players were detected in this frame."
        />
      </Slab>
    );
  }

  const columns = [...new Set(players.flatMap((player) => Object.keys(player)))];

  return (
    <Slab className="dash-table-wrap">
      <table className="dash-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column.replace(/_/g, ' ')}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {players.map((player, index) => (
            <tr key={`${player.player_id ?? 'row'}-${index}`}>
              {columns.map((column) => (
                <td key={column}>
                  <MetricValue value={player[column]} absentLabel="—" />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </Slab>
  );
}
