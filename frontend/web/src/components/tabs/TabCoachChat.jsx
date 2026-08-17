import { useCallback, useEffect, useRef, useState } from 'react';

import { ApiError } from '../../api/client.js';
import nexus, { NEXUS_BASE, ROUTING_POLICIES, streamChat } from '../../api/nexus.js';
import { useAction } from '../../hooks/useAsync.js';
import {
  Chip,
  Disclaimer,
  FactorList,
  MetaRow,
  MethodStrip,
  Panel,
  PanelGrid,
  RiskBadge,
  SectionHeader,
  Slab,
  StatNum,
  StatRow,
  SubHeading,
  TabSection,
} from '../ui/Primitives.jsx';
import { EmptyState, ErrorState, LoadingState } from '../ui/States.jsx';
import { NumberField, SelectField, SubmitButton, TextField, ToggleField } from '../ui/Form.jsx';

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
const NO_SESSION = null;

export default function TabCoachChat({ matchId }) {
  return (
    <TabSection animKey="coach">
      <SectionHeader
        tag="Coach Chat"
        title={<>Tactical<br />Assistant</>}
        meta={`Conversational analysis routed by NEXUS at ${NEXUS_BASE} — a separate service from the core backend on :8000.`}
      />

      <ChatPanel />

      <SubHeading>Narrated Mental-Readiness Report</SubHeading>
      <PsychologyReportPanel matchId={matchId} />
    </TabSection>
  );
}

                                                                          
                                                                          
                                                                          

function ChatPanel() {
  const [sessionId, setSessionId] = useState(NO_SESSION);
  const [policy, setPolicy] = useState('BALANCED');
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(1024);
  const [modelId, setModelId] = useState('');
  const [useStreaming, setUseStreaming] = useState(true);

  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [routing, setRouting] = useState(null);

  const controllerRef = useRef(null);
  const scrollRef = useRef(null);

  useEffect(
    () => () => {
      controllerRef.current?.abort();
    },
    [],
  );

  useEffect(() => {
    const node = scrollRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [messages]);

  const resetSession = useCallback(() => {
    controllerRef.current?.abort();
                                                                           
                                                                  
    setSessionId(NO_SESSION);
    setMessages([]);
    setError(null);
    setRouting(null);
    setBusy(false);
  }, []);

  const send = async (event) => {
    event.preventDefault();
    const text = draft.trim();
    if (!text || busy) return;

    const history = [...messages, { role: 'user', content: text }];
    setMessages(history);
    setDraft('');
    setError(null);
    setBusy(true);

    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    const body = {
      messages: history.map(({ role, content }) => ({ role, content })),
      session_id: sessionId,
      policy,
      temperature: Number(temperature),
      max_tokens: Number(maxTokens) || null,
      model_id: modelId.trim() || null,
    };

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    const isDeadSession = (caught) =>
      caught?.status === 404 && Boolean(body.session_id);

    try {
      if (useStreaming) {
                                                                            
                                                                             
        setMessages((current) => [...current, { role: 'assistant', content: '', streaming: true }]);

        await streamChat(body, {
          signal: controller.signal,
          onDelta: (delta) => {
            setMessages((current) => {
              const next = [...current];
              const last = next[next.length - 1];
              if (last?.role === 'assistant') {
                next[next.length - 1] = { ...last, content: last.content + delta };
              }
              return next;
            });
          },
          onDone: (event_) => {
            setMessages((current) => {
              const next = [...current];
              const last = next[next.length - 1];
              if (last?.role === 'assistant') next[next.length - 1] = { ...last, streaming: false };
              return next;
            });
            setRouting({
                                                                     
                                                                         
                                                     
              model: null,
              provider: event_.provider_name,
              reason: event_.routing_reason,
              cost: event_.cost_usd,
              taskType: event_.task_type,
              privacy: event_.privacy_level,
              streamed: true,
            });
            if (event_.session_id) setSessionId(event_.session_id);
          },
          onError: (message) => {
            setError(new ApiError(message, { status: 502, service: 'nexus' }));
            setMessages((current) => {
              const next = [...current];
              const last = next[next.length - 1];
              if (last?.role === 'assistant' && !last.content) return next.slice(0, -1);
              if (last?.role === 'assistant') next[next.length - 1] = { ...last, streaming: false };
              return next;
            });
          },
        });
      } else {
        const response = await nexus.chat(body, controller.signal);
        setMessages((current) => [
          ...current,
          { role: 'assistant', content: response.content, streaming: false },
        ]);
        setRouting({
          model: response.model_used,
          provider: response.provider_name,
          reason: response.routing_reason,
          cost: response.cost_usd,
          taskType: response.task_type,
          privacy: response.privacy_level,
          streamed: false,
          usage: response.usage,
        });
        if (response.session_id) setSessionId(response.session_id);
      }
    } catch (caught) {
      if (caught?.name === 'AbortError') {
        setBusy(false);
        return;
      }

                                                                              
                                                       
      setMessages((current) => {
        const last = current[current.length - 1];
        return last?.role === 'assistant' && !last.content ? current.slice(0, -1) : current;
      });

      if (isDeadSession(caught)) {
        setSessionId(NO_SESSION);
        try {
          const retryBody = { ...body, session_id: null };
          const response = await nexus.chat(
            { ...retryBody, stream: false },
            controller.signal,
          );
          setMessages((current) => [
            ...current,
            { role: 'assistant', content: response.content, streaming: false },
          ]);
          setRouting({
            model: response.model_used,
            provider: response.provider_name,
            reason: response.routing_reason,
            cost: response.cost_usd,
            taskType: response.task_type,
            privacy: response.privacy_level,
            streamed: false,
            usage: response.usage,
          });
          if (response.session_id) setSessionId(response.session_id);
          setBusy(false);
          return;
        } catch (retryFailed) {
          if (retryFailed?.name !== 'AbortError') setError(retryFailed);
          setBusy(false);
          return;
        }
      }

      setError(caught);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <Slab className="dash-chat-bar">
        <div className="dash-field-grid dash-field-grid-4">
          <SelectField
            label="Policy selected routing"
            value={policy}
            onChange={(value) => setPolicy(value || 'BALANCED')}
            options={ROUTING_POLICIES}
          />
          <NumberField
            label="Temperature"
            value={temperature}
            min={0}
            max={2}
            step={0.1}
            onChange={setTemperature}
          />
          <NumberField
            label="Max Tokens"
            value={maxTokens}
            min={1}
            step={128}
            nullable
            onChange={setMaxTokens}
          />
          <TextField
            label="Model ID Override"
            value={modelId}
            onChange={(value) => setModelId(value ?? '')}
            placeholder="leave blank to let the policy route"
          />
        </div>

        <div className="dash-chat-bar-actions">
          <ToggleField label="Stream response" value={useStreaming} onChange={setUseStreaming} />
          <button type="button" className="btn-ghost" onClick={resetSession}>
            New Session
          </button>
          <span className="dash-session-chip">
            <span className="dash-meta-key">session</span>
            {sessionId ? (
              <code className="dash-code-sm">{sessionId.slice(0, 8)}…</code>
            ) : (
                                                                              
                                      
              <span className="dash-absent">new on first message</span>
            )}
          </span>
        </div>

        <NexusModelIndicator routing={routing} busy={busy} />
      </Slab>

      <Slab className="dash-chat">
        <div className="dash-chat-log" ref={scrollRef} role="log" aria-live="polite">
          {messages.length === 0 && !error ? (
            <EmptyState
              title="No Messages Yet"
              message="Ask about a formation, a player's press resistance, or what the tracking actually shows. The routing policy above decides which model answers."
            />
          ) : null}

          {messages.map((message, index) => (
            <div key={index} className={`dash-msg is-${message.role}`}>
              <span className="dash-msg-role">{message.role}</span>
              <div className="dash-msg-body">
                {message.content}
                {message.streaming ? <span className="dash-caret" aria-hidden="true" /> : null}
              </div>
            </div>
          ))}

          {busy && !messages.some((message) => message.streaming) ? (
            <LoadingState label="Waiting for NEXUS" rows={1} />
          ) : null}
        </div>

        {error ? (
          <ErrorState
            error={error}
            title={error?.unreachable ? 'Coach Chat Unreachable' : 'NEXUS Request Failed'}
          />
        ) : null}

        <form className="dash-chat-form" onSubmit={send}>
          <textarea
            className="dash-input dash-chat-input"
            rows={3}
            value={draft}
            placeholder="Ask the tactical assistant…"
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) send(event);
            }}
          />
          <div className="dash-chat-form-actions">
            <SubmitButton busy={busy} busyLabel="Thinking" disabled={!draft.trim()}>
              Send
            </SubmitButton>
            {busy ? (
              <button
                type="button"
                className="btn-ghost"
                onClick={() => {
                  controllerRef.current?.abort();
                  setBusy(false);
                }}
              >
                Stop
              </button>
            ) : null}
            <span className="dash-hint">⌘/Ctrl + Enter to send</span>
          </div>
        </form>
      </Slab>
    </>
  );
}

                                                                    
function NexusModelIndicator({ routing, busy }) {
  if (busy && !routing) {
    return (
      <div className="dash-model-indicator is-pending">
        <span className="dash-meta-key">NEXUS Model</span>
        <span className="dash-absent">routing…</span>
      </div>
    );
  }

  if (!routing) {
    return (
      <div className="dash-model-indicator">
        <span className="dash-meta-key">NEXUS Model</span>
        <span className="dash-absent">No response yet</span>
      </div>
    );
  }

  return (
    <div className="dash-model-indicator">
      <span className="dash-meta-key">NEXUS Model</span>
      {routing.model ? (
        <Chip tone="accent">{routing.model}</Chip>
      ) : (
        <span className="dash-absent" title="The streaming done-frame reports the provider but not the model id.">
          Not reported in streaming mode
        </span>
      )}
      <MethodStrip
        items={[
          { label: 'provider', value: routing.provider },
          { label: 'routing reason', value: routing.reason },
          { label: 'task type', value: routing.taskType },
          { label: 'privacy', value: routing.privacy },
          {
            label: 'cost usd',
            value: typeof routing.cost === 'number' ? routing.cost.toFixed(6) : null,
          },
          {
            label: 'tokens',
            value: routing.usage ? `${routing.usage.total_tokens}` : null,
          },
        ]}
      />
    </div>
  );
}

                                                                          
                                                                          
                                                                          

function PsychologyReportPanel({ matchId }) {
  const [playerId, setPlayerId] = useState('');
  const [scopeToMatch, setScopeToMatch] = useState(Boolean(matchId));

  const report = useAction((id, scoped, signal) => nexus.getPsychologyReport(id, scoped, signal));

  return (
    <>
      <Slab>
        <form
          className="dash-form dash-form-inline"
          onSubmit={(event) => {
            event.preventDefault();
            const id = playerId.trim();
            if (id) report.run(id, scopeToMatch && matchId ? matchId : undefined);
          }}
        >
          <TextField
            label="Player ID"
            value={playerId}
            onChange={(value) => setPlayerId(value ?? '')}
            placeholder="same ID used on the Psychology tab"
          />
          {matchId ? (
            <ToggleField
              label={scopeToMatch ? `Scoped to ${matchId}` : 'Latest, any match'}
              value={scopeToMatch}
              onChange={setScopeToMatch}
            />
          ) : null}
          <SubmitButton busy={report.status === 'loading'} busyLabel="Generating" disabled={!playerId.trim()}>
            Generate Report
          </SubmitButton>
        </form>
        <p className="dash-hint">
          NEXUS writes the prose only. Every number below is copied verbatim from the football
          backend&apos;s psychology scoring — this panel keeps the two visually separate so a reader
          can always tell computed from generated.
        </p>
      </Slab>

      {report.status === 'idle' ? (
        <EmptyState
          title="No Report Generated"
          message="Enter a player ID who has submitted a psychology questionnaire, then generate. A 404 here means that player has no assessment yet — which is different from NEXUS being down."
        />
      ) : null}

      {report.status === 'loading' ? (
        <LoadingState
          label="Asking NEXUS to narrate the assessment"
          detail="The scoring already exists; only the narrative is being generated."
          rows={3}
        />
      ) : null}

      {report.status === 'error' ? (
        <ErrorState
          error={report.error}
          title={
            report.error?.unreachable
              ? 'Coach Chat Unreachable'
              : report.error?.notFound
                ? 'No Assessment To Narrate'
                : 'Report Failed'
          }
        />
      ) : null}

      {report.status === 'success' && report.data ? <ReportView data={report.data} /> : null}
    </>
  );
}

function ReportView({ data }) {
  return (
    <PanelGrid columns={2}>
      <Panel
        number="01"
        title="Computed — Football Backend"
        subtitle="Deterministic scoring. NEXUS passes these through unchanged."
        tone="good"
      >
        <StatRow>
          <StatNum value={data.mental_readiness} label="Mental Readiness" />
          <StatNum value={data.focus} label="Focus" />
          <StatNum value={data.confidence} label="Confidence" />
          <StatNum value={data.stress} label="Stress · higher is worse" tone="warn" />
        </StatRow>

        <div className="dash-badge-row">
          <RiskBadge label="pressure" risk={data.pressure_risk} />
          <RiskBadge label="mental performance" risk={data.mental_performance_risk} />
        </div>

        <MethodStrip
          items={[
            { label: 'method', value: data.method },
            { label: 'confidence level', value: data.confidence_level },
            { label: 'data source', value: data.data_source },
            { label: 'schema', value: data.schema_version },
          ]}
        />

        <div className="dash-factor-cols">
          <FactorList title="Positive" items={data.key_positive_factors} tone="good" />
          <FactorList title="Negative" items={data.key_negative_factors} tone="bad" />
        </div>

        {data.neutral_factors?.length ? (
          <FactorList title="Neutral" items={data.neutral_factors} tone="neutral" />
        ) : null}

        {data.historical_context?.length ? (
          <FactorList title="CV Corroboration" items={data.historical_context} tone="neutral" />
        ) : null}

        <Disclaimer>{data.disclaimer}</Disclaimer>
      </Panel>

      <Panel
        number="02"
        title="Generated — NEXUS Narrative"
        subtitle="The only LLM-written field in this response."
        tone="accent"
      >
        <div className="dash-badge-row">
          <Chip tone="accent">model: {data.model_used || 'not reported'}</Chip>
          <Chip tone="warn">generated text</Chip>
        </div>

        {data.narrative ? (
          <p className="dash-narrative">{data.narrative}</p>
        ) : (
          <p className="dash-absent">No narrative returned.</p>
        )}

        <div className="dash-meta-block">
          <MetaRow label="player_id">
            <code className="dash-code-sm">{data.player_id}</code>
          </MetaRow>
          <MetaRow label="match_id">
            {data.match_id ? <code className="dash-code-sm">{data.match_id}</code> : <span className="dash-absent">none</span>}
          </MetaRow>
        </div>

        <p className="dash-hint">
          This prose restates the numbers in the panel to its left. If the two ever disagree, the
          computed panel is the record.
        </p>
      </Panel>
    </PanelGrid>
  );
}
