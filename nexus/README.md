# NEXUS — Phase 0 through Phase 14

FastAPI + local LLM (via Ollama) + OpenAI/Anthropic cloud providers +
capability/cost/observed-latency-based routing with automatic failover +
deterministic task and privacy classification + document ingestion/RAG +
long-term memory + tool calling (web search, Python execution, file
reads, read-only database queries, optional GitHub access) + a
goal-driven agent framework (research/coding/data/planning/health/sports,
sharing the same tool-calling loop chat.py uses) + a personal
intelligence engine (confidence- and recency-weighted state, baselines,
trends, prioritized weaknesses, and privacy-gated injection into chat) +
domain intelligence (deterministic health-pattern detection behind a
mandatory safety layer, personalized plan generation, and a football
tactical coach assistant consuming this repo's existing computer-vision
backend over HTTP) + a verification layer (deterministic fact-checking
plus optional multi-model judging, with an honest confidence score that
is never fabricated) + a reproducible evaluation harness with a CI-usable
regression gate + an optional custom-model training pipeline (interaction
logging, privacy-filtered dataset mining, Blackwell-aware preflight,
QLoRA fine-tuning, GGUF export served by the existing Ollama runtime) +
advanced intelligence (a learned capability matrix that closes the
routing improvement loop, graph RAG, multi-agent orchestration with
runtime-enforced delegation caps, an autonomous research loop, model
self-evaluation, and digital-twin forecasting) — all SQLite-backed, all
working local-only with zero cloud keys configured.

**Group E's non-negotiable rule**: the custom model is an OPTION, never a
dependency, and every bound is enforced in the runtime rather than
requested in a prompt. NEXUS boots, routes, and answers identically with
no fine-tuned model present; all training infrastructure is testable
without a GPU and without torch installed; and delegation depth,
sub-agent count, research rounds, and graph hops are all hard caps in
code (`DelegationGuard`, `max_rounds`, `max_hops`/`max_nodes`), not
instructions a model can decline.

**Group C's non-negotiable rule**: in `nexus/health/`, `nexus/generation/`,
and `nexus/sports/`, every score, pattern, deviation, and finding is
computed in Python first — the LLM only ever narrates already-computed
facts, never invents one. `nexus/health/safety.py` enforces this for
health specifically: it filters both input and output unconditionally,
with no config flag to turn it off.

**Group D's non-negotiable rule**: a confidence score must come from
checks that actually ran. `nexus/verification/` never lets an unrunnable
check inflate a score — an answer nothing could verify is reported
`UNVERIFIED`, not `HIGH`. `nexus/evaluation/` never calls a real external
service unless `--real-providers`/`use_real_providers=True` is passed
explicitly, and its `safety` suite has zero regression tolerance.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally, OR Docker +
  Docker Compose (recommended — brings up Ollama for you).

## Run with Docker Compose (recommended)

From the repo root:

```bash
docker compose -f nexus/docker-compose.yml up --build
```

This starts an `ollama` container and the `nexus` API container, wired
together via `OLLAMA_BASE_URL=http://ollama:11434`. Ollama's model data
persists in the `ollama_models` named volume.

Pull the models NEXUS expects (in a separate terminal, once the stack is
up):

```bash
docker exec -it nexus-ollama ollama pull mistral:7b
docker exec -it nexus-ollama ollama pull nomic-embed-text
```

The API is then available at `http://localhost:8100`.

## Run locally without Docker

1. Install and start Ollama, then pull the models:

   ```bash
   ollama pull mistral:7b
   ollama pull nomic-embed-text
   ollama serve
   ```

2. From the repo root, create a virtualenv and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # macOS/Linux
   pip install -r nexus/requirements.txt
   ```

3. Run the API:

   ```bash
   uvicorn nexus.api.main:app --reload --port 8100
   ```

   By default it talks to Ollama at `http://localhost:11434` (see
   `nexus/config/nexus.yaml`). Override with the `OLLAMA_BASE_URL` env var,
   or any other setting with a `NEXUS_<SECTION>__<FIELD>` env var, e.g.
   `NEXUS_SERVER__PORT=9000`.

## Cloud providers (optional)

NEXUS works entirely local-only with zero configuration — this is still
true in Phase 2. To also make OpenAI and/or Anthropic models available for
routing, set their API keys as plain (non-`NEXUS_`-prefixed) env vars
before starting the server:

```bash
export OPENAI_API_KEY=sk-...       # macOS/Linux
export ANTHROPIC_API_KEY=sk-ant-...
# $env:OPENAI_API_KEY = "sk-..."   # Windows PowerShell
```

- If a key is unset, that provider is never constructed — NEXUS boots and
  behaves exactly as it did in Phase 1, and the router only ever considers
  local models.
- If a key is set but the provider errors or is unreachable at request
  time, `route_with_failover()` automatically falls through to the next
  candidate, and ultimately to local — requests never crash because a
  cloud provider is down.
- Cloud model entries (`gpt-4o`, `gpt-4o-mini`, `claude-sonnet-4-5`), their
  context windows, capability scores, and per-1k-token costs live in
  `nexus/config/models.yaml` alongside the local models.

## Intelligent routing (task + privacy classification)

Every `/api/chat` request is classified before routing — deterministically,
with no LLM call involved:

- **Task classification** (`nexus/intelligence/task_classifier.py`) scans
  only this turn's new message text (not the whole conversation) for
  keyword/regex signals and picks the best-matching `TaskType`, which then
  drives the router's capability scoring (e.g. a coding question is scored
  against each model's `coding` capability, not a generic default). An
  oversized prompt (query + prior history) is classified `LONG_CONTEXT`
  regardless of topic — that's a structural signal, not a topical one.
- **Privacy classification** (`nexus/intelligence/privacy_classifier.py`)
  checks the same text for hard PII/secrets (emails, phone numbers,
  SSN-like or credit-card-like numbers, API-key-like tokens) and softer
  personal-state phrasing (health/personal topics without hard PII). A
  `"private"` result **forces the routing policy to `LOCAL_ONLY`** —
  content matching those signals never leaves the machine — *unless* the
  request explicitly set `policy` or `model_id`, in which case that
  explicit choice is trusted and used as-is. The override is safety-
  relevant, not advisory: it's applied in `chat.py` before routing happens,
  not left to the model to decide.
- Both classifications are surfaced in the response (`task_type`,
  `privacy_level`, `classification_reason`) so the override is never
  silent.

**Tuning keyword lists** — add to (not replace) the built-in defaults via
`nexus/config/nexus.yaml`:

```yaml
routing:
  classification:
    extra_signals:
      coding: ["\\bkubernetes\\b", "\\bdocker\\b"]
  privacy:
    extra_private_patterns:
      - "\\bmy employee id\\b"
```

`extra_signals` keys are `TaskType` value strings (`coding`, `sports`,
`document_analysis`, ...); unknown keys are dropped with a startup warning
rather than crashing the boot.

**Disabling either classifier** (falls back to exact Phase 2 behavior — no
`task_type`/`privacy_level`/`classification_reason`, no privacy override,
routing always sees `TaskType.GENERAL`):

```yaml
routing:
  classification:
    enabled: false
  privacy:
    enabled: false
```

or via env vars: `NEXUS_ROUTING__CLASSIFICATION__ENABLED=false`,
`NEXUS_ROUTING__PRIVACY__ENABLED=false`.

## Observed-latency-aware routing

`nexus/core/latency_tracker.py`'s `LatencyTracker` times every real
`generate()`/`stream_generate()` call and keeps a bounded rolling window
(`routing.latency.window_size`, default 50) per `(provider, model)` in
memory, alongside a full history persisted to SQLite for observability.
Once a model has at least `routing.latency.min_samples` (default 5)
observations, the `LOW_LATENCY` and `BALANCED` policies use its real
observed p50 latency instead of the Phase 2 "local is always fastest"
assumption — normalized against the slowest observed candidate, the same
way cost is normalized against the most expensive one. A model with no
data yet falls back to the old heuristic *for that model only*, so cold
starts never break routing for the rest of the candidates.

## Documents + RAG

Ingest a document, then ask a question with `use_rag: true` to have NEXUS
retrieve the most relevant chunks and inject them into the prompt as a
synthetic system message, with citations back in the response.

- **Local-first**: the default embedding backend is Ollama
  (`rag.embedding_provider: local`, using `local.models.embeddings` —
  `nomic-embed-text` by default), so ingestion and retrieval work fully
  with zero cloud keys configured. Set `rag.embedding_provider: openai`
  to use OpenAI's `text-embedding-3-small` instead — only takes effect if
  `OPENAI_API_KEY` is also set; otherwise NEXUS falls back to local,
  matching the same local-first fallback the chat providers already use.
- **Chunking**: character-based sliding window (`rag.chunk_size`,
  `rag.chunk_overlap`, defaults 800/150) — simple and dependency-free.
  Semantic/AST-aware chunking is a documented future upgrade.
- **Vector store**: `nexus/memory/sqlite_vector_store.py` is a brute-force
  cosine-similarity search over chunks stored in SQLite. This is an MVP
  choice, intentionally not an ANN index — fine at personal/small-team
  scale (hundreds to low thousands of chunks). It sits behind the
  `VectorStore` interface (`nexus/core/vector_store.py`) so a real vector
  DB can replace it later without touching `RagService` or `chat.py`.
- **Reranking**: deterministic — `final_score = 0.75 * cosine_score +
  0.25 * keyword_overlap_score` (`nexus/rag/reranker.py`). No extra model
  call, per the project's "never use an LLM where deterministic software
  is better" rule.
- **Supported formats**: `txt`, `md`, `code` (decoded directly), `csv`
  (row-by-row `col: val` text), `pdf` (`pypdf`), `docx` (`python-docx`).

```bash
# Ingest a document (content must be base64-encoded)
curl -X POST http://localhost:8100/api/documents \
  -H 'Content-Type: application/json' \
  -d "{\"user_id\": \"me\", \"source_name\": \"notes.txt\", \"source_type\": \"txt\", \"content_base64\": \"$(base64 -w0 notes.txt)\"}"

# List / delete
curl http://localhost:8100/api/documents?user_id=me
curl -X DELETE http://localhost:8100/api/documents/<doc_id>?user_id=me

# Ask a RAG-augmented question
curl -X POST http://localhost:8100/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "What does the document say about X?"}], "user_id": "me", "use_rag": true}'
```

## Long-term memory

Simple per-user key/value fact storage (`nexus/memory/long_term.py`), a
deliberately separate store from session-scoped conversation history —
different lifecycle, different scoping key (`user_id`, not `session_id`).

```bash
curl -X POST http://localhost:8100/api/memory/me -d '{"key": "favorite_team", "value": "Arsenal"}' -H 'Content-Type: application/json'
curl http://localhost:8100/api/memory/me
curl -X DELETE http://localhost:8100/api/memory/me/item/favorite_team
curl -X POST http://localhost:8100/api/memory/me/clear
```

## Tool calling

Every tool is individually opt-in — listed in `tools.enabled` in
`nexus/config/nexus.yaml` — before the model can see or call it
(`nexus/tools/`):

| Tool | What it does | Notes |
| --- | --- | --- |
| `web_search` | DuckDuckGo HTML search, top N results | no API key; unofficial scrape, keep volume low |
| `python` | Runs a Python snippet in a subprocess | `-I` + timeout is **process isolation, not a security sandbox** — never expose to untrusted input |
| `files` | Read-only file access | confined to `tools.files_allowed_root` (default `nexus/data/workspace/`), rejects path traversal |
| `database` | Read-only `SELECT` against NEXUS's own SQLite DB | e.g. query `cost_records`/`latency_records`; rejects non-SELECT and chained statements |
| `github` | Read-only `get_file_contents`/`search_code` | **requires both** `GITHUB_TOKEN` set **and** `github` added to `tools.enabled` — omitted by default even with a token present, per the "every tool is individually enabled" rule |

Set `use_tools: true` on a chat request to let the model call any enabled
tool. NEXUS runs an internal loop (`routing.tools.max_iterations`, default
3): call the model with tool schemas attached → if it returns tool calls,
execute them and feed the results back as `role: "tool"` messages → repeat
until the model returns a plain answer or the iteration cap is hit (in
which case the last generated content is still returned, with a note in
`classification_reason`, never an error). `use_tools=True` also forces
routing to a tool-capable model/provider regardless of policy
(`ModelInfo.supports_tool_calling` — currently `gpt-4o`, `gpt-4o-mini`,
`claude-sonnet-4-5`; local models via Ollama don't reliably support
structured tool calling, so `OllamaRuntime` always returns no tool calls
by design).

**Scope boundary**: `use_tools=True` with `stream=True` does *not* stream
token-by-token — the loop above only ever calls non-streaming `generate()`
internally (there's nowhere for a streamed tool-call delta to go; the
final decision only exists once the loop finishes), so the complete final
answer is sent as a single SSE content chunk. This is an intentional
Phase 5 scope boundary, not a bug.

```bash
curl -X POST http://localhost:8100/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "What is 47 * 89? Use Python to compute it."}], "use_tools": true}'
```

## Agents (Phase 6)

Four goal-driven agents (`nexus/agents/`), each with its own tool
allowlist and system prompt, all executed by `AgentRuntime`
(`nexus/agents/runtime.py`) through the exact same components `/api/chat`
uses — `ModelRouter.route_with_failover(require_tool_calling=True)` for
routing, and the shared `run_tool_loop()` (`nexus/core/tool_loop.py`,
extracted from what used to be a chat.py-only `_run_tool_loop()`) for
execution. There is no second tool-calling implementation.

| Agent | Allowed tools | Notes |
| --- | --- | --- |
| `research` | `web_search`, `files` | Also pulls RAG chunks for the goal via `RagService.retrieve()` and injects them as context, same format as `/api/chat`'s `use_rag` path |
| `coding` | `files`, `python`, `github` | Reads code before proposing changes; verifies via `python` where possible |
| `data` | `python`, `database`, `files` | Inspects actual data shape before analyzing; reports sample sizes |
| `planning` | `files` | Injects the user's current personal state + top weaknesses (Phase 7) into its own context — plans adapt to real capacity, not just the stated goal |

**Tool permission enforcement** happens inside `run_tool_loop()`, not just
in each agent's declared `allowed_tools` list: the model is only ever
shown schemas for tools in that allowlist (it can't "discover" anything
else), and if a model asks for a tool outside the allowlist anyway, the
call is refused with a `role: "tool"` message explaining why — it is
never executed. `test_agent_permissions.py` covers both halves of this.

Enabled agents are gated by `agents.enabled` in `nexus/config/nexus.yaml`
(default: all four) — an agent class existing in code is not enough by
itself, the same "every capability is individually opt-in" pattern
`tools.enabled` already uses.

```bash
# List enabled agents
curl http://localhost:8100/api/agents

# Run the research agent on a goal
curl -X POST http://localhost:8100/api/agents/research/run \
  -H 'Content-Type: application/json' \
  -d '{"goal": "What are the key differences between REST and GraphQL?", "user_id": "me"}'
```

The response (`AgentRunResponse`) includes a full step trace
(`steps: [{index, thought, tool_name, tool_arguments, tool_output,
timestamp}, ...]`), `completed` (false if `agents.max_iterations` — or a
per-request `max_iterations` override in the request body — was hit
before a final answer), `iterations_used`, `usage`, `cost_usd`, and which
`model_used`/`provider_name` actually ran it.

## Personal Intelligence (Phase 7)

A generic, domain-agnostic engine for "who is this user and how are they
doing right now" — `nexus/personal/`. Deliberately pure deterministic
arithmetic, no LLM calls anywhere in this layer (only an agent's own
planning step uses a model): state is a confidence- and recency-weighted
mean over append-only signals, baselines are historical means, trends are
a hand-rolled least-squares fit, and weaknesses are a sign-corrected
deviation from baseline. Health/fitness/sports *interpretation* of this
data is explicitly out of scope here — that's Group C (Phase 8-10).

**Dimensions** (`nexus/personal/dimensions.py`) — four groups, each
member a `group.name` string normalized to `0.0`-`1.0`:

- `physical.*`: energy, recovery, activity, mobility, strength, endurance
- `mental.*`: focus, stress, mood, fatigue
- `cognitive.*`: working_memory, processing_speed, creativity
- `lifestyle.*`: sleep_quality, sleep_consistency, nutrition, hydration

`mental.stress` and `mental.fatigue` are the two **inverted** dimensions
(higher = worse) — every comparison (deviation sign, trend direction)
sign-corrects through `INVERTED_DIMENSIONS` so "worse" and "improving"
always mean the same thing regardless of which kind of dimension is being
described.

**Signals** are append-only evidence, each carrying a `source` that
determines how much it's trusted: `explicit` (user-reported, 0.95) >
`behavioral` (inferred from interaction, 0.75) > `temporal` (time-pattern
based, 0.65) > `inferred` (correlated signals, 0.55). State is always
*derived* from signal history (`PersonalStateEngine.get_state()`), never
stored directly, so it stays fully recomputable as weighting logic
evolves.

```bash
# Record a few signals for a user
curl -X POST http://localhost:8100/api/personal/me/signal \
  -H 'Content-Type: application/json' \
  -d '{"dimension": "physical.energy", "value": 0.4, "source": "explicit", "note": "felt drained today"}'
curl -X POST http://localhost:8100/api/personal/me/signal \
  -H 'Content-Type: application/json' \
  -d '{"dimension": "mental.stress", "value": 0.75, "source": "explicit"}'

# Current confidence-/recency-weighted state
curl http://localhost:8100/api/personal/me/state

# Historical baselines (excludes the last personal.state_recent_window_days
# — comparing "current" against itself would make every deviation trivial)
curl http://localhost:8100/api/personal/me/baselines

# Prioritized weaknesses: current vs. baseline, sign-corrected, filtered by
# personal.weakness_min_confidence AND personal.weakness_min_deviation,
# priority bumped one level if the dimension is also trending in the wrong
# direction
curl http://localhost:8100/api/personal/me/weaknesses

# Stated goals/preferences/constraints — free-form JSON
curl -X PUT http://localhost:8100/api/personal/me/profile \
  -H 'Content-Type: application/json' \
  -d '{"profile": {"primary_goal": "improve endurance", "constraints": ["knee injury — avoid high-impact"]}}'
curl http://localhost:8100/api/personal/me/profile

# Full deletion — removes every signal AND the profile for this user
curl -X DELETE http://localhost:8100/api/personal/me
```

**Injecting personal context into chat** — set `use_personal_context:
true` on a `/api/chat` request. This is gated two ways:

1. `personal.enabled` (default `true`) — a hard off switch.
2. **`personal.local_only_context` (default `true`)** — personal data is
   inherently sensitive, so by default it is withheld whenever the
   request's *actual resolved route* is a non-local provider, regardless
   of what policy was requested. This check runs **after**
   `route_with_failover()` resolves a concrete `(provider, model)` pair —
   not before — because an explicit `model_id`, a policy, or failover
   itself can all still land a request on a cloud provider even when the
   caller didn't obviously ask for one; only the resolved decision tells
   you where the request is actually about to go.

Either way, the response's `personal_context_used` field tells the caller
what happened: `null` if `use_personal_context` wasn't set at all,
`false` if it was requested but withheld (privacy gate, `personal.enabled:
false`, or simply no signals recorded yet for this user), `true` if it
was actually injected. When both RAG and personal context are on for the
same request, personal context is inserted first, then RAG context, then
the conversation.

```bash
# Local route — personal context is injected (assuming personal.enabled
# and personal.local_only_context are both at their defaults)
curl -X POST http://localhost:8100/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "How should I train today?"}], "user_id": "me", "use_personal_context": true, "model_id": "mistral:7b"}'

# Cloud route — withheld by default; personal_context_used will be false
curl -X POST http://localhost:8100/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "How should I train today?"}], "user_id": "me", "use_personal_context": true, "model_id": "gpt-4o-mini"}'
```

## Health Intelligence (Phase 8)

`nexus/health/analyzer.py`'s `HealthAnalyzer` reads `PersonalStateEngine` /
`BaselineCalculator` / `WeaknessEngine` output and applies a fixed set of
deterministic rules (`RULES`, a module-level list — add a new rule by
adding a function to it, no changes to `HealthAnalyzer` itself) to detect
multi-dimension **patterns**, never diagnoses:

| Pattern | Fires when |
| --- | --- |
| `high_load_low_recovery` | `physical.activity` above baseline AND `physical.recovery` below baseline |
| `sleep_inconsistency` | `lifestyle.sleep_consistency` below baseline AND trending down |
| `stress_accumulation` | `mental.stress` trending up over the window (inverted-dimension sign-corrected) |
| `fatigue_without_load` | `mental.fatigue` high while `physical.activity` is at/below baseline — flagged specifically because it is NOT explained by training |
| `recovery_debt` | `physical.recovery` below baseline across a sustained run of samples, not a single reading |
| `cognitive_dip` | `mental.focus` / `cognitive.working_memory` below baseline alongside poor `lifestyle.sleep_quality` |

Each pattern carries its own `severity` (`watch` / `notable` /
`significant`), `confidence`, `sample_size`, and a plain-language
`explanation` that cites the actual numbers. `data_sufficiency` is
`"none"` (no signals at all), `"sparse"` (not even the best-sampled
dimension reaches `health.min_sample_size`), or `"adequate"` — below
`"adequate"`, every pattern's severity is capped to `"watch"` no matter
how large the raw deviation is, so thin data can never support a
`"significant"` claim. `scorecard` is a confidence-weighted, sign-corrected
0..1 aggregate per dimension group (physical/mental/cognitive/lifestyle),
omitting any group with no data rather than defaulting it to 0.5.

**The safety layer (`nexus/health/safety.py`) is unconditional** — there
is no config flag to disable it, by design:

- `check_input(text)` matches incoming text against `RED_FLAG_PATTERNS`
  (chest pain, breathing difficulty, fainting, self-harm/suicidal
  ideation, sudden severe pain, uncontrolled bleeding, stroke symptoms,
  pregnancy complications). A match short-circuits the entire health path
  — no analysis, no reassurance, no delay — and returns an urgent-care
  referral message.
- `check_output(text)` matches any LLM-generated health text against
  `PROHIBITED_OUTPUT_PATTERNS` (diagnostic assertions, medication/dosage
  instructions, and phrasing that discourages seeking care). A match
  blocks the text and substitutes a safe rewrite plus
  `MEDICAL_DISCLAIMER`.
- `MEDICAL_DISCLAIMER` is a single versioned constant appended to every
  health API response (`disclaimer` field) — never re-typed inline.
- This layer runs on **both** `/api/health-intel/*` and `HealthAgent`.
  For the agent specifically, it's wired through `Agent.output_filter`
  (`nexus/agents/base.py`) — an optional, generic hook `AgentRuntime`
  applies to `final_answer` when an agent sets it, so the filter can't be
  forgotten or bypassed by a caller (`HealthAgent.output_filter =
  staticmethod(check_output)`).

`HealthAgent` (`nexus/agents/health.py`) routes `LOCAL_ONLY` unconditionally
(health data never leaves the machine, regardless of requested policy),
injects the structured `HealthAnalysis` into its own context, and is
instructed to interpret only what it's given, never diagnose, never
discuss medication, and always recommend professional evaluation for
`"significant"` patterns or insufficient data.

```bash
# Record a couple of signals, then pull the analysis
curl -X POST http://localhost:8100/api/personal/me/signal \
  -d '{"dimension": "physical.recovery", "value": 0.25, "source": "explicit"}' -H 'Content-Type: application/json'
curl -X POST http://localhost:8100/api/personal/me/signal \
  -d '{"dimension": "physical.activity", "value": 0.85, "source": "explicit"}' -H 'Content-Type: application/json'

curl http://localhost:8100/api/health-intel/me/analysis
curl http://localhost:8100/api/health-intel/me/scorecard

# Ask HealthAgent to interpret it (routes local-only automatically)
curl -X POST http://localhost:8100/api/agents/health/run \
  -H 'Content-Type: application/json' \
  -d '{"goal": "How is my recovery looking this week?", "user_id": "me"}'
```

## Personalized Generation (Phase 9)

`nexus/generation/planner.py`'s `BriefBuilder` assembles a `GenerationBrief`
— the deterministic INPUT to generation — from current state, weaknesses,
health patterns, and profile. `target_intensity` is arithmetic, not a
model call: start at `generation.default_intensity` (0.7), subtract a
weighted penalty for each of low `physical.recovery`, high
`mental.fatigue`, high `mental.stress`, and poor `lifestyle.sleep_quality`
that shows up as a detected `Weakness`, then clamp to
`[generation.min_intensity, generation.max_intensity]`. Every adjustment
(and any clamping) appends a human-readable line to `rationale`, which is
surfaced to the caller, not hidden. With zero recorded signals, the brief
uses the neutral default intensity and says explicitly in `rationale`
that the plan is generic — it never implies personalization that didn't
happen.

`nexus/generation/service.py`'s `PersonalizedGenerator` renders the brief
into a plan via the LLM: the model receives the brief as structured facts
and must justify every element of the plan against something specific in
it (a weakness, a health pattern, the stated intensity, a constraint) — it
never chooses the intensity or invents the user's state itself. When
`request_type` is `"workout"` or `"recovery"` **and** the health analysis
contains a `"significant"`-severity pattern, the generated plan is passed
through `nexus/health/safety.py`'s `check_output()` as well — training
advice layered on top of a concerning health pattern is exactly where
unsafe text could appear.

```bash
# A low-recovery, high-activity user asking for a workout — watch
# target_intensity and brief_rationale drop below the 0.70 default
curl -X POST http://localhost:8100/api/generate/workout \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "me", "constraints": {"time_minutes": 45, "equipment": "bodyweight only"}}'
```

## Sports Intelligence (Phase 10)

NEXUS **consumes** the football computer-vision pipeline that already
lives in this repo (`backend/api/*`, a separate FastAPI app) — it never
imports from `backend/` or `ai/`, and never touches that system's
database directly. `nexus/sports/adapter.py`'s `HttpSportsDataAdapter` is
the one integration seam, calling that backend over HTTP
(`sports.backend_base_url`, default `http://localhost:8000`). HTTP rather
than a direct DB connection keeps the two systems decoupled — NEXUS stays
independently deployable and can point at a remote football-backend
instance instead of a co-located one.

**The football pipeline's own scorers deliberately return `value: None`
with `confidence: "low_upstream_confidence"`** when their inputs aren't
trustworthy (e.g. team-assignment confidence too low) — the pipeline's
own code calls fabricating a confident answer over that "confident-looking
garbage" and explicitly rejects it. `SportsMetric.is_available` preserves
that distinction end to end:

```python
is_available = value is not None and confidence != "low_upstream_confidence"
```

Every metric the adapter returns is partitioned into `team_metrics` /
`player_metrics` (all metrics) and `unavailable` (the ones that failed
`is_available`) — a `None` value is never coerced to `0.0`, and an
unavailable metric is never silently dropped, since the fact that
something couldn't be measured is itself information a coach needs.
`coverage` is `available / total` (`0.0` when there are no metrics at
all, never treated as "fully covered").

`nexus/sports/tactical.py`'s `derive_findings()` thresholds AVAILABLE
metrics only (0-100 scale, matching the pipeline's own scorers) into
`TacticalFinding`s (`strength` / `neutral` / `weakness` per area) — an
area whose underlying metric is unavailable simply produces no finding;
findings are never derived from, or backfilled for, an unavailable
metric. `nexus/sports/coach.py`'s `CoachAssistant` then does
`adapter -> derive_findings -> LLM narrates`: the model is given the
derived findings plus an explicit `unavailable_metrics` list (name + why)
and instructed to report gaps as unavailable, never estimate them —
`CoachReport.narrative` is the only LLM-generated field on the whole
report.

Available player metrics can also be fed back into the personal state
engine as `source="inferred"` signals (`nexus/sports/ingest.py`) — CV-derived
proxies, not self-reported facts, hence the lower (0.55) confidence tier
— normalizing the pipeline's 0-100 scale to NEXUS's 0..1 scale via
`sports.metric_dimension_map` (config-driven, since the pipeline's metric
set is expected to grow). Unavailable metrics are skipped entirely, never
ingested as a fabricated 0.

```bash
# Pull a match-level coach report (works even with partial pipeline coverage)
curl http://localhost:8100/api/sports/<match_id>/report

# A single player's report
curl http://localhost:8100/api/sports/<match_id>/player/<player_id>

# Feed a player's available metrics into their personal state as inferred signals
curl -X POST http://localhost:8100/api/sports/<match_id>/ingest \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "me", "player_id": <player_id>}'
```

`SportsAgent` (`nexus/agents/sports.py`) is the agent-framework entry
point for the same data — pass `match_id` (and optionally `player_id`) in
the `POST /api/agents/sports/run` body; it's threaded through
`AgentContext.extra` since the generic agent-run request schema doesn't
otherwise know about matches.

## Verification (Phase 11)

Set `verify: true` on a `/api/chat` request to have NEXUS check its own
answer before returning it. `nexus/verification/engine.py`'s
`VerificationEngine` runs, in order:

1. **Five deterministic checks** (`nexus/verification/checks.py`), always,
   free (no LLM call): `arithmetic` (recomputes "X + Y = Z"-shaped
   assertions in Python), `citation_support` (lexical overlap between the
   answer and any RAG evidence — only runs if `use_rag` supplied chunks),
   `internal_contradiction` (the same named quantity asserted with two
   different numbers, or a direct negation pair), `unsupported_certainty`
   (absolute-certainty language — "definitely", "always" — with no
   citation or reasoning backing it), `refusal_consistency` (the answer
   says it can't determine something, then asserts it anyway).
2. **Model-based fact-checking** (`nexus/verification/fact_checker.py`,
   opt-in via `verification.enable_fact_check`): extracts discrete
   checkable claims, then checks each against RAG evidence. A claim with
   **no evidence provided at all** is `INCONCLUSIVE` ("we couldn't
   check"); a claim with evidence present that simply doesn't support it
   is `FAIL` ("we checked and it isn't there") — collapsing those two is
   exactly the kind of fabricated confidence this phase exists to
   prevent.
3. **Multi-model judging** (`nexus/verification/judge.py`,
   `verification.enable_judge`), only when the score from steps 1-2 is
   below `verification.escalate_below` (default 0.65): consults a
   **different** model than the one that produced the answer (via the
   router's ranked candidate list, skipping the original model_id and any
   unhealthy provider). Returns `None` — not a passing check — when no
   distinct healthy model exists (e.g. a local-only deployment); the
   engine records that as `INCONCLUSIVE`, never as agreement.

**Scoring is honest by construction**: `score = sum(weight for PASS) /
sum(weight for PASS or FAIL)` — `INCONCLUSIVE` checks are excluded from
both sides of that fraction, so an unrunnable check can never inflate the
score. If every check is `INCONCLUSIVE` (denominator zero), `score` is
`None` and the band is `UNVERIFIED`, never `HIGH`. Bands: `HIGH` (≥0.85),
`MEDIUM` (0.65-0.84), `LOW` (0.45-0.64), `UNCERTAIN` (<0.45), `UNVERIFIED`
(no usable signal at all).

The response's `verification` field carries the full report (`checks`,
`score`, `band`, `summary`, `uncertainty_notes`, `escalated`,
`extra_usage`), and `nexus/verification/presenter.py` appends a footer to
`content`: `HIGH` gets one short line; `MEDIUM` a short line plus a
count; `LOW`/`UNCERTAIN`/`UNVERIFIED` all visibly say "NOT been fully
verified" and list every specific uncertainty note — never just a number.

Verification tokens are billed through the **same** `CostTracker` as the
main answer, but as a **separate ledger entry** (principle 4 — "no shadow
accounting"), so `session_total()` still reflects true spend without
silently folding verification cost into the main generation's line item.

**Forced verification**: `verification.always_verify_task_types` (default
`[health, research, mathematics]`) verifies automatically regardless of
the `verify` flag — the response still reports which happened via
`verification` being non-null.

**Streaming**: verification needs the complete answer, so with
`stream: true` and `verify: true`, content deltas stream normally
(unmodified — the confidence footer is presentation-layer for the
non-streaming response only) and the structured report rides in the
final `"done": true` SSE event's `verification` field, mirroring how
Group A's `stream` + `use_tools` sends its single content chunk before
that same event.

**Agents**: `Agent.output_filter`'s sibling attribute `verify_output:
bool = False` (default False — every existing agent is unchanged) opts
an agent into having `AgentRuntime` verify its `final_answer` and attach
the report to `AgentResult.verification`. Only `ResearchAgent` sets it —
the agent whose output is most citation-dependent and load-bearing.

```bash
# A clean, easily-verified answer — expect band=high
curl -X POST http://localhost:8100/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "What is 12 + 8?"}], "model_id": "mistral:7b", "verify": true}'

# health/research/mathematics verify automatically even without "verify": true
curl -X POST http://localhost:8100/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Solve this equation: 2x + 5 = 15"}], "model_id": "mistral:7b"}'
```

## Evaluation Platform (Phase 12)

`nexus/evaluation/` is a repeatable benchmark harness so a change to a
model, prompt, or routing rule can be **proven** non-regressive instead
of hoped to be — it measures NEXUS itself (routing decisions, RAG
retrieval, tool selection, safety behavior), not only answer quality.

**Datasets** (`nexus/evaluation/datasets/*.jsonl`, one JSON object per
line, version-controlled and human-editable): `routing.jsonl`,
`classification.jsonl`, `rag.jsonl`, `tools.jsonl`, `safety.jsonl`,
`verification.jsonl` — 10-15 hand-written cases each. `rag.jsonl` cases
declare the seed documents inline (`input.documents`) and identify the
expected match by `source_name` rather than `doc_id`, since `doc_id` is a
fresh uuid generated at ingest time and can't be hardcoded ahead of a
run.

**Harness** (`nexus/evaluation/runner.py`): `EvalHarness` owns every
component under test and **defaults to fake providers** — a scripted
`FakeChatProvider`, and a deterministic hashing bag-of-words
`EmbeddingProvider` for RAG (crude, but it genuinely differentiates
semantically-different text via real cosine similarity, unlike a
length-only fake) — so a normal eval run makes **zero real network
calls**. Pass `use_real_providers=True` (or `--real-providers` on the
CLI) to opt into hitting configured real providers instead.

Six suite evaluators (`nexus/evaluation/suites/`): `routing_eval.py` and
`classification_eval.py` are pure/fast (no LLM — they call
`TaskClassifier`/`PrivacyClassifier`/`ModelRouter` directly);
`rag_eval.py` seeds documents and reports recall@3; `tools_eval.py`
scripts a fake provider to force a specific `tool_call` and checks both
*which* tool ran and that allowlist permission filtering actually held;
`safety_eval.py` calls `nexus/health/safety.py` directly — **this suite
must be 100%**, any failure is a build breaker, not a percentage;
`verification_eval.py` runs planted-error and clean answers through
`VerificationEngine` and additionally reports precision/recall
(`nexus.evaluation.suites.verification_eval.precision_recall()`)
separately from the plain pass rate, since a checker that flags every
answer would score perfect recall while being useless.

**Persistence** (`nexus/evaluation/store.py`, `EvalStore` — same
`async init()`/engine pattern as `CostTracker`): every run is saved with
its full config snapshot (secrets stripped) and git SHA, so two runs can
be diffed later. Per-case `actual` detail is intentionally NOT persisted
(only `passed`/`score`/`detail`/`latency`/`cost` round-trip) — a live run
keeps full detail in memory, a reloaded one keeps the scoring history,
which is what regression comparison actually needs.

**Regression detection** (`nexus/evaluation/regression.py`):
`compare_runs(baseline, candidate)` flags a suite's `pass_rate` dropping
beyond `evaluation.pass_rate_tolerance` (default 2%), ANY individual case
flipping pass→fail (unconditional, no tolerance), and cost/latency
increasing beyond their tolerances. **The `safety` suite gets zero
pass_rate tolerance** regardless of the configured tolerance — a single
safety case regressing is always reported.

```bash
# Run the full suite locally (fakes only, no network)
python -m nexus.evaluation.cli

# Run specific suites, machine-readable output
python -m nexus.evaluation.cli --suites routing,safety --json

# Compare against a previous run — exits 1 if any regression is found
python -m nexus.evaluation.cli --compare-to <baseline_run_id>
```

**As a CI gate**: `python -m nexus.evaluation.cli [--compare-to <id>]`
exits `1` when any regression is found OR the safety suite isn't 100%,
and `0` otherwise — wire it as a required check (e.g. a GitHub Actions
step) with `--json` piped to whatever the CI system parses. `--suites`
lets a fast-path CI gate run only `safety` (and any suite touched by the
change) instead of the full set on every push.

```bash
curl -X POST http://localhost:8100/api/eval/run -H 'Content-Type: application/json' -d '{}'
curl http://localhost:8100/api/eval/runs
curl "http://localhost:8100/api/eval/compare?baseline=<id>&candidate=<id>"
```

## Custom NEXUS Model (Phase 13)

Full details, including the exact Blackwell install command and the OOM
remediation ladder, are in **[`nexus/training/README.md`](training/README.md)**.

The custom model is an **option, never a dependency**. Nothing in
`nexus/training/` is imported at module scope by the running API — the ML
stack lives only in `nexus/training/requirements-train.txt`, and nothing
was added to `nexus/requirements.txt`, so the API still deploys to a
machine with no CUDA toolchain.

### Enabling interaction logging, and what it stores

Off by default. It stores full prompts and responses, which is a consent
decision rather than a convenience default.

```yaml
training:
  log_interactions: false        # set true to start collecting
  interaction_retention_days: 180
  exclude_privacy_levels: [private]
```

When enabled, each answered request writes one `interactions` row:
session/user id, task type, privacy level, the **full prompt JSON and
response text**, the model and provider that served it, which tools ran,
and the verification band/score. That routing and verification metadata is
exactly what makes a record usable as a training example, and is why this
is a separate table from `messages` (conversation replay) with its own
consent and retention rules.

`POST /api/chat` then returns an `interaction_id` (`null` when logging is
off), and users can rate answers:

```bash
curl -X POST http://localhost:8100/api/feedback \
  -H 'Content-Type: application/json' \
  -d '{"interaction_id": 42, "feedback": 1}'
```

### train -> quantize -> register -> evaluate

```bash
# 1. Mine a dataset from four sources, re-classifying privacy at export
python -m nexus.training.dataset --output nexus/training/data/dataset.jsonl

# 2. Check the plan on any machine (imports nothing from the ML stack)
python -m nexus.training.train --dataset nexus/training/data/dataset.jsonl --dry-run

# 3. Verify this machine can actually train (Blackwell/sm_120, bitsandbytes,
#    FREE VRAM, Windows spillover, dataset, disk)
python -m nexus.training.preflight

# 4. Train (QLoRA, resumable — laptops throttle, runs get interrupted)
python -m nexus.training.train --dataset nexus/training/data/dataset.jsonl
python -m nexus.training.train --dataset nexus/training/data/dataset.jsonl --resume

# 5. Merge + export GGUF + write an Ollama Modelfile (stop Ollama first —
#    merging peaks HIGHER than training)
python -m nexus.training.quantize --adapter nexus/training/output/adapter
ollama create nexus-custom -f nexus/training/output/Modelfile

# 6. Uncomment the nexus-custom entry in models.yaml, then gate promotion
python -m nexus.training.evaluate --model-id nexus-custom
```

The registered model serves through the **existing** `OllamaRuntime` — a
fine-tuned model is just another registry entry, with no new provider
code. Its capabilities start at `0.0` deliberately: an untested model must
never outrank a proven one on day one. Phase 14's `CapabilityMatrix` earns
it real scores from measured eval runs.

`evaluate` is a real gate: a model that regresses safety or drops
pass-rate is **not promoted regardless of subjective quality**, and the
command exits non-zero.

## Advanced Intelligence (Phase 14)

### Learned capability matrix — closing the improvement loop

`nexus/intelligence/capability_matrix.py` turns measured eval results into
routing decisions, which is what makes the eval platform a feedback loop
rather than a report.

```yaml
capability_learning:
  enabled: true
  min_samples: 20
  max_learned_weight: 0.7
```

```
learned_weight = min(max_learned_weight, sample_size / (sample_size + 20))
effective      = learned_weight * measured + (1 - learned_weight) * static
```

**Zero samples yields exactly the static `models.yaml` value**, so an
install that has never run an eval routes identically to one without the
matrix at all. Below `min_samples` the learned score carries no weight —
a handful of cases must never swing model selection. The cap means a
static prior always keeps a say.

Reads are synchronous off an in-memory cache (the same pattern
`LatencyTracker` uses for `p50()`), because `_ranked_decisions()` is not
async. `safety` and `forecast` are deliberately **excluded** from the
suite->capability map: safety is a gate rather than a skill, and forecast
is deterministic arithmetic that exercises no model.

### Graph RAG

Off by default — entity extraction is the one genuinely LLM-dependent step
in this phase.

```yaml
rag:
  graph:
    enabled: false
    max_hops: 2
    max_nodes: 20
```

It **plugs into** the existing pipeline rather than replacing it: vector
retrieve -> optional graph expand -> rerank the combined set with the
existing reranker. With `enabled: false` there is zero behavior change —
no extra queries, no extra provider calls.

What it buys you is material that shares **no vocabulary with the query**
and is therefore invisible to vector search, but is reachable through a
shared entity. Graph-expanded chunks enter the rerank carrying a
provenance score decayed per hop, so a two-hop connection is weaker
evidence than a one-hop one. Hop and node caps are enforced in the
traversal loop.

### Multi-agent orchestration

```yaml
agents:
  enabled: [research, coding, data, planning, health, sports,
            orchestrator, autonomous_research]
  orchestration: {max_depth: 2, max_total_delegations: 6}
  research: {max_rounds: 3}
```

`OrchestratorAgent` decomposes a goal, delegates each sub-goal to the
best-suited specialist, and synthesizes. Delegation is exposed as a
`delegate` pseudo-tool through the **existing** `ToolRegistry`, so it
flows through the same permission checks and tool-call loop as everything
else — there is no parallel control path.

`DelegationGuard` enforces the caps **in `AgentRuntime`**, not in a
prompt: max depth, max total delegations, and rejection of an exact repeat
`(agent_name, sub_goal)` within a run. Exceeding a cap returns a tool
result saying so, and the orchestrator must synthesize from what it has.
The delegation tree comes back in `AgentRunResponse.delegation_steps`.

```bash
curl -X POST http://localhost:8100/api/agents/orchestrator/run \
  -H 'Content-Type: application/json' \
  -d '{"goal": "Investigate why our API latency regressed, then propose a fix."}'
```

### Autonomous research

`AutonomousResearchAgent` iterates search -> assess coverage -> identify
what remains unknown -> search again, up to `max_rounds`. **The LLM
proposes follow-up queries; the code decides when to stop** — a
deterministic coverage check asks whether any sub-question still has zero
supporting evidence. Output separates Known / Likely / Uncertain /
Unknown, computed from the evidence rather than asked of the model.

### Model self-evaluation

```yaml
self_eval:
  enabled: false        # a second generation per answer
  log_low_scores: true
```

`SelfEvaluator` critiques an answer for **completeness** and whether it
addressed the question — distinct from `VerificationEngine`, which checks
**correctness** against evidence. A self-eval finding is additive signal
and **never overrides a verification verdict**: it appends to
`uncertainty_notes` and leaves score, band, and checks untouched. A model
grading its own homework must not be able to raise its own confidence.
Low-scoring self-evals are flagged for dataset mining, feeding Phase 13's
`eval_failure` source — that is how the loop actually closes.

### Digital-twin forecasting

```yaml
personal:
  forecast: {max_horizon_days: 30, min_trend_confidence: 0.5}
```

`StateForecaster` projects each dimension along the slope `compute_trend()`
already fits. Pure arithmetic. Confidence decays with horizon (halving
every 14 days), the horizon is hard-capped at 30 days, and projections are
clamped to the `[0, 1]` scale.

Where the data is too thin it returns `method="insufficient_data"` and
**no number at all** — the same "no data means say nothing" discipline
`build_personal_context_message()` follows.

`ScenarioSimulator` answers "what if sleep_consistency improved by 0.2?"
by applying the delta and re-running the **existing** `WeaknessEngine`
thresholds. It is purely mechanical and says so: dimensions here are not
causally wired to each other, so it reports which weaknesses would stop
being flagged and makes no claim beyond that.

```bash
curl "http://localhost:8100/api/personal/u1/forecast?horizon_days=14"

curl -X POST http://localhost:8100/api/personal/u1/simulate \
  -H 'Content-Type: application/json' \
  -d '{"deltas": {"lifestyle.sleep_consistency": 0.2}}'
```

### Video understanding

`POST /api/sports/video/analyze` submits footage to the football backend's
existing processing endpoint, polls until it finishes, and returns a
`CoachReport` through the Group C adapter.

**No computer vision runs in `nexus/`.** NEXUS orchestrates and narrates;
`backend/` and `ai/` do the vision, reached over HTTP. That separation is
the entire point of the seam — it is also why `nexus/` can be deployed
independently or pointed at a remote football backend.

```bash
curl -X POST http://localhost:8100/api/sports/video/analyze \
  -H 'Content-Type: application/json' \
  -d '{"video_path_or_url": "/data/match118.mp4", "match_id": "118"}'
```

### Deliberately deferred: voice and multimodal memory

The architecture doc lists eleven Phase 14 capabilities. This build
implements the six that compose cleanly with what already exists, plus
video through the already-built CV pipeline.

**Voice interaction** and **image/multimodal memory** are deferred, not
stubbed. Both need new runtime dependencies (STT/TTS engines, image
embedding models) and, for voice, a real-time audio transport — which
would roughly double this phase's surface area for capability unrelated to
everything else here. There is no placeholder implementation of either;
when they are built they should be built properly.

## API

### `GET /api/health`

Reports per-provider connectivity and which local models are currently
loaded:

```json
{
  "status": "ok",
  "providers": {"local": true, "openai": true, "anthropic": false},
  "local_models": ["mistral:7b", "nomic-embed-text"]
}
```

`status` is `"degraded"` if any *configured* provider is unhealthy, but
the endpoint always returns `200 OK` — a down cloud provider (or even a
down local backend) is reported, never raised as a server error.

### `POST /api/documents`, `GET /api/documents`, `DELETE /api/documents/{doc_id}`

Ingest, list, and delete documents for RAG — see
[Documents + RAG](#documents--rag) above for request/response shapes and
example `curl` calls.

### `GET/POST /api/memory/{user_id}`, `DELETE /api/memory/{user_id}/item/{key}`, `POST /api/memory/{user_id}/clear`

Long-term fact CRUD — see [Long-term memory](#long-term-memory) above.

### `GET /api/agents`, `POST /api/agents/{name}/run`

List enabled agents / run one against a goal — see [Agents](#agents-phase-6)
above.

### `GET/POST /api/personal/{user_id}/signal`, `GET /api/personal/{user_id}/state`, `GET /api/personal/{user_id}/baselines`, `GET /api/personal/{user_id}/weaknesses`, `GET/PUT /api/personal/{user_id}/profile`, `DELETE /api/personal/{user_id}`

Personal signal/state/baseline/weakness/profile CRUD, plus full-user
deletion — see [Personal Intelligence](#personal-intelligence-phase-7)
above.

### `GET /api/health-intel/{user_id}/analysis`, `GET /api/health-intel/{user_id}/scorecard`

Deterministic health-pattern analysis and scorecard, both carrying
`MEDICAL_DISCLAIMER` — see [Health Intelligence](#health-intelligence-phase-8)
above.

### `POST /api/generate/{request_type}`

`request_type` one of `workout`, `recovery`, `study`, `daily_plan`,
`nutrition`; body `{user_id, constraints}` — see [Personalized
Generation](#personalized-generation-phase-9) above.

### `GET /api/sports/{match_id}/report`, `GET /api/sports/{match_id}/player/{player_id}`, `POST /api/sports/{match_id}/ingest`

Coach report (match- or player-scoped) and metric-to-signal ingestion —
see [Sports Intelligence](#sports-intelligence-phase-10) above.

### `POST /api/eval/run`, `GET /api/eval/runs`, `GET /api/eval/runs/{run_id}`, `GET /api/eval/compare`

Run the eval harness, list/inspect past runs, and diff two runs for
regressions — see [Evaluation Platform](#evaluation-platform-phase-12)
above.

### `POST /api/feedback`

Rates a logged interaction: `{"interaction_id": 42, "feedback": -1|0|1}`.
Returns 404 when interaction logging is disabled (there is nothing to
rate) or the id doesn't exist. Feeds Phase 13's `high_feedback` dataset
source — see [Custom NEXUS Model](#custom-nexus-model-phase-13) above.

### `GET /api/personal/{user_id}/forecast`, `POST /api/personal/{user_id}/simulate`

Trend extrapolation per dimension (`?horizon_days=14`, capped at 30), and
mechanical what-if recalculation against the existing weakness
thresholds. Both return an explicit caveat; the forecast returns
`method: "insufficient_data"` and no number where the data is too thin —
see [Advanced Intelligence](#advanced-intelligence-phase-14) above.

### `POST /api/sports/video/analyze`

`{"video_path_or_url": "...", "match_id": "...", "player_id": null}` —
submits to the football backend's processing endpoint, polls to
completion, and returns a `CoachReport`. Returns 503 if the backend is
unreachable or processing failed.

### `POST /api/chat`

```json
{
  "messages": [{"role": "user", "content": "hello"}],
  "model_id": null,
  "session_id": null,
  "temperature": 0.7,
  "max_tokens": null,
  "stream": false,
  "policy": null,
  "user_id": "default",
  "use_rag": false,
  "rag_top_k": 5,
  "use_tools": false,
  "use_personal_context": false,
  "verify": false
}
```

Omit `session_id` on the first call; the response returns one to reuse
for follow-up turns so conversation history is included automatically.

- `model_id`: pin an explicit model (e.g. `"mistral:7b"`, `"gpt-4o-mini"`,
  `"claude-sonnet-4-5"`). Explicit requests always win, but still fail
  over to the next-best candidate (and ultimately local) if that model's
  provider is unhealthy.
- `policy`: one of `MAX_QUALITY`, `BALANCED`, `LOW_COST`, `LOW_LATENCY`,
  `LOCAL_ONLY`. Omit (or `null`) to use the server default configured in
  `nexus/config/nexus.yaml` under `routing.default_policy` (`BALANCED`).
  Ignored when `model_id` is set.
- `user_id`: scoping key for documents/long-term memory — there's no auth
  system yet, so this is a plain client-supplied string (defaults
  `"default"`), the same deferred-auth approach `session_id` already uses.
- `use_rag` / `rag_top_k`: see [Documents + RAG](#documents--rag) above.
- `use_tools`: see [Tool calling](#tool-calling) above.
- `use_personal_context`: see [Personal Intelligence](#personal-intelligence-phase-7)
  above.
- `verify`: see [Verification](#verification-phase-11) above. Also forced
  on automatically for task types in `verification.always_verify_task_types`.

Non-streaming response:

```json
{
  "content": "...",
  "model_used": "gpt-4o-mini",
  "usage": {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46},
  "session_id": "...",
  "provider_name": "openai",
  "routing_reason": "BALANCED policy: gpt-4o-mini scored 0.81 (capability 0.75, cost-normalized 0.94, observed p50 latency-normalized 0.88 from 12 samples) for task=coding.",
  "cost_usd": 0.00003,
  "task_type": "coding",
  "privacy_level": "public",
  "classification_reason": "task: Classified as coding via 3 matched signal(s): keyword:\\bfunction\\b, keyword:\\bbug\\b, keyword:\\berror\\b. | privacy: No private or sensitive signals matched; defaulting to public.",
  "citations": null,
  "tool_calls_made": null,
  "personal_context_used": null,
  "verification": null
}
```

`provider_name`, `routing_reason`, `cost_usd`, `task_type`,
`privacy_level`, `classification_reason`, `citations`, `tool_calls_made`,
`personal_context_used`, and `verification` are all additive — every
Phase 1 field is unchanged, so existing Phase 1 clients keep working.
`cost_usd` is `0.0` for local models (no configured cost rates); the
classification fields are `null` when their classifier is disabled in
config; `citations` is `null` unless `use_rag` was set (then a list,
possibly empty); `tool_calls_made` is `null` unless `use_tools` was set
(then a list, possibly empty); `personal_context_used` is `null` unless
`use_personal_context` was set (then `true`/`false` — see [Personal
Intelligence](#personal-intelligence-phase-7) above for what `false`
can mean); `verification` is `null` unless verification actually ran
(`verify: true`, or a forced task type — see
[Verification](#verification-phase-11) above).

Set `"stream": true` to instead receive `text/event-stream` chunks; the
final `"done": true` event carries the same routing/cost/classification/
citations/tool_calls_made/personal_context_used/verification fields as
the non-streaming response (content deltas stream unmodified; the
verification report only ever appears in this final event, never mixed
into delta text — see [Verification](#verification-phase-11) above):

```
data: {"delta": "Hel", "done": false}

data: {"delta": "lo", "done": false}

data: {"delta": "", "done": true, "session_id": "...", "provider_name": "local", "routing_reason": "...", "cost_usd": 0.0, "task_type": "coding", "privacy_level": "public", "classification_reason": "...", "citations": null}
```

### Example requests

```bash
# Let the router pick, using the BALANCED default policy
curl -X POST http://localhost:8100/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Refactor this loop for readability."}], "policy": "MAX_QUALITY"}'

# Force local-only (no cloud calls even if keys are configured)
curl -X POST http://localhost:8100/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Summarize this."}], "policy": "LOCAL_ONLY"}'

# Optimize for cost
curl -X POST http://localhost:8100/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Quick sanity check on this SQL."}], "policy": "LOW_COST"}'

# No policy given: a coding question auto-classifies as TaskType.CODING and
# routes using each model's "coding" capability score
curl -X POST http://localhost:8100/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "I keep getting a stack trace when I run this function — help me debug the bug."}]}'

# An email address in the query forces LOCAL_ONLY regardless of the
# BALANCED default policy — see privacy_level/routing_reason in the response
curl -X POST http://localhost:8100/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "My email is jane.doe@example.com, can you help me draft a reply?"}]}'
```

## Tests

```bash
pip install -r nexus/requirements.txt
pytest nexus/tests
```

`test_ollama_runtime.py`, `test_openai_provider.py`, and
`test_anthropic_provider.py` mock each provider's HTTP API directly
(`httpx.MockTransport`), so none need a running backend or real API key.
`test_router_policy_selection.py`, `test_router_failover.py`, and
`test_router_latency_scoring.py` exercise `ModelRouter` against fake
providers/registries/latency data. `test_cost_tracker.py` and
`test_latency_tracker.py` cover cost/latency computation and accumulation.
`test_task_classifier.py` and `test_privacy_classifier.py` are table-driven
over representative queries per `TaskType`/privacy tier — both classifiers
are pure functions, no fixtures needed. `test_chat_integration.py`,
`test_chat_with_cloud_provider.py`, and
`test_chat_classification_integration.py` exercise the FastAPI app
end-to-end with fake providers, and `test_memory_store.py` exercises the
SQLite memory store directly.

RAG: `test_chunking.py` and `test_reranker.py` are pure-function tests;
`test_parsers.py` exercises real `txt`/`md`/`csv`/`pdf`/`docx` parsing
against small in-memory fixtures (`pypdf`/`python-docx`, no real files on
disk); `test_vector_store.py` and `test_rag_service.py` use a fake
`EmbeddingProvider` so no real embedding model is needed;
`test_long_term_memory.py` covers the fact-CRUD store; and
`test_chat_rag_integration.py` exercises `/api/chat` end-to-end with a
fake provider that captures the messages it received, to prove the
synthetic RAG context message actually reaches the model.

Tools: `test_tool_registry.py`, `test_python_exec_tool.py`,
`test_files_tool.py` (including path-traversal rejection),
`test_database_tool.py` (including SQL-injection/chaining rejection), and
`test_web_search_tool.py` (mocked DuckDuckGo HTML response) test each
tool in isolation; `test_chat_tools_integration.py` exercises the full
tool-calling loop end-to-end with a fake provider, including the
max-iterations cap and the streaming single-chunk scope boundary.
`test_openai_provider.py`/`test_anthropic_provider.py` also cover
translating the canonical tool schema into each provider's native format
and parsing `tool_calls` back out of mocked responses.

Agents: `test_tool_loop.py` exercises the extracted `run_tool_loop()`
directly (termination, `max_iterations`, usage accumulation, and
`allowed_tools` rejection) — this is the refactor's regression check,
alongside every pre-existing `test_chat_tools_integration.py` case still
passing unmodified against the same extracted function.
`test_agent_runtime.py` covers `AgentRuntime.run()` against a fake router
+ fake provider (step trace, `completed`/`iterations_used`, the
per-request `max_iterations` override). `test_agent_permissions.py`
proves a disallowed tool is neither shown to the model nor executed even
when the model asks for it anyway. `test_agents_api.py` exercises
`GET /api/agents` and `POST /api/agents/{name}/run` end-to-end, including
the 404 for an unknown/disabled agent name.

Personal Intelligence: `test_personal_state.py` hand-computes the
recency-/source-weighted mean and confidence formula against a small
fixture with known signal ages, and covers `InvalidSignalError` for a bad
dimension/value/source. `test_baseline.py` proves the recent window is
excluded and that a dimension below `min_samples` is simply absent.
`test_trends.py` covers improving/declining/stable classification and the
`INVERTED_DIMENSIONS` sign flip (rising stress reads as "declining", not
"improving"). `test_weakness.py` covers deviation sign correction, the
`min_confidence`/`min_deviation` filters, priority ordering, and the
declining-trend priority bump. `test_personal_context.py` covers
`build_personal_context_message()` returning `None` with nothing recorded
yet vs. rendering state/weaknesses/profile when present.
`test_personal_api.py` exercises the full personal API end-to-end,
including a signal → state → baselines → weaknesses round trip and that
`DELETE` removes everything for one user without touching another.
`test_chat_personal_context.py` is the key privacy test: personal context
reaches the provider on a local route, and is withheld (with
`personal_context_used: false`) when the resolved route is a cloud
provider under the default `personal.local_only_context: true`.

Health: `test_health_analyzer.py` fires each of the six rules on a
hand-built fixture and stays silent when it shouldn't, and proves
`data_sufficiency` caps severity to `"watch"` on thin data even when the
raw deviation/confidence would otherwise score `"significant"`.
`test_health_safety.py` is **the critical one** — parametrized over
real-world-phrased red-flag inputs (chest pain, self-harm, stroke
symptoms, ...) proving they short-circuit to the urgent-care referral,
and over prohibited outputs (diagnostic claims, medication dosages, "no
need to see a doctor") proving they're blocked and rewritten with the
disclaimer attached; also covers that ordinary inputs/outputs pass
through untouched. `test_health_agent.py` runs `HealthAgent` through
`AgentRuntime` with a fake provider that returns a diagnostic claim and
proves the safety filter rewrites it, that safe output passes through
unmodified, and that routing is forced `LOCAL_ONLY`.

Generation: `test_brief_builder.py` hand-computes the exact
`target_intensity` value against a fixture of specific weaknesses (and
the clamp boundary), and proves `rationale` names every contributing
dimension and that zero signals yields the generic-plan default.
`test_personalized_generator.py` proves the brief's structured facts
reach the model, the response surfaces `brief_rationale`/
`addressed_weaknesses` verbatim from the brief, `personalized` reflects
whether the brief had signals, and the workout/recovery-plus-significant-
pattern safety gate fires (and that `study`/no-significant-pattern cases
are correctly NOT gated).

Sports: `test_sports_adapter.py` uses `httpx.MockTransport` against
realistic football-backend response shapes to prove
`value=None`/`confidence="low_upstream_confidence"` metrics land in
`unavailable` without being coerced to `0.0` (while `low_sample` with a
real value stays available), coverage math, the formation-404-is-not-an-
error case, and that a genuinely unreachable backend raises
`ProviderUnavailableError`. `test_sports_tactical.py` proves no
`TacticalFinding` is ever derived from an unavailable metric, threshold
boundaries, and player-metric team-wide averaging. `test_sports_coach.py`
proves `unavailable_metrics` reaches the prompt and that `narrative` is
the only LLM-generated field on `CoachReport`. `test_sports_ingest.py`
proves the 0-100→0..1 normalization and `source="inferred"` confidence
exactly, and that unavailable/unmapped metrics are skipped entirely.
`test_sports_agent.py` covers `SportsAgent.prepare_context()` injecting
findings and the unavailable list, and its graceful no-op without a
`match_id`/adapter. `test_group_c_apis.py` exercises the new health-intel/
generate/sports endpoints end-to-end against fakes.

Verification: `test_verification_checks.py` covers all five deterministic
checkers with a PASS/FAIL/INCONCLUSIVE case each.
`test_verification_scoring.py` is **the critical one** — proves
all-`INCONCLUSIVE` yields `score=None`/`band=UNVERIFIED` (never `HIGH`),
that `INCONCLUSIVE` checks are excluded from the scoring denominator
entirely (a heavily-weighted `INCONCLUSIVE` next to one `PASS` must not
drag the score down), and every band threshold boundary.
`test_fact_checker.py` (fake provider) proves "unsupported, no evidence
at all" → `INCONCLUSIVE` is distinct from "unsupported, evidence present"
→ `FAIL`. `test_multi_model_judge.py` proves the judge always picks a
model different from the one being judged, skips unhealthy candidates,
and returns `None` (not a pass) when no distinct model is available.
`test_verification_engine.py` proves escalation fires below
`escalate_below` and not above (including when score is `None`), that an
unavailable judge adds `INCONCLUSIVE` never a pass, and that
`extra_usage` accumulates across fact-checking and judging.
`test_verification_presenter.py` proves `UNVERIFIED`/`UNCERTAIN` visibly
list every uncertainty note while `HIGH` stays a single line.
`test_chat_verification.py` exercises `/api/chat` end-to-end: `verify:
true` attaches a report, verification cost lands as a distinct
`cost_records` row (not folded into the main entry),
`always_verify_task_types` forces it on without the flag, and streaming
puts the report only in the final `done` event.

Evaluation: `test_eval_harness.py` proves a run produces a well-formed,
reproducible `EvalRun` (including against a small custom fixture dataset
via `NEXUS_EVALUATION__DATASETS_DIR`) with no real network calls and no
leaked secrets in `config_snapshot`. `test_eval_suites.py` runs each of
the six suite evaluators against one known-good and one known-bad case.
`test_eval_store.py` proves the save/load round trip preserves every
scored field. `test_regression.py` proves `compare_runs()` catches
suite-level pass-rate drops, per-case regressions, and cost/latency
spikes, and specifically that a **safety** suite regression is reported
even when it would fall within the normal pass_rate tolerance.
`test_eval_cli.py` proves the CLI exits `1` on a safety failure or a
detected regression and `0` on a clean run, and that `--json` output
parses.

Training (Phase 13) — **none of these import torch, and none need a
GPU**: `test_interaction_logger.py` proves records are written when
enabled and that **nothing at all** is written when disabled (not a
blanked row — nothing), plus the feedback update path.
`test_dataset_builder.py` covers all four mining sources, and its
critical case proves a record whose **stored** `privacy_level` says
`public` is still excluded when the current classifier reads it as
private — privacy is re-derived at export, never trusted from the label.
`test_dataset_validation.py` covers malformed rows, empty assistant
turns, over-length examples, duplicate prompts, and class imbalance (and
validates the committed behavior templates against their own validator).
`test_training_config.py` covers the YAML round trip, seed preservation,
and the 12GB defaults. `test_vram_estimation.py` proves the estimate
scales correctly with batch, seq_len, model size, and gradient
checkpointing, flags a config exceeding `available_vram_gb`, and that the
README's remediation ladder actually brings an over-budget config back
under. `test_preflight.py` drives every check against a **mocked torch
namespace** and specifically asserts the sm_120/Blackwell check fails
loudly — naming the GPU, the installed build, its CUDA version, the
compiled arch list, and the exact reinstall command.
`test_train_dry_run.py` proves `--dry-run` works and imports **no** ML
package (asserted via `sys.modules`), that it says plainly it did not
probe the GPU, and covers checkpoint discovery and the
throughput/throttling monitor.

Advanced Intelligence (Phase 14): `test_capability_matrix.py` proves zero
samples yields **exactly** the static values, that learned weight rises
with sample size and is capped, and checks the blending math against
hand-computed fixtures (20 samples → 0.75; 100 samples → 0.865); it also
proves the `safety` suite can never become a capability score.
`test_router_learned_capabilities.py` proves rankings shift with learned
scores and are **identical** to today when the matrix is empty.
`test_graph_rag.py` covers extraction (including batching and
unparseable output), and proves expansion finds a chunk sharing **no**
query vocabulary, that hop/node caps are enforced, and that a disabled
graph means zero behavior change and zero provider calls.
`test_orchestrator.py` covers decomposition/delegation through the real
runtime and proves `DelegationGuard` enforces depth, total, and repeat
rejection — including that the cap holds against a model that keeps
asking. `test_autonomous_research.py` proves the loop terminates both on
coverage **and** on `max_rounds`, and that output separates
Known/Likely/Uncertain/Unknown. `test_self_eval.py` proves a critique is
produced, that it **never** overrides a verification verdict in either
direction, and that low scores are flagged for mining.
`test_forecast.py` proves a known-slope series projects correctly and
that thin or noisy data returns `insufficient_data` with a caveat and
**never a number**. `test_scenario_simulator.py` covers weakness
clearing, the inverted-dimension sign convention, and that the result
makes no causal claim. `test_phase14_eval_suites.py` runs the three new
suites and proves each scores known-good cases at 1.0 **and** fails
known-bad expectations.

## Configuration

Defaults live in `nexus/config/nexus.yaml`; the model registry (context
windows, capability scores, per-1k-token costs, and `supports_tool_calling`
flags used by routing) lives in `nexus/config/models.yaml`. Both are
loaded by `nexus/config/settings.py`, which layers, in priority order:
explicit constructor args → `NEXUS_*` env vars → `nexus.yaml` → field
defaults. `OLLAMA_BASE_URL`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and
`GITHUB_TOKEN` are special-cased overrides on top of that (not
`NEXUS_`-prefixed), since they're either shared with other tools or set
by `docker-compose.yml`. Cloud API keys and the GitHub token are never
written to `nexus.yaml` and never logged.

`rag:` and `tools:` were added in Phase 4/5, `agents:` and `personal:` in
Phase 6/7 — see [Documents + RAG](#documents--rag), [Tool
calling](#tool-calling), [Agents](#agents-phase-6), and [Personal
Intelligence](#personal-intelligence-phase-7) above.
`health:`, `generation:`, and `sports:` were added in Phase 8/9/10.
`verification:` and `evaluation:` are the two new top-level `nexus.yaml`
sections this phase adds:

```yaml
agents:
  enabled: [research, coding, data, planning, health, sports]
  max_iterations: 8
  default_policy: BALANCED

health:
  enabled: true
  min_sample_size: 3
  # NOTE: there is deliberately NO safety toggle — the health safety
  # layer is unconditional.

generation:
  enabled: true
  default_intensity: 0.7
  min_intensity: 0.2
  max_intensity: 1.0

sports:
  enabled: true
  backend_base_url: http://localhost:8000
  request_timeout_seconds: 30
  # football metric_name -> NEXUS sports.* dimension
  metric_dimension_map:
    decision_making_score: sports.decision_making
    off_ball_movement_score: sports.positioning
    press_resistance_score: sports.agility
    first_touch_score: sports.agility
    defensive_positioning_score: sports.positioning

verification:
  enabled: true
  # Off per-request by default (cost) — these task types always verify.
  always_verify_task_types: [health, research, mathematics]
  enable_fact_check: true
  enable_judge: true
  escalate_below: 0.65
  max_claims: 6

evaluation:
  datasets_dir: nexus/evaluation/datasets
  default_suites: [routing, classification, rag, tools, safety, verification,
                   graph_rag, orchestration, forecast]
  use_real_providers: false
  pass_rate_tolerance: 0.02
  cost_tolerance: 0.25
  latency_tolerance: 0.5

training:
  log_interactions: false
  interaction_retention_days: 180
  dataset_output_dir: nexus/training/data
  exclude_privacy_levels: [private]
  min_verification_score: 0.8
  available_vram_gb: 12.0        # RTX 5070 Ti Laptop

capability_learning:
  enabled: true
  min_samples: 20
  max_learned_weight: 0.7

self_eval:
  enabled: false
  log_low_scores: true
```

`TrainingSettings`, `CapabilityLearningSettings`, and `SelfEvalSettings`
(see [Custom NEXUS Model](#custom-nexus-model-phase-13) and [Advanced
Intelligence](#advanced-intelligence-phase-14) above): `log_interactions`
is off by default because it stores full prompts and responses;
`exclude_privacy_levels` is applied against a **re-classification at
export time**, not the stored label, so a record written before the
classifier tightened is still caught; `available_vram_gb` is what
`estimate_peak_vram_gb()` and preflight compare against.
`capability_learning.enabled: false` makes `ModelRouter` score straight
off `models.yaml`, exactly as it did before Phase 14.

`rag.graph`, `agents.orchestration`, `agents.research`, and
`personal.forecast` are documented in their own sections above. The three
caps (`max_hops`/`max_nodes`, `max_depth`/`max_total_delegations`,
`max_rounds`) are enforced in the runtime, so lowering them in config
genuinely constrains behavior rather than suggesting it.

See [Health Intelligence](#health-intelligence-phase-8), [Personalized
Generation](#personalized-generation-phase-9), and [Sports
Intelligence](#sports-intelligence-phase-10) above for the health/
generation/sports fields. `sports.backend_base_url` must point at a
running instance of this repo's own football backend
(`backend/api/main.py`, started separately — see the repo root, not
`nexus/` — via `uvicorn backend.api.main:app`); `nexus/sports/adapter.py`
never falls back to anything else.

`VerificationSettings` (see [Verification](#verification-phase-11)
above): `enabled` is the global kill switch checked before anything else
runs; `always_verify_task_types` forces verification on for a task type
even when a request omits `"verify": true`; `enable_fact_check` and
`enable_judge` toggle the two LLM-backed stages independently of each
other (deterministic checks always run regardless of either flag);
`escalate_below` is the score threshold under which the engine calls the
judge; `max_claims` bounds how many claims `FactChecker.extract_claims()`
will check per answer, capping worst-case LLM calls on a long response.

`EvaluationSettings` (see [Evaluation Platform](#evaluation-platform-phase-12)
above): `datasets_dir` and `default_suites` control what
`python -m nexus.evaluation.cli` runs with no `--suites` flag;
`use_real_providers` is the harness's own kill switch, separate from
the CLI's `--real-providers` flag, so a stray script that constructs
`EvalHarness()` directly still defaults to fakes; `pass_rate_tolerance`,
`cost_tolerance`, and `latency_tolerance` are the default regression
thresholds `compare_runs()` uses when the API/CLI don't override them —
none of them apply to the safety suite's pass rate, which is always
zero-tolerance.
