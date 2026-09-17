# AI Coach — Competition Presentation Package

Built from a full codebase audit on 2026-08-17 (repo: `sajadsoleimanilh8/AI-coach-strategy`, active checkout `SportsStrategyCoachAI`). Every claim below is traceable to real code. Where the codebase has scaffolding, empty folders, or unresolved bugs, they are labeled as such — nothing here is invented.

**Core message the whole deck reinforces:**

> Most football AI looks at the game. AI Coach looks at the player, the game, and the context — then uses an LLM to turn those signals into understandable decision support.

> Computer Vision = the eyes. Data = the memory. The LLM = the reasoning layer. The coach = the decision maker.

---

## 0. What's real vs. what's scaffolding (read this first)

| Claim you might be tempted to make | Reality |
|---|---|
| "AI predicts player psychology from video/biometrics" | No. It's a **13-item self-report questionnaire**, scored by a transparent formula. Say "AI-assisted structuring," not "AI prediction." |
| "A trained ML model scores readiness" | No. `ai/health_ai/`, `ai/llm_coach/`, `ai/recommendation_ai/` are **empty folders** (`.gitkeep` only). The real scorer is a documented, deterministic heuristic in `ai/performance_ai/match_readiness_predictor/`, and it says so explicitly in its own code comments. |
| "The LLM decides if a player should start" | No. There is no start/bench boolean anywhere in the code. The system outputs **risk bands** (low/moderate/high) plus an LLM narrative — framed as decision *support*, never a verdict. |
| "Our calibration accurately maps broadcast video to the pitch" | No. On real broadcast test clips, valid calibration rate is **0%**. The keypoint model has a diagnosed generalization gap (mirrored pitch-end output, likely a 255-image training-set ceiling). Present this honestly — it's a legitimate engineering finding, not a failure to hide. |
| "Simulation uses reinforcement learning" | No. `ai/simulation_ai/what_if_analysis/engine.py` is a **deterministic recompute engine** — it reruns the same scoring functions with one changed parameter. There is no simulation trainer (the empty `training/train_simulation.py` stub was deleted 2026-09-16). |
| "The LLM is a chatbot bolted on top" | No — this is the one to lean into. The LLM layer (NEXUS) is a genuinely separate, well-engineered service: multi-provider routing with automatic local fallback, an 8-agent framework, RAG, a second-model verification judge, and hardcoded safety filters that intercept medical red-flag language before it ever reaches the model. |

---

## A. 5–7 Minute Competition Presentation (full spoken script)

**[Opening — 30 sec]**

Before a football match begins, one of the most important questions isn't "What formation should we use?"

*(pause)*

It's "Are my players actually ready to play?"

Tactics matter. But tactics assume something that often goes unchecked — that the eleven players on the pitch are actually fit to execute them. A player can be tactically perfect and still be running on four hours of sleep, carrying muscle soreness, and quietly anxious about a big match. No formation fixes that.

AI Coach starts before the first whistle.

**[The problem — 40 sec]**

Most football AI systems point a camera at the pitch and tell you what happened — detection, tracking, statistics. That's valuable, but it's only half the picture. It tells you what happened. It doesn't tell you why, and it definitely doesn't tell you whether the player was set up to succeed in the first place.

We built AI Coach around a different question: what if the coaching intelligence started with the player, not just the footage?

**[Pre-match intelligence — 60 sec]**

So before kickoff, AI Coach asks the player two short questionnaires. One covers physical readiness — sleep, training load over the last 24 to 48 hours, fatigue, soreness, hydration, nutrition. The other covers psychological state — focus, pre-match stress, confidence, motivation, and how the player recovers from mistakes.

Those answers go through a transparent scoring engine — deterministic, explainable, no black box — that turns thirty-odd raw ratings into a structured player profile: physical readiness, fatigue score, recovery score, mental readiness, focus, stress, and two risk bands, performance risk and workload risk.

That structured profile is the input to the next layer. And this is where it gets interesting.

**[The LLM as reasoning engine — 90 sec, the core of the talk]**

We didn't want to just show a coach a wall of numbers. Numbers need interpretation, and interpretation is exactly what large language models are good at.

So we built a separate reasoning service — we call it NEXUS — that takes the structured player profile, plus match context, and produces a plain-language explanation a coach can actually use in ninety seconds between drills.

Here's the important design decision: the LLM is not allowed to invent or recalculate a single number. Every score you see was already computed deterministically upstream. The LLM's system prompt literally says — I'm paraphrasing the actual prompt in our code — "every number below has already been computed; never calculate, re-derive, adjust, or estimate any score; if you state a number, it must be one that appears verbatim above." The model's only job is to reason about what those numbers *mean* for this player, in this match, and explain it in coach-readable language.

That's the architecture: Computer Vision and questionnaires are the eyes and the intake. Deterministic scoring is the memory — consistent, auditable, reproducible. The LLM is the reasoning layer that turns that memory into something a human coach can act on in seconds.

And it's not one lightly-wrapped API call. Underneath, NEXUS routes across multiple providers — OpenAI, Anthropic, Gemini, or a fully local model — with automatic failover, so the system keeps working even if a cloud provider is down or a club wants to run entirely on local hardware for data privacy. There's a second-model verification step for higher-stakes answers, where a *different* model checks the first model's claims before they're trusted. And there's a hardcoded safety filter that scans for medical red flags — chest pain, breathing difficulty — and routes straight to an urgent-referral message, bypassing the LLM entirely, before any narrative is ever generated.

**[Computer Vision, positioned second — 60 sec]**

We also give the AI a second source of information: what's actually happening on the pitch. Five trained YOLO models detect players, the ball, the field, and goalposts. A tracking algorithm follows every player and the ball across the video. Team assignment groups players by jersey color automatically.

We'll be honest about where this stands: our pitch-calibration model — the piece that maps camera pixels to real pitch coordinates — works well on our training data, but doesn't yet generalize to real broadcast footage. We know exactly why: the training set is small enough that the model sometimes locks onto a mirrored, wrong-half-of-pitch template. That's an honest, diagnosed limitation, and it's the next thing on our list — not something we're pretending isn't there.

**[Bringing it together — 40 sec]**

So the full picture is: player questionnaires plus video analysis both feed structured, honestly-labeled data into one reasoning layer. That layer doesn't replace the coach's judgment — it gives the coach a faster, clearer starting point, with the reasoning shown, not hidden.

**[Closing — 30 sec]**

Football has always been about decisions — who starts, how hard to push a tired player, when to change shape. We can't make those decisions for a coach, and we don't want to. What we built is the layer underneath the decision: turning a player's own words, a scoring engine's numbers, and a video feed into one coherent, explained picture, before the whistle even blows.

AI Coach doesn't replace the coach. It gives the coach another intelligence layer.

---

## B. Slide-by-Slide Deck

**Slide 1 — Title**
*Visual:* Full-bleed pitch photo, dimmed, title "AI Coach" centered, subtitle "An AI-powered football decision intelligence system."
*UI element:* None (title card).
*Say:* "AI Coach starts before the first whistle."
*Technical point:* n/a — framing slide.
*Wow moment:* Withhold it — let the tagline land in silence for a beat.

**Slide 2 — The question nobody asks first**
*Visual:* Two stacked questions, second one highlighted: "What formation should we use?" (greyed) vs. "Are my players actually ready to play?" (bold).
*UI element:* None.
*Say:* "Before a match begins, most football AI answers the wrong first question."
*Technical point:* Sets up the pre-match framing.
*Wow moment:* The contrast itself.

**Slide 3 — Most football AI, one layer**
*Visual:* Simple linear diagram: Video → Detection → Statistics.
*UI element:* None.
*Say:* "This is what most football AI looks like: point a camera at the pitch, count things."
*Technical point:* Sets the comparison baseline honestly (this is a real, useful pattern — just incomplete).
*Wow moment:* Held for the next slide's contrast.

**Slide 4 — AI Coach, five layers**
*Visual:* Vertical stack: Player (Psychology + Health/Readiness) → Match Context → Video → AI Perception → Structured Data → **LLM Reasoning** (highlighted) → Coach Decision Support.
*UI element:* None.
*Say:* "AI Coach adds everything above the video — the player's own words, structured into something an AI can reason about."
*Technical point:* This is the architecture diagram judges will remember. Keep it visually dominant.
*Wow moment:* The visual density difference from Slide 3.

**Slide 5 — Pre-match: the player answers first**
*Visual:* Live app — [TabPreMatchHealth.jsx](../frontend/web/src/components/tabs/TabPreMatchHealth.jsx) questionnaire screen.
*UI element:* The real health questionnaire form (sleep, training load, fatigue, soreness, hydration).
*Say:* "Before anything else, AI Coach talks to the player. Sleep, training load, fatigue, soreness, hydration — eighteen fields, all self-reported, all real questions."
*Technical point:* This is a genuine Pydantic-validated form submitted to `POST /api/prematch_health/{player_id}/submit`.
*Wow moment:* It's a real, working form — click through it live.

**Slide 6 — Pre-match: psychology**
*Visual:* Live app — [TabPsychology.jsx](../frontend/web/src/components/tabs/TabPsychology.jsx).
*UI element:* The 13-item psychology questionnaire (focus, stress, confidence, motivation, pressure response).
*Say:* "And separately, how the player is feeling — focus, pressure, confidence, and how they bounce back from mistakes."
*Technical point:* 13-item scale-based form, `POST /api/psychology/{player_id}/submit`.
*Wow moment:* Naming the actual dimensions out loud — it sounds like a real intake, because it is one.

**Slide 7 — From answers to structured data**
*Visual:* Pipeline diagram: Player Answers → Health Signals + Psychological Signals + Match Context → Structured Player Profile.
*UI element:* Results panel of either tab, showing computed scores (physical_readiness, fatigue_score, mental_readiness, focus, stress, risk bands).
*Say:* "Those answers become a structured profile — readiness, fatigue, recovery, mental readiness, focus, stress, and two risk bands."
*Technical point:* Say plainly: this scoring is a **transparent deterministic formula**, not a black-box model — every response is tagged `"heuristic_proxy"` so nothing pretends to be more than it is.
*Wow moment:* The honesty itself is the wow moment for a technical judge — most teams overclaim here, you won't.

**Slide 8 — The LLM is not a chatbot bolted on**
*Visual:* Big text: "Football Reasoning Engine," with a small chat-bubble icon crossed out beside it.
*UI element:* None.
*Say:* "Here's the shift in how we think about the LLM. It's not a chatbot feature. It's a reasoning engine that sits on top of everything else."
*Technical point:* Sets up the architecture deep dive.
*Wow moment:* The reframe.

**Slide 9 — The LLM pipeline**
*Visual:* Full pipeline diagram (see Section D for exact stages) — Player Answers → Signals + Match Context → Structured Profile → **LLM Reasoning** → Risk/Readiness Assessment → Recommendation → Explanation for Coach.
*UI element:* None, or a code excerpt of the system prompt (see Slide 10).
*Say:* Walk each arrow — "structured data goes in, a single constrained LLM call comes out, and what comes out is explanation, not new numbers."
*Technical point:* Every number in the final report was computed *before* the LLM ever saw it.
*Wow moment:* "The LLM is contractually forbidden from doing math" — memorable line, literally true.

**Slide 10 — The actual guardrail, verbatim**
*Visual:* A real, redacted excerpt from `nexus/sports/coach.py`'s `_PREMATCH_SYSTEM_PROMPT`: *"Every number below has ALREADY been computed deterministically... Never calculate, re-derive, adjust, average, or estimate any score. Use the exact values given; if you state a number, it must be one that appears verbatim below."*
*UI element:* None — code/prompt excerpt as the visual.
*Say:* "This isn't a policy we describe in a slide deck. It's a line in the actual system prompt, in the actual code."
*Technical point:* Directly answers "how do you prevent hallucination" before a judge even asks.
*Wow moment:* Showing real code on a competition slide — instant credibility.

**Slide 11 — Live result: the narrated report**
*Visual:* Live app — [TabCoachChat.jsx](../frontend/web/src/components/tabs/TabCoachChat.jsx), the "Computed" card and "Generated — NEXUS Narrative" card side by side.
*UI element:* Both report cards.
*Say:* "The UI itself enforces the same honesty — one card is computed fact, one card is the LLM's explanation, and they're never merged."
*Technical point:* This UI/UX decision is a guardrail in itself, not just backend prompt engineering.
*Wow moment:* Point at the two labeled cards — judges visually register "these people separated fact from generated text on purpose."

**Slide 12 — Why an LLM, not traditional ML, for this step**
*Visual:* Two-row comparison: Traditional ML: Input → Prediction. Our layer: Input + Context → Reasoning → Recommendation + Explanation.
*UI element:* None.
*Say:* "Traditional ML is excellent at structured numeric prediction. But a coach doesn't want a number — they want to know what it means, right now, for this player, in this match. That's contextual reasoning and natural-language explanation, which is what LLMs are actually good at."
*Technical point:* Explicitly do not claim LLM > ML universally — frame as fit-for-purpose.
*Wow moment:* The precision of the claim — judges respect a team that doesn't overclaim.

**Slide 13 — Under the hood: multi-provider, multi-agent**
*Visual:* Small architecture diagram: ModelRouter → {OpenAI, Anthropic, Gemini, Local Ollama} with a "guaranteed local fallback" arrow; below it, 8 labeled agent boxes (Sports, Health, Research, Orchestrator, etc.).
*UI element:* None.
*Say:* "This runs on a real routing layer — it can use a cloud model for quality, or fall back to a fully local model automatically if no API key is configured, or a club wants everything on-premise."
*Technical point:* Mention the second-model verification "judge" and the hardcoded medical safety filter that intercepts red-flag language before the LLM ever sees it.
*Wow moment:* "A different model checks the first model's claims" — a real, working cross-verification step.

**Slide 14 — Now, the pitch: Computer Vision**
*Visual:* Live app — [TabMatchAnalysis.jsx](../frontend/web/src/components/tabs/TabMatchAnalysis.jsx), video with tracking overlay.
*UI element:* Annotated video playback, tracking boxes, team colors.
*Say:* "Now we give the AI a second source of information: what's actually happening on the pitch."
*Technical point:* Five trained YOLOv8 models (player, ball, field, goalpost, calibration keypoints), ByteTrack tracking, jersey-color team assignment.
*Wow moment:* Real detection boxes moving on real footage.

**Slide 15 — Honest engineering: calibration**
*Visual:* Before/after: a keypoint overlay on a training image (clean) next to a broadcast frame where calibration fails to lock.
*UI element:* [TabCalibration.jsx](../frontend/web/src/components/tabs/TabCalibration.jsx) debug view.
*Say:* "We also built a calibration layer to move from image coordinates toward pitch-aware analysis. It works well on our training data. On real broadcast footage, it doesn't generalize yet — and we know exactly why: on a limited training set, the model sometimes locks onto a mirrored, wrong-half-of-pitch template. That's the next problem we're solving, not something we're hiding."
*Technical point:* This is a genuine, diagnosed root cause — say it with confidence, not apology.
*Wow moment:* Technical honesty, stated plainly, is itself the wow moment for a judging panel that has seen ten teams overclaim CV accuracy.

**Slide 16 — Tactical intelligence, feeding back into reasoning**
*Visual:* [TabTeamIntelligence.jsx](../frontend/web/src/components/tabs/TabTeamIntelligence.jsx) — formation/shape output where available.
*UI element:* Team shape / formation panel.
*Say:* "Where calibration succeeds, tactical data — formation, team shape — becomes another input the reasoning layer can draw on."
*Technical point:* Every metric is tagged with a method (ml_trained / deterministic / heuristic_proxy) and a confidence level — an "honesty contract" enforced across the whole backend.
*Wow moment:* The honesty-contract concept — most teams don't have a systematic way to say "trust this number less."

**Slide 17 — The comparison slide**
*Visual:* Side-by-side: "Traditional Football AI" (Video → Detection → Statistics) vs. "AI Coach" (Player + Psychology + Health/Readiness + Match Context + Video → AI Perception → Structured Data → LLM Reasoning → Coach Decision Support).
*UI element:* None.
*Say:* "This is the difference in one picture."
*Technical point:* The strongest single slide in the deck — let it breathe, minimal narration.
*Wow moment:* Visual contrast alone.

**Slide 18 — Honest limitations**
*Visual:* Short bulleted list, plain styling (not hidden in small print).
*UI element:* None.
*Say:* "A competition prototype doesn't need to pretend it's finished. It needs a strong architecture and a clear path forward: calibration needs to generalize to broadcast footage, the LLM's reasoning needs broader validation against real coach judgment, and both questionnaire scorers need real outcome data before they can graduate from heuristic to trained models."
*Technical point:* Names the exact three gaps found in the audit — calibration generalization, LLM output validation, missing outcome labels for training.
*Wow moment:* Judges relax — this team knows their own system.

**Slide 19 — What's next: multi-agent football coach**
*Visual:* Central "Coach Agent" node with spokes to Tactical Analysis, Player Psychology, Physical Readiness, Scouting, Opponent Analysis.
*UI element:* None.
*Say:* "We already have a working multi-agent framework underneath NEXUS. The next step is specializing it fully for football — a tactical agent, a readiness agent, a scouting agent, coordinated by one coach-facing agent."
*Technical point:* Be clear this is *future*, built on infrastructure that already exists and is tested today — not vaporware from zero.
*Wow moment:* "We're not starting from scratch — the orchestration layer already runs."

**Slide 20 — Closing**
*Visual:* Return to the Slide 1 pitch photo, tagline now completed: "AI Coach doesn't replace the coach. It gives the coach another intelligence layer."
*UI element:* None.
*Say:* Deliver the closing line slowly, then stop talking.
*Technical point:* n/a.
*Wow moment:* Silence after the last line.

---

## C. Full Live Demo Script

**Setup beforehand:** Have the backend (`:8000`), NEXUS (`:8100`), and frontend dev server running and warmed up (make one throwaway LLM call before judges arrive so the first real call isn't slow). Pick one player with a *pre-loaded but not yet submitted* questionnaire state so you're not typing 30 fields live — or pre-fill and just click "Submit" if judges are time-boxed.

**Step 1 — Open the Pre-Match Assessment tab (Psychology).**
Say: *"Before the match even starts, AI Coach talks to the player."*
Show the 13-item form. Point at two or three specific fields (e.g., "pre-match stress," "mistake recovery speed") so judges see these are real, considered questions, not filler.

**Step 2 — Answer the psychology questions.**
Fill (or reveal pre-filled) sliders showing a *realistic mixed profile* — not all-good, not all-bad. E.g., low confidence, moderate stress, good focus. A messy, real-looking profile sells the demo better than a clean one.

**Step 3 — Switch to the Pre-Match Health tab.**
Fill in a believable case: poor sleep (5.5h), moderate fatigue, some muscle soreness, low perceived readiness.

**Step 4 — Submit both.**
Pause. Say: *"Now the AI has to reason over all of these signals together."*
Let the loading state actually show — don't skip past it, a beat of real latency makes it feel real.

**Step 5 — Show the LLM result in Coach Chat.**
Navigate to the Coach Chat tab, open the narrated psychology/readiness report.
Point at the two cards explicitly: *"This card — Computed — is the deterministic scoring. This card — Generated — is what NEXUS wrote after reasoning over it. They're kept visually separate on purpose."*
Read one or two sentences of the actual narrative aloud.
Say: *"The important part isn't just the recommendation. It's that the coach can see the reasoning behind it, and can tell exactly which parts are fact versus generated explanation."*

**Step 6 — Move to the football video.**
Open Match Analysis, play a short pre-processed clip with the tracking overlay already rendered (don't process live on stage — latency is currently ~40s per job on GPU, longer on CPU; render ahead of time).
Say: *"Now we give the AI another source of information: what is actually happening on the pitch."*
Point out player boxes, team colors, the events timeline.

**Step 7 — Show the tactical/calibration view briefly.**
Open Team Intelligence or Calibration debug tab. If showing calibration, say the honest line from Slide 15 rather than avoiding the tab — a judge who later asks "how good is your calibration" should already have heard your own honest answer.

**Step 8 — Close.**
Say: *"We started with a player questionnaire. We ended with an AI-assisted football intelligence system."*
Stop. Don't add anything after this line.

**Fallback plan:** If NEXUS or the CV pipeline is unavailable live, have a screen-recorded backup of Steps 4–7 ready. Never debug live in front of judges — narrate over the recording instead ("this is the same flow, captured yesterday").

---

## D. LLM Technical Deep Dive

**Architecture.** The LLM layer is not embedded in the main backend — it's a separate FastAPI service, "NEXUS," running on port 8100, talking to the football backend (port 8000) over plain HTTP. This is a real architectural boundary: `frontend/web/src/api/nexus.js` is a dedicated client, and its own comments say NEXUS is optional — only the Coach Chat tab depends on it, so the rest of the app works even if NEXUS is down.

**Provider layer.** Four concrete providers implement one `AIProvider` interface (`nexus/core/providers.py`):
- OpenAI (`gpt-4o-mini` default, `gpt-4o` available) via `POST /chat/completions`
- Anthropic (`claude-sonnet-4-5`) via `POST /v1/messages`
- Gemini (`gemini-2.5-flash` default, `-pro`, `-flash-lite`) via Google's `generateContent`
- Local Ollama (`mistral:7b`) via `POST /api/chat` against a local daemon — the only provider guaranteed to work with zero API keys configured, and the system's hard-coded final fallback.

A `ModelRouter` scores available/healthy providers against a routing policy (`BALANCED`, `MAX_QUALITY`, `LOW_COST`, `LOW_LATENCY`, `LOCAL_ONLY`) and fails over down the ranked list, guaranteeing the local model as a last resort. Which provider is *actually* live in a given deployment depends on which API keys are set — the repo itself ships with none, so out of the box the system runs entirely local.

**Prompt construction.** Prompts are short, hardcoded Python string literals, assembled by plain f-string concatenation — not a template engine, not loaded from the long-form architecture docs (those `docs/NEXUS_*.md` files are design vision documents, never read by any code path; worth knowing so you don't accidentally claim them as live prompts). The real prompts live in `nexus/agents/*.py` and `nexus/sports/coach.py`. Three purpose-built system prompts drive the football-facing outputs:
- Tactical match report prompt
- Pre-match readiness prompt — explicitly forbids recalculating any score
- Psychology prompt — additionally forbids clinical/diagnostic language ("do not use clinical language — anxiety, depression, disorder — and do not speculate about the player's emotional state or mental health")

**Context construction.** Everything injected into the prompt is computed deterministically first: tactical findings are derived by fixed thresholds over CV-pipeline metrics; readiness/psychology context is rendered from the questionnaire scorer's output via `as_prompt_context()` / `build_prompt_context()` methods that turn structured fields into text blocks. RAG-retrieved document chunks and prior conversation history can also be injected as additional context messages when relevant.

**Inputs.** Player context (health + psychology + readiness, all self-reported and deterministically scored), match context (derived tactical findings when calibration succeeds), and — for the general chat agent — retrieved documents and conversation history.

**Outputs.** Overwhelmingly free-text narrative. In the football-facing responses, exactly one field is LLM-generated (`narrative`); every numeric field in the same response object was computed upstream and passed through unchanged — this is stated explicitly in code comments ("`narrative` is the only LLM-generated field"). Separately, the internal verification subsystem *does* require strict JSON output for claim-extraction and judge verdicts, enforced by instruction plus a defensive parser with a regex fallback — not by an API-level structured-output/JSON-mode contract.

**Reasoning workflow / structured outputs.** True function/tool calling is wired for OpenAI, Anthropic, and Gemini (native tool-calling protocols), used by the general-purpose chat and agent framework for tools like web search, Python execution, file access, and database queries. Ollama doesn't support tool calling, so it silently runs without tools when selected. The football-specific "coach report" calls (readiness/psychology/tactical) are simpler, single-shot calls — not tool-using agents — by design, since their job is narrow narration, not open-ended research.

**Multi-agent framework.** Eight registered agents (Research, Coding, Data Analysis, Planning, Health, Sports, Orchestrator, Autonomous Research) share one `AgentRuntime`, the same tool-calling loop, and the same router as the primary chat path — deliberately built on shared infrastructure rather than a parallel one-off system. The Orchestrator can delegate sub-goals to specialist agents and synthesize results, with hard depth/count caps enforced in code (described in the codebase's own comment as "a wall it cannot get past," as opposed to a prompt instruction a model could ignore).

**Guardrails.**
- A hardcoded health-safety filter scans player-facing input for medical red-flag language (chest pain, breathing difficulty, fainting, self-harm, uncontrolled bleeding, stroke symptoms) and routes straight to an urgent-referral message, bypassing the LLM entirely if triggered.
- The same filter scans LLM *output* for diagnostic claims, medication/dosage advice, or "no need to see a doctor" language, and rewrites to a safe fallback plus a medical disclaimer if triggered.
- A verification engine runs deterministic checks always, optional LLM fact-checking against retrieved evidence when enabled, and — for low-confidence or unverified answers — escalates to a **second, different model acting as judge**. If no distinct healthy model is available, the system marks the result inconclusive rather than faking agreement.
- Context-window validation happens before every provider call, raising a clear error rather than silently truncating.
- A privacy gate can force fully local routing when input matches private-data patterns, and personal context is withheld from non-local providers by default.

**Limitations, stated honestly.** No API-level structured-output enforcement on the football narration calls (correctness of the "never recalculate" rule depends on the model following the instruction, not a hard schema constraint). No systematic evaluation harness yet comparing LLM narratives against real coach judgments — the verification layer checks internal consistency and factual grounding, not football-domain correctness. The psychology and readiness *scorers* feeding the LLM are heuristic formulas, not trained models, because no outcome-label data exists yet to train against.

---

## E. AI Stack

| Technology | Role | Status |
|---|---|---|
| LLM (OpenAI / Anthropic / Gemini / local Ollama, policy-routed) | Contextual reasoning, coach-facing narrative generation | Implemented, wired end-to-end |
| Multi-agent framework (8 agents + Orchestrator) | Task decomposition, specialist reasoning, tool use | Implemented, shares infra with main chat path |
| Prompt engineering (task-specific system prompts, "never recalculate" constraint) | Controls what the LLM is and isn't allowed to do with pre-computed data | Implemented |
| RAG (vector store + chunking + reranking) | Grounding chat/research answers in retrieved documents | Implemented, wired into chat + Research agent |
| Verification engine + second-model judge | Reduces hallucination risk on higher-stakes answers | Implemented |
| Health safety filter (regex red-flag input/output scanning) | Prevents medical/diagnostic overreach | Implemented, mandatory, no config flag to disable |
| Deterministic heuristic scoring (psychology, readiness) | Turns questionnaire answers into structured, explainable numeric profiles | Implemented, explicitly labeled non-ML |
| Computer Vision — YOLOv8 (player, ball, field, goalpost, pitch-keypoint models) | Visual perception of the pitch | Implemented, trained, real metrics |
| Tracking — ByteTrack | Temporal player/ball identity across frames | Implemented, production path |
| Team assignment — HSV + k-means jersey clustering | Splits players into two teams without needing calibration | Implemented |
| Pitch calibration — 32-keypoint model + RANSAC homography | Maps camera pixels to pitch coordinates | Implemented, honest reprojection-error math, but **does not yet generalize to broadcast footage** |
| Event heuristics (pass/shot detection) | Detects match events from tracked positions | Implemented; correctly returns zero events when calibration hasn't validated |
| Deterministic what-if simulation engine | Recomputes scoring functions under one changed parameter | Implemented (not reinforcement learning, despite the folder name) |
| Data layer — Postgres + SQLAlchemy, "honesty contract" (method + confidence tags on every metric) | Structures and audits every number the system produces | Implemented |
| Frontend — React 19 + Vite, hand-built UI | Coach-facing dashboard across 8 tabs | Implemented, tested with Playwright |

---

## F. 15 Difficult Judge Questions — Strong, Honest Answers

**1. Why use an LLM instead of traditional ML?**
Traditional ML is the right tool when you have clean numeric features and a single well-defined prediction target. Our readiness and psychology scoring is exactly that, and we do it deterministically — no LLM involved. Where ML falls short is turning several heterogeneous signals — readiness numbers, psychology numbers, match context — into an explanation a coach can read and trust in seconds. That's contextual reasoning and natural-language synthesis, which is what LLMs are built for. We use each tool where it's actually strong, not where it's fashionable.

**2. Why not traditional ML for the readiness/psychology scores themselves?**
Because we don't have outcome labels yet — no dataset linking a questionnaire answer to an actual match outcome. Our training script for this literally refuses to run and says so in its own comments. Rather than fabricate a black-box model on no ground truth, we shipped an honest, auditable heuristic formula, labeled as such in every API response, with the data pipeline already in place to train a real model the moment outcome data exists.

**3. How do you validate LLM outputs?**
Three layers: the system prompt hard-constrains the model to only use numbers it was given, never to compute new ones; a verification engine runs deterministic checks and, for higher-stakes answers, has a *second, different model* independently judge the first model's output rather than trusting a single model's self-report; and the UI itself visually separates computed fact from generated narrative so a coach is never confused about which is which.

**4. How do you prevent hallucinations?**
Structurally, not just by asking nicely: the football narration prompts explicitly forbid recalculating or inventing numbers, and we've verified that constraint in code, not just in the prompt text. For open-ended chat, we have RAG grounding and the second-model verification/judge step. It's not perfect — we don't have API-level JSON-schema enforcement on every call — and we say that openly as a known gap.

**5. Why YOLO for detection?**
It's the right accuracy/speed tradeoff for real-time-adjacent sports video analysis, and it's what we could actually train well on our available datasets. Our player, field, and goalpost detectors hit strong validation metrics (player mAP50 0.982, field 0.995, goalpost 0.986). Our ball detector is lower (mAP50 0.916) because the dataset itself is under-labeled — under 40% of images have ball annotations — which we've documented, not hidden.

**6. How accurate is detection, really?**
Strong on curated validation data; we're honest that broadcast-footage generalization is uneven across detectors — player/goalpost detection holds up reasonably, calibration does not yet. We'd rather give you the real number than a marketing number.

**7. How does tracking work?**
ByteTrack, run through the standard Ultralytics tracking interface, assigning persistent IDs to players and the ball across frames. We also have a second, independently-tested hand-rolled ByteTrack implementation kept for offline re-tracking of merged detections — it's real, tested code, just not in the live production path today.

**8. How does calibration work, and why is it unreliable?**
A 32-keypoint pitch-landmark model predicts known pitch reference points per frame; we fit a homography via RANSAC from those points to real pitch coordinates, with honest all-point reprojection-error reporting, not inlier-only cherry-picking. The homography math itself is solid — median reprojection error under 30cm on curated points. The problem is upstream: on real broadcast footage, the keypoint model sometimes locks onto a mirrored, wrong-half-of-pitch template. We believe this is a training-set-size ceiling — about 255 labeled images — not an algorithm bug, and it's our top engineering priority next.

**9. What is the role of Data Science here?**
Structuring heterogeneous inputs — self-report answers, video-derived metrics — into consistent schemas with explicit method and confidence tagging on every value, so downstream consumers (the LLM included) know exactly how much to trust each number. That auditability is a data science decision as much as a modeling one.

**10. What makes this innovative?**
Two things most football-AI projects don't do: starting the intelligence pipeline before kickoff with the player's own input, not just the footage; and constraining the LLM to be a narrator of pre-computed truth rather than a free-floating predictor — which is a much safer and more defensible use of an LLM in a domain where a wrong number matters.

**11. Is this medical advice?**
No, explicitly not, and the system is built to actively prevent that framing — a hardcoded filter intercepts medical red-flag language before the LLM sees it, and blocks diagnostic or medication-related claims in the LLM's output, replacing them with a safe fallback and a disclaimer. The readiness score is decision *support* — informing training-load and selection conversations — never a clinical judgment.

**12. How scalable is it?**
The LLM layer is provider-agnostic with automatic failover and a local-only mode, so a club with strict data policies can run entirely on-premise. The CV pipeline is the current bottleneck — GPU inference for a single job runs tens of seconds today — that's a known, measured number, not a guess, and it's the next thing we'd optimize for multi-match, multi-club use.

**13. How would a football club actually use it?**
Two touchpoints: a short pre-match check-in players fill out on a phone or tablet, and a coach dashboard that turns those answers plus available video analysis into a same-day briefing before team selection — not a replacement for medical staff or the coach's own judgment, but a faster, more consistent first read.

**14. What happens when the LLM is wrong?**
Because the LLM only narrates pre-computed numbers, a "wrong" LLM output is almost always a *misinterpretation* of correct data, not a fabricated number — which is a much easier failure mode to catch and correct than a hallucinated statistic. The verification/judge layer is our first line of defense against that; a human coach reading the explanation, with the underlying numbers shown right next to it, is the second.

**15. What would you improve next?**
In priority order: fix pitch-calibration generalization to broadcast footage (root cause already diagnosed), collect real outcome data so the readiness/psychology scorers can graduate from heuristic to trained models, and build a systematic evaluation harness that scores LLM narratives against real coach judgments, not just internal consistency.

---

## G. 60-Second Elevator Pitch

Most football AI points a camera at the pitch and tells you what happened. AI Coach starts earlier than that — before kickoff, it asks the player two short questions: how ready is your body, and how ready is your mind? Those answers get turned into a structured, transparent readiness profile — no black box. Then a reasoning layer, built on a real multi-provider LLM system with safety filters and cross-model verification, takes that profile plus whatever the video pipeline can tell us about the match, and turns it into a plain-language briefing a coach can read in under a minute — with the numbers and the AI's reasoning shown side by side, never blended. We also built a full computer-vision pipeline — player and ball detection, tracking, pitch calibration — and we're upfront about where it's strong and where it still needs work, because a system that hides its weak points isn't one a coaching staff should trust. AI Coach doesn't replace the coach's judgment. It gives the coach another intelligence layer, starting before the first whistle.

---

## Appendix

### 15 Show-Off Lines

1. "The pitch is only half of the story. The player is the other half."
2. "AI Coach doesn't just analyze the match. It starts before the match."
3. "Computer Vision gives us eyes. The LLM gives us reasoning."
4. "We don't want AI to replace the coach. We want AI to give the coach a better picture."
5. "Raw data tells you what happened. Intelligence helps you understand what it means."
6. "From player psychology to pitch intelligence — one AI coaching layer."
7. "The LLM in our system is contractually forbidden from doing math — it explains numbers, it doesn't invent them."
8. "One card is computed. One card is generated. We never let them blur together."
9. "A different model checks the first model's work — because trusting an AI's self-report is how hallucinations slip through."
10. "We didn't hide our calibration problem. We diagnosed it."
11. "Most teams show you their best demo. We're showing you our real one, weaknesses included."
12. "Football has always been about decisions. We're not replacing the decision-maker — we're upgrading what they see before they decide."
13. "The questionnaire isn't a gimmick. It's the first sensor in the pipeline."
14. "Every number in our system knows how much to trust itself — we call it the honesty contract."
15. "Before the whistle blows, AI Coach has already started working."

### Why This Is Different (comparison slide source text)

**Traditional Football AI:** Video → Detection → Statistics

**AI Coach:** Player → Psychology + Health/Readiness + Match Context + Video → AI Perception → Structured Data → LLM Reasoning → Coach Decision Support

### Honest Engineering — Limitations

- Pitch calibration does not yet generalize to real broadcast footage (0% valid on tested clips); root cause diagnosed as a likely training-set-size ceiling causing mirrored-template predictions.
- Psychology and readiness scores come from deterministic heuristic formulas, not trained ML — because no outcome-label dataset exists yet to train against. The training scripts for both openly state this rather than faking a model.
- LLM narrative quality is validated by internal consistency and fact-checking against provided data, not yet by a systematic comparison against real coach judgments.
- No authentication layer exists yet — it's explicitly scoped as future work, not an oversight papered over.
- Real-time updates use HTTP polling, not WebSockets — fine at prototype scale, worth revisiting for a live-match product.

### Future Direction

- **Multi-agent football coach**: specialize the existing (already-working) agent framework into dedicated Tactical, Psychology, Readiness, Scouting, and Opponent-Analysis agents, coordinated by a central Coach Agent — built on infrastructure that already runs today, not a from-scratch effort.
- Fix broadcast-footage calibration generalization (expand/rebalance the training set; the root cause is already diagnosed).
- Collect real match-outcome data to graduate the heuristic scorers to trained models.
- A systematic evaluation harness scoring LLM narratives against real coach judgments.
- Longer-term: personalized player recommendations, automatic match reports, opponent-specific tactical prep, a voice-based coach assistant, and a multimodal LLM combining video, text, and structured data directly.

### Final Message — 5 Alternatives (pick the strongest for your delivery style)

1. "AI Coach doesn't replace the coach. It gives the coach another intelligence layer." *(recommended — matches the deck's tone throughout)*
2. "Football has always been about decisions. Now those decisions can start with more information than the pitch alone ever gave you."
3. "We didn't build an AI that watches football. We built one that listens first, then watches."
4. "The whistle hasn't blown yet — and AI Coach is already working."
5. "Every layer in this system — the questionnaire, the score, the model, the explanation — exists to answer one question honestly: is this player ready? We think that's worth building first."
