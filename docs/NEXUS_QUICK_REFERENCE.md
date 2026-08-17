# NEXUS Quick Reference Guide
## For Developers & Implementers

---

## Core Components at a Glance

### 1. Request → Understanding Pipeline

```
Raw Request
    ↓
Parser (extract intent, entities, attachments)
    ↓
Task Classifier (determine type: conversation, coding, etc.)
    ↓
Context Aggregator (retrieve memory, docs, state)
    ↓
Privacy Assessment (can this be cloud-processed?)
    ↓
Intelligence Router (which model/engine to use?)
    ↓
Execution Plan (tools, approaches, fallbacks)
```

### 2. Model Router Decision Factors

```
Task Type (conversation/coding/research/etc.)
    +
Context Length (how much memory needed?)
    +
Privacy Level (local-only/hybrid/cloud-ok)
    +
Latency Budget (instant/seconds/minutes)
    +
Cost Preference (free/balanced/premium)
    +
Quality Target (good-enough/better/best)
    +
Tool Requirements (what can solve this?)
    +
User Preferences (remembered settings)
    =
BEST MODEL SELECTION
```

### 3. Execution Paths

| Path | Use Case | Examples |
|------|----------|----------|
| **Local Only** | Sensitive data, simple tasks | Chat, brainstorm, summarize |
| **Cloud Primary** | Complex reasoning, quality | Research, coding, analysis |
| **Hybrid** | Sensitive + complex | Private docs + advanced analysis |
| **Multi-Model** | High-stakes, complex | Health decisions, research synthesis |
| **Tool-Augmented** | External info needed | Current events, real-time data |

### 4. Tool Categories

```
Information:       Web Search, Wikipedia, APIs
Computation:       Python, Calculator, Database
Files:             Read, Write, Directory, Archive
Code:              GitHub, Linter, Debugger
Vision:            Image Analysis, OCR, Detection
Time:              Calendar, Reminders, Scheduling
```

---

## Memory Layers

### Short-Term (Session)
```python
{
    "conversation_id": "...",
    "messages": [
        {"role": "user", "content": "...", "timestamp": ...},
        {"role": "assistant", "content": "...", "timestamp": ...}
    ],
    "active_task": {...},
    "temporary_state": {...},
    "ttl": 3600  # 1 hour
}
```

### Long-Term (Persistent)
```python
{
    "user_id": "...",
    "preferences": {
        "response_style": "concise",
        "model_preference": "claude",
        "privacy_level": "balanced",
        "tools_enabled": [...]
    },
    "projects": {
        "project_id": {
            "name": "...",
            "context": "...",
            "files": [...],
            "members": [...]
        }
    },
    "expertise": {
        "python": 0.85,
        "ai": 0.72,
        "devops": 0.65
    },
    "goals": [
        {"name": "...", "status": "...", "priority": ...}
    ]
}
```

### Personal State
```python
{
    "physical": {
        "energy": 0.72,
        "recovery": 0.61,
        "activity": 0.85
    },
    "mental": {
        "focus": 0.74,
        "stress": 0.45,
        "mood": 0.68
    },
    "timestamp": "2024-08-08T14:30:00Z"
}
```

---

## Implementation Checklist

### Phase 0: Foundation
- [ ] Repository structure
- [ ] Configuration system
- [ ] Logging framework
- [ ] Database schema
- [ ] FastAPI skeleton
- [ ] Base interfaces

### Phase 1: Local LLM
- [ ] Ollama integration
- [ ] Model manager
- [ ] Streaming responses
- [ ] Conversation storage
- [ ] Basic chat API

### Phase 2: Cloud Integration
- [ ] Provider abstraction
- [ ] OpenAI integration
- [ ] Anthropic integration
- [ ] Model registry
- [ ] Health checks

### Phase 3: Intelligent Routing
- [ ] Task classifier
- [ ] Model scorer
- [ ] Routing policies
- [ ] Cost tracking
- [ ] Performance monitoring

### Phase 4: Memory & RAG
- [ ] Vector database
- [ ] Embeddings
- [ ] Document ingestion
- [ ] Retrieval pipeline
- [ ] Reranking

### Phase 5: Tools
- [ ] Tool registry
- [ ] Web search
- [ ] Python execution
- [ ] File operations
- [ ] GitHub integration

### Phase 6: Agents
- [ ] Agent runtime
- [ ] Planning module
- [ ] Research agent
- [ ] Coding agent
- [ ] Data agent

### Phase 7+: Advanced Features
- [ ] Personal intelligence
- [ ] Health analysis
- [ ] Sports analytics
- [ ] Verification layer
- [ ] Evaluation framework

---

## Configuration Templates

### Minimal Setup (Development)

```yaml
# config/dev.yaml
local:
  backend: ollama
  models:
    general: mistral:7b
    embeddings: nomic-embed-text

cloud: {}  # Optional

memory:
  backend: sqlite
  path: ./data/nexus.db

agents:
  enabled: []
```

### Production Setup

```yaml
# config/prod.yaml
local:
  backend: ollama
  models:
    general: mistral:7b
    reasoning: neural-chat:7b
    embeddings: nomic-embed-text

cloud:
  openai:
    api_key: ${OPENAI_API_KEY}
    models:
      reasoning: gpt-4
      fast: gpt-3.5-turbo
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    models:
      reasoning: claude-opus

memory:
  backend: postgres
  connection_string: ${DATABASE_URL}
  vector_db: weaviate
  retention_days: 90

safety:
  health_claims: strict
  fact_checking: enabled
  confidence_threshold: 0.7

agents:
  enabled:
    - research
    - coding
    - planning
    - health
    - sports
```

---

## Key APIs

### Chat Endpoint

```python
POST /api/chat

{
    "messages": [
        {"role": "user", "content": "..."}
    ],
    "user_id": "...",
    "project_id": "...",  # optional
    "model": "auto",  # auto|local|cloud
    "tools": ["web_search", "python"],  # optional
    "temperature": 0.7,
    "max_tokens": 2048,
    "stream": true
}

Response:
{
    "id": "...",
    "content": "...",
    "model_used": "mistral:7b",
    "routing_decision": {...},
    "tools_used": [...],
    "confidence": 0.85,
    "citations": [...],
    "usage": {
        "input_tokens": 150,
        "output_tokens": 450,
        "cost_usd": 0.002
    }
}
```

### Memory Endpoint

```python
GET /api/memory/{user_id}
    → Retrieve user's persistent memory

POST /api/memory/{user_id}
    → Store or update memory

DELETE /api/memory/{user_id}/item/{item_id}
    → Delete specific memory item

POST /api/memory/{user_id}/clear
    → Clear all memory (requires confirmation)
```

### State Endpoint

```python
GET /api/state/{user_id}
    → Get current personal state

POST /api/state/{user_id}
    → Update state with new evidence

GET /api/state/{user_id}/history
    → Get historical state data
```

### Tools Endpoint

```python
GET /api/tools
    → List available tools

POST /api/tools/{tool_name}/execute
    → Execute specific tool

GET /api/tools/{tool_name}/schema
    → Get tool schema for LLM calling
```

---

## Common Patterns

### Pattern 1: Simple Conversation
```python
# Route to local model, no verification needed
response = await router.route(
    query="Hello, how are you?",
    task_type=TaskType.GENERAL_CONVERSATION,
    policy=RoutingPolicy.LOW_LATENCY
)
# Use first model in chain
model = response.primary_model
```

### Pattern 2: Research Question
```python
# Need web search + reasoning
response = await router.route(
    query="Latest developments in quantum computing",
    task_type=TaskType.RESEARCH,
    required_tools=["web_search"],
    policy=RoutingPolicy.BALANCED
)
# Likely routes to cloud model
# Execute search tool, pass results to LLM
```

### Pattern 3: Code Analysis
```python
# Sensitive local data + complex reasoning
response = await router.route(
    query="Optimize this code",
    task_type=TaskType.CODING,
    has_sensitive_data=True,
    policy=RoutingPolicy.HYBRID  # or PRIVACY_FIRST
)
# Local analysis + optional cloud verification
```

### Pattern 4: Health Question
```python
# ALWAYS local + verification + safety checks
response = await router.route(
    query="Why am I tired?",
    task_type=TaskType.HEALTH,
    policy=RoutingPolicy.LOCAL_ONLY
)
# Use health safety checks
# Flag confidence levels
# Recommend professional consultation
```

---

## Performance Metrics to Track

### Quality Metrics
- Response helpfulness (user rating)
- Factual accuracy (fact-check pass rate)
- Task completion rate
- User satisfaction score

### Performance Metrics
- Response latency (p50, p95, p99)
- Token/second throughput
- Tool execution time
- End-to-end latency

### Cost Metrics
- Cost per request
- Cost per successful response
- Cloud API spend
- Local compute utilization

### Routing Metrics
- Model selection accuracy
- Routing policy effectiveness
- Fallback activation rate
- Cost vs. quality tradeoff

---

## Debugging Checklist

### Request Not Routed Correctly?
1. Check task classification accuracy
2. Verify model availability
3. Review routing policy
4. Check user preferences/memory
5. Verify context aggregation

### Response Quality Issues?
1. Is the right model being used?
2. Is context being retrieved?
3. Are tools being called?
4. Is verification running?
5. What was the confidence score?

### Performance Issues?
1. Local model performance
2. Cloud API latency
3. Memory/RAG retrieval time
4. Tool execution time
5. Overall orchestration time

### Memory Issues?
1. Is memory being stored?
2. Is memory being retrieved?
3. Are retention policies working?
4. Check memory database health

---

## Testing Strategy

### Unit Tests
- Task classifier accuracy
- Memory CRUD operations
- Tool schema validation
- Model provider interface compliance

### Integration Tests
- End-to-end chat flow
- Routing decisions
- Memory integration
- Tool execution with LLM

### Evaluation Tests
- Model quality benchmarks
- Routing strategy effectiveness
- Cost vs. quality tradeoff
- Latency requirements

### Safety Tests
- Health safety violations
- Privacy boundary checks
- Harmful content filtering
- Bias detection

---

## Deployment Checklist

- [ ] All environment variables configured
- [ ] Database migrations run
- [ ] Vector database initialized
- [ ] Local models downloaded/cached
- [ ] Cloud API credentials validated
- [ ] Logging configured
- [ ] Monitoring configured
- [ ] Health checks passing
- [ ] Rate limiting configured
- [ ] Backup strategy in place
- [ ] Privacy policies implemented
- [ ] Error handling verified
- [ ] Load testing passed
- [ ] Security audit completed

---

## Troubleshooting

### "Model not available"
→ Check Ollama running / API credentials / rate limits

### "Tool execution failed"
→ Check tool permissions / arguments / environment

### "Memory not retrieved"
→ Check vector DB / embeddings / retrieval threshold

### "High latency"
→ Profile request → Check local model / cloud API / tools

### "Incorrect routing"
→ Check task classifier / policy configuration / user preferences

---

## Resources

- **Architecture**: `/docs/NEXUS_ARCHITECTURE_EXPANDED.md`
- **System Prompt**: `/docs/NEXUS_SYSTEM_PROMPT.md`
- **API Docs**: `/docs/api/` (TBD)
- **Configuration**: `/config/`
- **Examples**: `/examples/`

---

## Next Steps

1. Start with Phase 0-1 (Foundation + Local LLM)
2. Build out core APIs
3. Implement basic memory
4. Add cloud providers
5. Implement intelligent routing
6. Add tools
7. Expand to advanced features

Good luck building NEXUS!
