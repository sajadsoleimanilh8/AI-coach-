"""Cloud AIProvider adapters (OpenAI, Anthropic, Gemini).

All adapters call their provider's REST API directly via `httpx.AsyncClient`
rather than depending on the official `openai`/`anthropic` SDKs. This keeps
the dependency footprint identical to `nexus.models.local.runtime.OllamaRuntime`
(already httpx-only) and lets tests inject `httpx.MockTransport` the same way
`test_ollama_runtime.py` does, with no real network calls and no extra test
dependency.
"""
