"""Deterministic classification layer (task type, privacy tier).

Everything in this package is rule/keyword/regex-based and synchronous —
no LLM calls, no I/O — per the project principle that classification this
cheap and safety-relevant should never depend on a model call. `router.py`
in `nexus.core` imports from here; this package never imports from
`nexus.core.router` to avoid a cycle.
"""
