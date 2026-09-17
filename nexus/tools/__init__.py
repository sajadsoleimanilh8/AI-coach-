"""Tool-calling layer: the Tool ABC, ToolRegistry, and individual tools.

Every tool must be individually enabled via `nexus.yaml`'s `tools.enabled`
list before the model can see or call it — see `nexus/api/main.py`'s
lifespan for how the enabled set is constructed.
"""
