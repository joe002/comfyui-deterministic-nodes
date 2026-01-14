"""
ComfyUI Deterministic Nodes
===========================

Batch-invariant inference nodes for guaranteed reproducibility.

Based on:
- ThinkingMachines batch-invariant-ops research
- NVIDIA CES 2026 multi-model agent patterns
- Ralph v3 file-centric state management
- ECHO 2.0 4-tier context memory

Nodes:
- DeterministicSampler: Batch-invariant sampling (batch_size=1 per item)
- ChecksumValidator: Reproducibility proof with tolerance modes
- MoERouterNode: Deterministic expert routing (hash-based)
- CascadeRefiner: Sequential 3-stage refinement (Nemotron pattern)
- ECHOContextNode: 4-tier context memory retrieval
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "1.0.0"
