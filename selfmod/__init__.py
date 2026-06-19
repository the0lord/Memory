"""selfmod — a self-modifying memory-routing AI with a guarded deployment pipeline.

A decision engine (the memory router) evolves its own decision weights from
real-world outcomes. Every self-edit is bounded by an edit budget and must clear a
five-stage pipeline (SANDBOX → SHADOW → GATED → MONITORED → PROMOTED) before it
reaches production. Failed edits feed a rejected-edit buffer; value drift is
detected and auto-corrected toward a charter goal.
"""

from .orchestrator import SelfModifyingMemoryAI, CycleLog
from .pipeline import DeploymentPipeline, Stage, PipelineResult
from .weights import WeightVector, EditBudget, FEATURES, is_major_change

__all__ = [
    "SelfModifyingMemoryAI",
    "CycleLog",
    "DeploymentPipeline",
    "Stage",
    "PipelineResult",
    "WeightVector",
    "EditBudget",
    "FEATURES",
    "is_major_change",
]
