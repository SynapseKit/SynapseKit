"""Time-Travel Codebase subpackage for reasoning across code evolution and git history."""

from .agent import TimeTravelAgent
from .drift_detector import DriftCandidate, DriftDetector
from .evolution_index import EvolutionEntry, EvolutionIndex
from .git_backend import AsOf, CommitInfo, GitBackend
from .narrative import DiffNarrativeGenerator

__all__ = [
    "AsOf",
    "CommitInfo",
    "DiffNarrativeGenerator",
    "DriftCandidate",
    "DriftDetector",
    "EvolutionEntry",
    "EvolutionIndex",
    "GitBackend",
    "TimeTravelAgent",
]
