"""Real label-noise baselines used by the formal expansion matrix."""
from __future__ import annotations

from src.baselines.base import BaselineResult
from src.baselines.confident_learning import ConfidentLearningBaseline
from src.baselines.coteaching import CoTeachingBaseline, CoTeachingLiteBaseline
from src.baselines.decoupling import DecouplingBaseline
from src.baselines.fine_style import FINEBaseline, FINEStyleBaseline
from src.baselines.mcre import MCReBaseline
from src.baselines.morse import MORSEBaseline
from src.baselines.noisy_supervised import NoisySupervisedBaseline

__all__ = [
    "BaselineResult",
    "ConfidentLearningBaseline",
    "CoTeachingBaseline",
    "CoTeachingLiteBaseline",
    "DecouplingBaseline",
    "FINEBaseline",
    "FINEStyleBaseline",
    "MCReBaseline",
    "MORSEBaseline",
    "NoisySupervisedBaseline",
]
