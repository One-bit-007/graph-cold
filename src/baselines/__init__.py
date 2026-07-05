"""Real label-noise baselines used by the D5.5 expansion gate."""
from __future__ import annotations

from src.baselines.base import BaselineResult
from src.baselines.confident_learning import ConfidentLearningBaseline
from src.baselines.coteaching import CoTeachingLiteBaseline
from src.baselines.decoupling import DecouplingBaseline
from src.baselines.fine_style import FINEStyleBaseline
from src.baselines.noisy_supervised import NoisySupervisedBaseline

__all__ = [
    "BaselineResult",
    "ConfidentLearningBaseline",
    "CoTeachingLiteBaseline",
    "DecouplingBaseline",
    "FINEStyleBaseline",
    "NoisySupervisedBaseline",
]
