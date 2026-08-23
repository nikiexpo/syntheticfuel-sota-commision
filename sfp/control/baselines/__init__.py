"""Baseline dispatch strategies used as comparators in the report."""

from sfp.control.baselines.greedy import GreedyController
from sfp.control.baselines.rulebased import RuleBasedController

__all__ = ["GreedyController", "RuleBasedController"]
