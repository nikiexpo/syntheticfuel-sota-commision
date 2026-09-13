"""Baseline dispatch strategies used as comparators in the report."""

from sfp.control.baselines.greedy import GreedyController
from sfp.control.baselines.oracle import PerfectForesightOracle
from sfp.control.baselines.rulebased import RuleBasedController

__all__ = ["GreedyController", "PerfectForesightOracle", "RuleBasedController"]
