"""The controller interface.

Every strategy -- the greedy baseline, the rule-based PLC emulation, the
perfect-foresight oracle, and the hierarchical economic MPC that is the point of
the project -- implements this one interface and is therefore directly
comparable. Nothing else about the simulation changes between runs, so a
difference in the report is a difference in the control strategy and not in the
harness.

A controller sees:

    t          seconds since the start of the run
    state      the *estimated* plant state (in M0 this is the true state; from
               M5 it comes from the moving-horizon estimator, and the difference
               starts to matter)
    measurement the current noisy sensor readings and weather
    forecast   a WeatherSeries view of the future -- imperfect unless the
               controller is the oracle

and returns a `Request`, which the DC bus then makes feasible. A controller is
never allowed to write plant state directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from sfp.sim.bus import Request


@dataclass
class ControlContext:
    """Everything a controller is allowed to know at construction time.

    Deliberately includes the plant *models* -- a model-based controller is
    supposed to have a model. What it does not include is the truth simulator's
    perturbed parameters, so the controller's model is always slightly wrong,
    which is the entire point of the plant/model split.
    """

    plant: Any
    site: Any
    dt_s: float
    horizon_s: float = 86400.0
    forecast: Any = None
    economics: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Controller(ABC):
    """Base class for all dispatch strategies."""

    #: short name used in report tables and filenames
    name: str = "controller"

    #: one-line description for the comparison table
    description: str = ""

    def reset(self, context: ControlContext) -> None:
        """Called once before a run. Store what you need; do not touch the plant."""
        self.context = context

    @abstractmethod
    def act(
        self,
        t: float,
        state: Mapping[str, np.ndarray],
        measurement: Mapping[str, Any],
        forecast: Any = None,
    ) -> Request:
        """Return the desired dispatch for the interval starting at `t`."""

    def diagnostics(self) -> dict[str, float]:
        """Optional per-step internals worth logging (solve time, objective, duals)."""
        return {}

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(name={self.name!r})"
