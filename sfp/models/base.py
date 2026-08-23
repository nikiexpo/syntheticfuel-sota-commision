"""The subsystem interface every reduced-order model implements.

One model, three consumers. A `Subsystem` is written once against `sfp.models.mathx`
and is then evaluated numerically by the truth simulator and symbolically by the
estimator and the controller. That is the whole point: a digital twin whose
controller silently disagrees with the plant is a demo, not a control system.

Conventions
-----------
states      SI, ordered, named. `rhs` returns dx/dt in state units per second.
inputs      the manipulated variables the controller may set.
disturbances the exogenous signals (weather, upstream flows) the controller cannot set.
outputs     everything else worth logging: powers, flows, temperatures, rates.

A subsystem with no states (a purely algebraic block such as the PV array) is
legal: `states` is empty, `rhs` returns an empty vector, and only `outputs` does
work.

Sign convention for power: **positive means consumed from the DC bus**. A
generator therefore reports negative electrical power. This keeps the bus balance
a plain sum with no special cases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from sfp.models import mathx as mx


@dataclass(frozen=True)
class Signal:
    """A named, dimensioned scalar in a state / input / output vector."""

    name: str
    units: str
    description: str = ""
    lower: float = -np.inf
    upper: float = np.inf

    def clip(self, value):
        return mx.clip(value, self.lower, self.upper)


@dataclass
class VectorSpec:
    """An ordered, named vector with bounds -- states or inputs of a subsystem."""

    signals: tuple[Signal, ...] = ()

    def __len__(self) -> int:
        return len(self.signals)

    def __iter__(self):
        return iter(self.signals)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.signals)

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"no signal named {name!r}; have {self.names}") from None

    def get(self, vector, name: str):
        """Pull one named element out of a state/input vector."""
        return vector[self.index(name)]

    def unpack(self, vector) -> dict[str, Any]:
        """Whole vector as a {name: value} mapping."""
        return {s.name: vector[i] for i, s in enumerate(self.signals)}

    def pack(self, values: Mapping[str, float]) -> np.ndarray:
        """{name: value} mapping -> ordered numeric vector."""
        missing = set(self.names) - set(values)
        if missing:
            raise KeyError(f"missing values for {sorted(missing)}")
        return np.array([float(values[n]) for n in self.names], dtype=float)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.array([s.lower for s in self.signals], dtype=float)
        upper = np.array([s.upper for s in self.signals], dtype=float)
        return lower, upper


class Subsystem(ABC):
    """Base class for every plant subsystem."""

    #: short identifier, used as a prefix in the combined plant state
    name: str = "subsystem"

    def __init__(self, params) -> None:
        self.p = params

    # --- structure --------------------------------------------------------
    @property
    @abstractmethod
    def states(self) -> VectorSpec:
        """Ordered state specification (may be empty for algebraic blocks)."""

    @property
    @abstractmethod
    def inputs(self) -> VectorSpec:
        """Ordered manipulated-variable specification."""

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_inputs(self) -> int:
        return len(self.inputs)

    # --- dynamics ---------------------------------------------------------
    @abstractmethod
    def rhs(self, t, x, u, w: Mapping[str, Any]):
        """dx/dt in state-units per **second**.

        `t` is seconds since the start of the simulation, `x` the state vector,
        `u` the input vector, `w` a mapping of disturbances (weather etc).
        Must be written with `sfp.models.mathx` so it works numerically and
        symbolically.
        """

    @abstractmethod
    def outputs(self, t, x, u, w: Mapping[str, Any]) -> dict[str, Any]:
        """Derived quantities: at minimum `power_electrical_W` (positive = consumed)."""

    # --- initialisation and limits ---------------------------------------
    @abstractmethod
    def initial_state(self) -> np.ndarray:
        """A physically sensible cold-start state."""

    def state_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.states.bounds()

    def input_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.inputs.bounds()

    def electrical_power(self, t, x, u, w: Mapping[str, Any]):
        """Net electrical draw in W, positive = consumed from the bus."""
        return self.outputs(t, x, u, w)["power_electrical_W"]

    # --- convenience ------------------------------------------------------
    def zero_input(self) -> np.ndarray:
        """The 'do nothing' input -- clipped into the admissible box."""
        lower, upper = self.input_bounds()
        return np.clip(np.zeros(self.n_inputs), lower, upper)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"states={self.states.names}, inputs={self.inputs.names})"
        )


@dataclass
class Trajectory:
    """A logged simulation result: time plus any number of named series."""

    time_s: np.ndarray
    series: dict[str, np.ndarray] = field(default_factory=dict)

    def add(self, name: str, values: Sequence[float]) -> None:
        arr = np.asarray(values, dtype=float)
        if arr.shape[0] != self.time_s.shape[0]:
            raise ValueError(
                f"series {name!r} has length {arr.shape[0]}, expected {self.time_s.shape[0]}"
            )
        self.series[name] = arr

    def __getitem__(self, name: str) -> np.ndarray:
        return self.series[name]

    def __contains__(self, name: str) -> bool:
        return name in self.series

    @property
    def hours(self) -> np.ndarray:
        return self.time_s / 3600.0

    def integrate(self, name: str) -> float:
        """Trapezoidal integral of one series over time, in <units>*s."""
        return float(np.trapezoid(self.series[name], self.time_s))
