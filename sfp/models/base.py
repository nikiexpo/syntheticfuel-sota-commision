"""The subsystem interface every reduced-order model implements.

One model, two consumers. A `Subsystem` is written once against
`sfp.models.mathx` and then evaluated numerically by the truth simulator and
symbolically by the controller. A digital twin whose controller silently
disagrees with the plant is a demo, not a control system.

Conventions
-----------
states       SI, ordered, named. `rhs` returns dx/dt in state units per second.
inputs       the manipulated variables the controller may set.
disturbances exogenous signals (weather, upstream flows) it cannot set.
outputs      everything else worth logging: powers, flows, temperatures, rates.

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

    # --- coupling between subsystems --------------------------------------
    @property
    def requires(self) -> tuple[str, ...]:
        """Coupling keys this subsystem's `outputs` reads out of `w`.

        Declared so `Plant` can verify at construction that an earlier subsystem
        or the weather produces them. Without that check a typo silently yields
        `w.get(key, 0.0)` -- a zero rate indistinguishable from a plant that
        chose not to run.
        """
        return ()

    @property
    def requires_for_rhs(self) -> tuple[str, ...]:
        """Coupling keys `rhs` reads. No ordering constraint: every subsystem's
        outputs are available by the time any `rhs` is evaluated."""
        return ()

    @property
    def provides(self) -> tuple[str, ...]:
        """Coupling keys this subsystem publishes for others to consume."""
        return ()

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

    # --- uniform dispatch protocol ----------------------------------------
    # Every controllable subsystem takes (setpoint, enable) as its inputs, so the
    # DC bus can size and shed any of them without knowing what it is.

    def power_for_setpoint(
        self, x, setpoint: float, enable: float = 1.0, w: Mapping[str, Any] | None = None
    ) -> float:
        """Electrical draw at a commanded setpoint, W."""
        if self.n_inputs < 2:
            return 0.0
        u = np.array([setpoint, enable], dtype=float)
        return float(self.outputs(0.0, x, u, w or {})["power_electrical_W"])

    def parasitic_power_W(
        self, x, enable: float = 1.0, w: Mapping[str, Any] | None = None
    ) -> float:
        """Draw with the setpoint at zero but the subsystem energised.

        The floor a dispatch must clear before this subsystem can do any useful
        work. For the Sabatier reactor this is almost its entire draw, which is
        why turning its feed down saves nothing and only shutting it off does.
        """
        return self.power_for_setpoint(x, 0.0, enable, w)

    def setpoint_for_power(
        self,
        x,
        power_W: float,
        enable: float = 1.0,
        w: Mapping[str, Any] | None = None,
        min_setpoint: float = 0.0,
    ) -> float:
        """Largest setpoint whose draw fits inside `power_W`.

        Bisection rather than an analytic inverse: draw is monotone in setpoint
        everywhere, but the shape varies (cubic fan, linear heater,
        polarisation curve, flat reactor). Returns 0 if even the parasitic
        floor cannot be met.
        """
        if enable < 0.5 or self.n_inputs < 2:
            return 0.0
        if self.power_for_setpoint(x, min_setpoint, enable, w) > power_W + 1e-9:
            return 0.0
        if self.power_for_setpoint(x, 1.0, enable, w) <= power_W:
            return 1.0
        low, high = min_setpoint, 1.0
        for _ in range(40):
            mid = 0.5 * (low + high)
            if self.power_for_setpoint(x, mid, enable, w) <= power_W:
                low = mid
            else:
                high = mid
        return float(low)

    @property
    def min_setpoint(self) -> float:
        """Lowest setpoint at which the subsystem may operate at all.

        Nonzero where a real turndown limit exists -- gas crossover in the
        electrolyser, for instance. The bus snaps anything below this to zero.
        """
        return 0.0

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
