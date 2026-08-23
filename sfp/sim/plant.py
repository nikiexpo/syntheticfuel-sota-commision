"""The plant: an ordered collection of subsystems sharing one DC bus.

Holds the concatenated state vector and knows how to slice it per subsystem, so
adding the real chain in M1 is a matter of registering more subsystems rather
than rewriting the simulator. Integration is fixed-step RK4 with a zero-order
hold on the inputs, which is what a real DCS does anyway: the controller writes
setpoints at its own cadence and the plant integrates continuously between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import numpy as np

from sfp.models.base import Subsystem


@dataclass
class Plant:
    """An ordered set of subsystems with a shared state vector."""

    subsystems: dict[str, Subsystem]

    def __post_init__(self) -> None:
        self._slices: dict[str, slice] = {}
        offset = 0
        for key, sub in self.subsystems.items():
            n = sub.n_states
            self._slices[key] = slice(offset, offset + n)
            offset += n
        self.n_states = offset

    # --- structure --------------------------------------------------------
    def __getitem__(self, key: str) -> Subsystem:
        return self.subsystems[key]

    def __iter__(self) -> Iterator[tuple[str, Subsystem]]:
        return iter(self.subsystems.items())

    def slice_of(self, key: str) -> slice:
        return self._slices[key]

    def split(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Combined state vector -> {subsystem: its state slice}."""
        return {key: x[sl] for key, sl in self._slices.items()}

    def join(self, states: Mapping[str, np.ndarray]) -> np.ndarray:
        """{subsystem: state} -> combined vector."""
        return np.concatenate(
            [np.asarray(states[key], dtype=float).reshape(-1) for key in self.subsystems]
        ) if self.n_states else np.zeros(0)

    def state_names(self) -> list[str]:
        return [f"{key}.{name}" for key, sub in self for name in sub.states.names]

    def initial_state(self) -> np.ndarray:
        return self.join({key: sub.initial_state() for key, sub in self})

    # --- dynamics ---------------------------------------------------------
    def rhs(self, t: float, x: np.ndarray, u: Mapping[str, np.ndarray], w: Mapping[str, Any]):
        """Combined dx/dt. `u` maps subsystem name -> its input vector."""
        parts = []
        states = self.split(x)
        for key, sub in self:
            if sub.n_states == 0:
                continue
            parts.append(np.asarray(sub.rhs(t, states[key], u[key], w), dtype=float).reshape(-1))
        return np.concatenate(parts) if parts else np.zeros(0)

    def outputs(
        self, t: float, x: np.ndarray, u: Mapping[str, np.ndarray], w: Mapping[str, Any]
    ) -> dict[str, float]:
        """Every subsystem's outputs, flattened into one mapping."""
        out: dict[str, float] = {}
        states = self.split(x)
        for key, sub in self:
            for name, value in sub.outputs(t, states[key], u[key], w).items():
                out[name if name != "power_electrical_W" else f"{key}_power_W"] = value
        return out

    def bus_residual_W(
        self, t: float, x: np.ndarray, u: Mapping[str, np.ndarray], w: Mapping[str, Any]
    ) -> float:
        """Sum of electrical powers; zero on a balanced bus (positive = consumed)."""
        states = self.split(x)
        return float(
            sum(sub.electrical_power(t, states[key], u[key], w) for key, sub in self)
        )

    # --- integration ------------------------------------------------------
    def step(
        self,
        t: float,
        x: np.ndarray,
        u: Mapping[str, np.ndarray],
        w: Mapping[str, Any],
        dt: float,
    ) -> np.ndarray:
        """One fixed-step RK4 advance with a zero-order hold on `u` and `w`."""
        if self.n_states == 0:
            return x
        k1 = self.rhs(t, x, u, w)
        k2 = self.rhs(t + 0.5 * dt, x + 0.5 * dt * k1, u, w)
        k3 = self.rhs(t + 0.5 * dt, x + 0.5 * dt * k2, u, w)
        k4 = self.rhs(t + dt, x + dt * k3, u, w)
        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.clip_state(x_next)

    def clip_state(self, x: np.ndarray) -> np.ndarray:
        """Project the state back into its physical box.

        Integration error can push a state a hair outside its bounds (a state of
        charge of 1.0000001). Clipping keeps downstream code honest; anything
        more than a rounding error would be a real modelling bug, so the
        simulator asserts on gross violations separately.
        """
        out = np.array(x, dtype=float, copy=True)
        for key, sub in self:
            if sub.n_states == 0:
                continue
            lower, upper = sub.state_bounds()
            sl = self._slices[key]
            out[sl] = np.clip(out[sl], lower, upper)
        return out
