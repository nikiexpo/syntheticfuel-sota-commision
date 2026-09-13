"""The plant: coupled subsystems sharing one DC bus and several material buffers.

Holds the concatenated state vector, knows how to slice it per subsystem, and --
new in M1 -- resolves the coupling between subsystems.

The coupling problem
--------------------
Subsystems are no longer independent. The solids inventory is driven by rates
computed in the contactor and the calciner; the gas buffers are driven by the
calciner, the electrolyser and the reactor; the reactor's feed is throttled by
what the gas buffers actually hold. None of that fits through a subsystem's own
inputs, because none of it is a decision -- it is physics between components.

So evaluation happens in two phases:

    phase 1   outputs, in registration order, each subsystem seeing the weather,
              every subsystem's *state*, and the outputs of everything before it
    phase 2   dx/dt, every subsystem seeing the complete set of outputs

Phase 1 is order-dependent, which is a real hazard: a subsystem reading a
coupling key that has not been produced yet would get `w.get(key, 0.0)`, and a
silent zero rate looks exactly like a plant that chose not to run. So each
subsystem declares what it `requires` and `provides`, and `Plant` verifies the
ordering at construction. Get the order wrong and you get an exception naming
the offending key, not a quietly wrong simulation.

Phase 2 has no ordering constraint, which is why the genuinely circular couplings
(the gas buffer needs the reactor's draw, the reactor needs the buffer's level)
live there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import casadi as ca
import numpy as np

from sfp.models import mathx as mx
from sfp.models.base import Subsystem

#: Keys supplied by the weather series, always available in phase 1.
WEATHER_KEYS = frozenset(
    {
        "poa_global",
        "ghi",
        "ghi_clear",
        "clearsky_index",
        "temp_air",
        "relative_humidity",
        "wind_speed",
        "pressure",
        "cos_zenith",
    }
)


class CouplingError(ValueError):
    """Raised when a subsystem needs a coupling signal nobody produces in time."""


@dataclass
class Plant:
    """An ordered set of coupled subsystems with a shared state vector.

    Order matters: it is the evaluation order for phase 1. The reference
    ordering is solids -> contactor -> calciner -> gas -> water -> electrolyser
    -> sabatier, which follows the material flow.
    """

    subsystems: dict[str, Subsystem]

    def __post_init__(self) -> None:
        self._slices: dict[str, slice] = {}
        offset = 0
        for key, sub in self.subsystems.items():
            n = sub.n_states
            self._slices[key] = slice(offset, offset + n)
            offset += n
        self.n_states = offset
        self._validate_coupling()

    # --- validation -------------------------------------------------------
    def _validate_coupling(self) -> None:
        """Check that every phase-1 dependency is satisfiable in this order."""
        available = set(WEATHER_KEYS)
        # every subsystem's state is published before phase 1 begins
        for key, sub in self.subsystems.items():
            available.update(f"{key}.{name}" for name in sub.states.names)

        produced_by: dict[str, str] = {}
        for key, sub in self.subsystems.items():
            missing = [r for r in sub.requires if r not in available]
            if missing:
                raise CouplingError(
                    f"subsystem {key!r} requires {missing} in phase 1, but nothing "
                    f"before it provides them. Either reorder the plant so its "
                    f"producer comes first, or move the dependency into `rhs` "
                    f"(phase 2), where ordering does not matter."
                )
            for name in sub.provides:
                if name in produced_by:
                    raise CouplingError(
                        f"coupling key {name!r} is provided by both "
                        f"{produced_by[name]!r} and {key!r}; keys must be unique"
                    )
                produced_by[name] = key
            available.update(sub.provides)

        # phase-2 dependencies only need to exist somewhere
        for key, sub in self.subsystems.items():
            missing = [r for r in sub.requires_for_rhs if r not in available]
            if missing:
                raise CouplingError(
                    f"subsystem {key!r} requires {missing} in `rhs`, but no "
                    f"subsystem provides them anywhere in this plant"
                )

    # --- structure --------------------------------------------------------
    def __getitem__(self, key: str) -> Subsystem:
        return self.subsystems[key]

    def __contains__(self, key: str) -> bool:
        return key in self.subsystems

    def __iter__(self) -> Iterator[tuple[str, Subsystem]]:
        return iter(self.subsystems.items())

    def slice_of(self, key: str) -> slice:
        return self._slices[key]

    def split(self, x: np.ndarray) -> dict[str, np.ndarray]:
        return {key: x[sl] for key, sl in self._slices.items()}

    def join(self, states: Mapping[str, np.ndarray]) -> np.ndarray:
        if not self.n_states:
            return np.zeros(0)
        return np.concatenate(
            [np.asarray(states[key], dtype=float).reshape(-1) for key in self.subsystems]
        )

    def state_names(self) -> list[str]:
        return [f"{key}.{name}" for key, sub in self for name in sub.states.names]

    def initial_state(self) -> np.ndarray:
        return self.join({key: sub.initial_state() for key, sub in self})

    # --- coupled evaluation -----------------------------------------------
    def publish_states(self, x: np.ndarray, w: Mapping[str, Any]) -> dict[str, Any]:
        """Weather plus every subsystem state, keyed `<subsystem>.<state>`."""
        out = dict(w)
        states = self.split(x)
        for key, sub in self:
            for i, name in enumerate(sub.states.names):
                out[f"{key}.{name}"] = states[key][i]
        return out

    def evaluate(
        self, t: float, x: np.ndarray, u: Mapping[str, np.ndarray], w: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Phase 1. Returns (outputs, extended-w including those outputs)."""
        context = self.publish_states(x, w)
        states = self.split(x)
        outputs: dict[str, Any] = {}

        for key, sub in self:
            produced = sub.outputs(t, states[key], u[key], context)
            for name, value in produced.items():
                # Each subsystem reports its own draw as `power_electrical_W`, so
                # namespace it. It is also stored under `power.<key>`, which is
                # the canonical channel the bus and the metrics read: matching on
                # a `_power_W` suffix would wrongly pick up quantities like
                # `contactor_fan_power_W` that are components of a draw, not draws.
                if name == "power_electrical_W":
                    outputs[f"{key}_power_W"] = value
                    outputs[f"power.{key}"] = value
                    context[f"{key}_power_W"] = value
                    context[f"power.{key}"] = value
                else:
                    outputs[name] = value
                    context[name] = value

        return outputs, context

    def electrical_powers_W(
        self, t: float, x: np.ndarray, u: Mapping[str, np.ndarray], w: Mapping[str, Any]
    ) -> dict[str, float]:
        """Per-subsystem electrical draw, W. Positive = consumed from the bus."""
        outputs, _ = self.evaluate(t, x, u, w)
        return {
            key: float(outputs[f"power.{key}"])
            for key in self.subsystems
            if f"power.{key}" in outputs
        }

    def outputs(
        self, t: float, x: np.ndarray, u: Mapping[str, np.ndarray], w: Mapping[str, Any]
    ) -> dict[str, Any]:
        return self.evaluate(t, x, u, w)[0]

    def rhs(self, t: float, x: np.ndarray, u: Mapping[str, np.ndarray], w: Mapping[str, Any]):
        """Phase 2. Combined dx/dt with every coupling signal resolved.

        Dispatches on whether anything in the assembled derivative is symbolic,
        so the same coupled model serves the simulator numerically and the inner
        NMPC symbolically. Forcing the numeric path (`np.asarray(..., float)`)
        unconditionally is what would otherwise stop the controller from using
        the plant's own equations -- and a controller with its own re-typed copy
        of the dynamics is the thing this project exists to avoid.
        """
        _, context = self.evaluate(t, x, u, w)
        states = self.split(x)
        parts = []
        for key, sub in self:
            if sub.n_states == 0:
                continue
            parts.append(sub.rhs(t, states[key], u[key], context))
        if not parts:
            return np.zeros(0)
        if mx.any_sym(*parts):
            return ca.vertcat(*parts)
        return np.concatenate(
            [np.asarray(p, dtype=float).reshape(-1) for p in parts]
        )

    def electrical_load_W(
        self, t: float, x: np.ndarray, u: Mapping[str, np.ndarray], w: Mapping[str, Any]
    ) -> float:
        """Net consumption across every subsystem, W. Zero on a balanced bus."""
        return float(sum(self.electrical_powers_W(t, x, u, w).values()))

    # --- integration ------------------------------------------------------
    def step(
        self,
        t: float,
        x: np.ndarray,
        u: Mapping[str, np.ndarray],
        w: Mapping[str, Any],
        dt: float,
        clip: bool = True,
    ) -> np.ndarray:
        """One fixed-step RK4 advance with a zero-order hold on `u` and `w`.

        `clip=False` skips the projection back into the state box, which is
        required for symbolic use: `clip_state` calls `np.array(..., dtype=float)`
        and cannot accept a CasADi expression. It is also the *right* thing for an
        optimiser, which enforces those bounds as explicit constraints -- clipping
        inside the dynamics would hide a violation from the solver rather than
        letting it see and respect the bound.
        """
        if self.n_states == 0:
            return x
        k1 = self.rhs(t, x, u, w)
        k2 = self.rhs(t + 0.5 * dt, x + 0.5 * dt * k1, u, w)
        k3 = self.rhs(t + 0.5 * dt, x + 0.5 * dt * k2, u, w)
        k4 = self.rhs(t + dt, x + dt * k3, u, w)
        advanced = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.clip_state(advanced) if clip else advanced

    def clip_state(self, x: np.ndarray) -> np.ndarray:
        """Project the state back into its physical box.

        Integration error can push a state a hair outside its bounds. Clipping
        keeps downstream code honest; a gross violation would be a real
        modelling bug and the test suite checks for that separately.
        """
        out = np.array(x, dtype=float, copy=True)
        for key, sub in self:
            if sub.n_states == 0:
                continue
            lower, upper = sub.state_bounds()
            sl = self._slices[key]
            out[sl] = np.clip(out[sl], lower, upper)
        return out
