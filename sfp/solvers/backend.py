"""The solver interface, and the builder that keeps index bookkeeping honest.

Three types matter here.

`NLPBuilder` accumulates decision variables and constraints *by name*. Writing a
240-step planner by hand-slicing one long vector is how sign errors and off-by-one
index bugs get into an optimiser, and those are invisible: the solve converges, the
answer is wrong, and nothing raises. The builder means the planner never computes an
index.

`NLP` is the resulting problem in a standard form:

    min  f(x)   s.t.   lbg <= g(x) <= ubg,   lbx <= x <= ubx

`SolverBackend` solves one. The contract is deliberately narrow -- everything a
backend needs is a numerical callable through `NLP.functions()` -- so a backend
need not be CasADi-based: it consumes f, g, their derivatives and the Lagrangian
Hessian, and never sees a CasADi solver object.

Duals are first-class. Lambda, the dual of the energy-balance constraint, is the
internal electricity price the price-coordinated hierarchy runs on, so a
`Solution` carries `lam_g` and returns it by constraint name via `dual(name)`.
A backend that cannot report duals says so through `provides_duals`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import casadi as ca
import numpy as np


# ---------------------------------------------------------------------------
# problem definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Block:
    """A named contiguous slice of the variable or constraint vector."""

    name: str
    start: int
    size: int

    @property
    def stop(self) -> int:
        return self.start + self.size

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass
class NLPFunctions:
    """Numerical callables for a problem: f, g and their derivatives.

    The entire surface a solver backend needs, deliberately plain -- numpy in,
    numpy out -- so a backend written without CasADi can be dropped in against
    exactly this contract.
    """

    n_x: int
    n_g: int
    f: Callable[[np.ndarray], float]
    g: Callable[[np.ndarray], np.ndarray]
    grad_f: Callable[[np.ndarray], np.ndarray]
    jac_g: Callable[[np.ndarray], np.ndarray]
    hess_lag: Callable[[np.ndarray, np.ndarray, float], np.ndarray]

    def kkt_residual(
        self, x: np.ndarray, lam_g: np.ndarray, lam_x: np.ndarray | None = None
    ) -> float:
        """Infinity norm of the stationarity residual of the Lagrangian.

        Reported alongside every solve so "it converged" is a measured claim
        rather than a status string, and so two backends can be compared on a
        quantity both must agree about.
        """
        r = self.grad_f(x) + self.jac_g(x).T @ np.asarray(lam_g, dtype=float).ravel()
        if lam_x is not None:
            r = r + np.asarray(lam_x, dtype=float).ravel()
        return float(np.max(np.abs(r))) if r.size else 0.0


@dataclass
class NLP:
    """A nonlinear program plus the names of its parts.

    `x`, `f` and `g` are CasADi expressions; the blocks record which slice of the
    vectors each named group of variables or constraints occupies.
    """

    x: ca.MX
    f: ca.MX
    g: ca.MX
    lbx: np.ndarray
    ubx: np.ndarray
    lbg: np.ndarray
    ubg: np.ndarray
    x0: np.ndarray
    var_blocks: dict[str, Block] = field(default_factory=dict)
    con_blocks: dict[str, Block] = field(default_factory=dict)
    name: str = "nlp"

    _functions: NLPFunctions | None = field(default=None, repr=False, compare=False)

    @property
    def n_x(self) -> int:
        return int(self.x.numel())

    @property
    def n_g(self) -> int:
        return int(self.g.numel()) if self.g.numel() else 0

    def functions(self) -> NLPFunctions:
        """Compile (once) the numerical callables a backend needs."""
        if self._functions is not None:
            return self._functions

        x = self.x
        lam = ca.MX.sym("lam", self.n_g)
        sigma = ca.MX.sym("sigma")

        f_fn = ca.Function("f", [x], [self.f])
        g_fn = ca.Function("g", [x], [self.g])
        grad_fn = ca.Function("grad_f", [x], [ca.gradient(self.f, x)])
        jac_fn = ca.Function("jac_g", [x], [ca.jacobian(self.g, x)])
        lagrangian = sigma * self.f + (ca.dot(lam, self.g) if self.n_g else 0.0)
        hess_fn = ca.Function("hess_lag", [x, lam, sigma], [ca.hessian(lagrangian, x)[0]])

        def as_array(fn, *args) -> np.ndarray:
            return np.asarray(fn(*args)).astype(float)

        self._functions = NLPFunctions(
            n_x=self.n_x,
            n_g=self.n_g,
            f=lambda v: float(f_fn(v)),
            g=lambda v: as_array(g_fn, v).ravel(),
            grad_f=lambda v: as_array(grad_fn, v).ravel(),
            jac_g=lambda v: as_array(jac_fn, v).reshape(self.n_g, self.n_x),
            hess_lag=lambda v, l, s=1.0: as_array(hess_fn, v, l, s).reshape(
                self.n_x, self.n_x
            ),
        )
        return self._functions


# ---------------------------------------------------------------------------
# solutions
# ---------------------------------------------------------------------------


@dataclass
class SolveStats:
    """What happened during a solve. Logged for every call."""

    backend: str = ""
    status: str = ""
    success: bool = False
    iterations: int = 0
    wall_time_s: float = 0.0
    objective: float = float("nan")
    kkt_residual: float = float("nan")
    constraint_violation: float = float("nan")
    message: str = ""

    def as_dict(self) -> dict[str, float | str]:
        return {
            "solve_backend": self.backend,
            "solve_status": self.status,
            "solve_success": float(self.success),
            "solve_iterations": float(self.iterations),
            "solve_wall_time_s": self.wall_time_s,
            "solve_objective": self.objective,
            "solve_kkt_residual": self.kkt_residual,
            "solve_constraint_violation": self.constraint_violation,
        }


@dataclass
class Solution:
    """A solved (or failed) NLP, addressable by the names used to build it."""

    x: np.ndarray
    f: float
    lam_g: np.ndarray
    lam_x: np.ndarray
    stats: SolveStats
    nlp: NLP | None = field(default=None, repr=False)

    @property
    def success(self) -> bool:
        return self.stats.success

    def value(self, name: str) -> np.ndarray:
        """The solved values of a named variable block."""
        block = self._block(name, self.nlp.var_blocks if self.nlp else {}, "variable")
        return np.asarray(self.x[block.slice], dtype=float)

    def dual(self, name: str) -> np.ndarray:
        """The multipliers on a named constraint block.

        This is how the planner reads lambda off its energy-balance constraint,
        so the sign convention is worth stating exactly. CasADi/IPOPT returns
        `lam_g` satisfying stationarity in the form

            grad f + J^T lam = 0

        which for a *minimisation* of `f` subject to `g(x) = c` gives
        `d f* / dc = -lam`. This class stores a maximisation as the minimisation
        of `f = -J`, so the chain rule flips it back:

            d J* / dc = +lam                                (verified in tests)

        **For a problem built with `NLPBuilder.maximise`, `lam` is directly the
        marginal value of relaxing the constraint by one unit, in the objective's
        own units.** No further sign flip is needed or permitted -- getting this
        backwards would invert the planner's price signal, which would not raise
        anything, it would just make the plant do the opposite of the right thing
        at every hour. Pinned by `test_dual_of_maximisation_is_marginal_value`.
        """
        block = self._block(name, self.nlp.con_blocks if self.nlp else {}, "constraint")
        return np.asarray(self.lam_g[block.slice], dtype=float)

    @staticmethod
    def _block(name: str, blocks: dict[str, Block], kind: str) -> Block:
        if name not in blocks:
            known = ", ".join(sorted(blocks)) or "(none)"
            raise KeyError(f"unknown {kind} block {name!r}; known blocks: {known}")
        return blocks[name]


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------


class SolverBackend(ABC):
    """One way of solving an `NLP`."""

    #: short name used in configuration and in benchmark tables
    name: str = "backend"

    #: whether `Solution.lam_g` is meaningful. The outer planner requires duals.
    provides_duals: bool = True

    def __init__(self, **options: Any) -> None:
        self.options = dict(options)

    @abstractmethod
    def solve(self, nlp: NLP, *, x0: np.ndarray | None = None,
              lam_g0: np.ndarray | None = None) -> Solution:
        """Solve `nlp`, warm-starting from `x0` and `lam_g0` if given."""

    def _finish(
        self,
        nlp: NLP,
        x: np.ndarray,
        lam_g: np.ndarray,
        lam_x: np.ndarray,
        stats: SolveStats,
        started: float,
    ) -> Solution:
        """Fill in the measured diagnostics every backend must report."""
        fns = nlp.functions()
        stats.backend = self.name
        stats.wall_time_s = time.perf_counter() - started
        stats.objective = fns.f(x)
        if nlp.n_g:
            g = fns.g(x)
            violation = np.maximum(nlp.lbg - g, 0.0) + np.maximum(g - nlp.ubg, 0.0)
            stats.constraint_violation = float(np.max(violation))
        else:
            stats.constraint_violation = 0.0
        try:
            stats.kkt_residual = fns.kkt_residual(x, lam_g, lam_x)
        except Exception:  # pragma: no cover - diagnostics must never break a solve
            stats.kkt_residual = float("nan")
        return Solution(x=x, f=stats.objective, lam_g=lam_g, lam_x=lam_x,
                        stats=stats, nlp=nlp)


_REGISTRY: dict[str, type[SolverBackend]] = {}


def register_backend(cls: type[SolverBackend]) -> type[SolverBackend]:
    """Register a backend under its `name`, so it can be selected by string."""
    _REGISTRY[cls.name] = cls
    return cls


def available_backends() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_backend(name: str = "ipopt", **options: Any) -> SolverBackend:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown solver backend {name!r}; available: {', '.join(available_backends())}"
        )
    return _REGISTRY[name](**options)


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


class NLPBuilder:
    """Accumulate a named NLP without ever computing an index by hand.

        b = NLPBuilder("planner")
        soc = b.variable("soc", n=241, lb=0.1, ub=0.95, x0=0.5)
        b.constraint("soc_dynamics", soc[1:] - f(soc[:-1]), equals=0.0)
        b.minimise(-profit)
        nlp = b.build()

    Then `solution.value("soc")` and `solution.dual("bus_balance")` address the
    result by the same names. Bounds broadcast from scalars.
    """

    def __init__(self, name: str = "nlp") -> None:
        self.name = name
        self._vars: list[ca.MX] = []
        self._var_blocks: dict[str, Block] = {}
        self._lbx: list[np.ndarray] = []
        self._ubx: list[np.ndarray] = []
        self._x0: list[np.ndarray] = []
        self._n_x = 0

        self._cons: list[ca.MX] = []
        self._con_blocks: dict[str, Block] = {}
        self._lbg: list[np.ndarray] = []
        self._ubg: list[np.ndarray] = []
        self._n_g = 0

        self._objective: ca.MX | None = None

    # --- variables --------------------------------------------------------
    def variable(
        self,
        name: str,
        n: int = 1,
        *,
        lb: float | Sequence[float] = -np.inf,
        ub: float | Sequence[float] = np.inf,
        x0: float | Sequence[float] = 0.0,
    ) -> ca.MX:
        if name in self._var_blocks:
            raise ValueError(f"duplicate variable block {name!r}")
        if n <= 0:
            raise ValueError(f"variable block {name!r} must have n >= 1, got {n}")
        sym = ca.MX.sym(name, n)
        self._vars.append(sym)
        self._var_blocks[name] = Block(name, self._n_x, n)
        self._lbx.append(_broadcast(lb, n, f"{name}.lb"))
        self._ubx.append(_broadcast(ub, n, f"{name}.ub"))
        self._x0.append(np.clip(_broadcast(x0, n, f"{name}.x0"),
                                _broadcast(lb, n, "lb"), _broadcast(ub, n, "ub")))
        self._n_x += n
        return sym

    # --- constraints ------------------------------------------------------
    def constraint(
        self,
        name: str,
        expr: ca.MX,
        *,
        lb: float | Sequence[float] | None = None,
        ub: float | Sequence[float] | None = None,
        equals: float | Sequence[float] | None = None,
    ) -> None:
        """Add `lb <= expr <= ub`, or `expr == equals`."""
        if name in self._con_blocks:
            raise ValueError(f"duplicate constraint block {name!r}")
        if equals is not None:
            if lb is not None or ub is not None:
                raise ValueError("give either `equals` or `lb`/`ub`, not both")
            lb = ub = equals
        if lb is None and ub is None:
            raise ValueError(f"constraint {name!r} has no bounds")

        expr = ca.reshape(expr, expr.numel(), 1)
        n = int(expr.numel())
        if n == 0:
            return
        self._cons.append(expr)
        self._con_blocks[name] = Block(name, self._n_g, n)
        self._lbg.append(_broadcast(-np.inf if lb is None else lb, n, f"{name}.lb"))
        self._ubg.append(_broadcast(np.inf if ub is None else ub, n, f"{name}.ub"))
        self._n_g += n

    # --- objective --------------------------------------------------------
    def minimise(self, expr: ca.MX) -> None:
        self._objective = expr

    def maximise(self, expr: ca.MX) -> None:
        """Convenience for a profit objective; stored negated.

        Recorded here because it is the one place a sign can silently invert the
        meaning of every dual downstream. See `Solution.dual`.
        """
        self._objective = -expr

    # --- assembly ---------------------------------------------------------
    def build(self) -> NLP:
        if self._objective is None:
            raise ValueError("no objective set; call minimise() or maximise()")
        if not self._vars:
            raise ValueError("no decision variables")
        empty = ca.MX.zeros(0, 1)
        return NLP(
            x=ca.vertcat(*self._vars),
            f=self._objective,
            g=ca.vertcat(*self._cons) if self._cons else empty,
            lbx=np.concatenate(self._lbx),
            ubx=np.concatenate(self._ubx),
            lbg=np.concatenate(self._lbg) if self._lbg else np.zeros(0),
            ubg=np.concatenate(self._ubg) if self._ubg else np.zeros(0),
            x0=np.concatenate(self._x0),
            var_blocks=dict(self._var_blocks),
            con_blocks=dict(self._con_blocks),
            name=self.name,
        )


def _broadcast(value, n: int, what: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    arr = arr.ravel()
    if arr.size != n:
        raise ValueError(f"{what}: expected {n} values, got {arr.size}")
    return arr.astype(float)
