"""A linear-programming seam, separate from the NLP backend.

`sfp.solvers.backend` exists so the inner NMPC's nonlinear solver can be swapped
at M7. This is a different animal and deliberately not forced through the same
interface: an LP has no initial guess, no iteration count worth reporting, and a
dual convention of its own. Pretending otherwise would put a nonlinear shape on
a linear problem for no benefit.

HiGHS via `scipy.optimize.linprog` is the default and the only implementation.
It is bundled with SciPy, so this adds no dependency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


@dataclass
class LPProblem:
    """minimise c'x  s.t.  lo <= A x <= hi,  xl <= x <= xu."""

    c: np.ndarray
    A: Any                      # scipy.sparse matrix
    lo: np.ndarray
    hi: np.ndarray
    xl: np.ndarray
    xu: np.ndarray
    #: named row blocks, as (name -> slice), so duals can be read back by name
    blocks: dict[str, slice] = field(default_factory=dict)
    #: 1 where a column must take an integer value, 0 elsewhere. Any non-zero
    #: entry makes this a MILP and `solve_lp` switches solvers accordingly.
    integrality: np.ndarray | None = None

    @property
    def is_mip(self) -> bool:
        return self.integrality is not None and bool(np.any(self.integrality))

    @property
    def n_x(self) -> int:
        return len(self.c)

    @property
    def n_g(self) -> int:
        return self.A.shape[0]


@dataclass
class LPSolution:
    """A solved LP. `duals` follows the *marginal value* convention."""

    x: np.ndarray
    f: float
    success: bool
    status: str
    wall_time_s: float
    _duals: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    _blocks: dict[str, slice] = field(repr=False, default_factory=dict)

    def dual(self, block: str) -> np.ndarray:
        """Marginal value of relaxing the named constraint block.

        SciPy reports `marginals` as the derivative of the *minimised* objective
        with respect to the row's bound. The planner's problems are built as
        minimise(-profit), so the marginal value of profit is the negative of
        that. The sign is fixed here, once, rather than at every call site --
        getting it wrong silently turns a price into its own negation, and
        `test_dispatch_lambda_is_a_positive_price` pins it.
        """
        return -self._duals[self._blocks[block]]


class LPBuilder:
    """Accumulates rows and columns, then emits an `LPProblem`.

    Rows are appended in coordinate form and named in blocks. Columns are
    declared up front because the layout is fixed by the horizon.
    """

    def __init__(self, n_x: int) -> None:
        self.n_x = n_x
        self._rows: list[int] = []
        self._cols: list[int] = []
        self._vals: list[float] = []
        self._lo: list[float] = []
        self._hi: list[float] = []
        self._blocks: dict[str, slice] = {}
        self._open: str | None = None
        self._open_start = 0
        self.c = np.zeros(n_x)
        self.xl = np.full(n_x, -np.inf)
        self.xu = np.full(n_x, np.inf)
        self.integrality = np.zeros(n_x)

    # --- rows -------------------------------------------------------------
    def block(self, name: str) -> "LPBuilder":
        """Start a named row block. Rows added until the next `block` belong to it."""
        self._close()
        self._open = name
        self._open_start = len(self._lo)
        return self

    def _close(self) -> None:
        if self._open is not None:
            self._blocks[self._open] = slice(self._open_start, len(self._lo))
            self._open = None

    def row(self, entries, lo: float, hi: float) -> None:
        i = len(self._lo)
        for col, val in entries:
            self._rows.append(i)
            self._cols.append(int(col))
            self._vals.append(float(val))
        self._lo.append(lo)
        self._hi.append(hi)

    def equality(self, entries, rhs: float) -> None:
        self.row(entries, rhs, rhs)

    # --- columns ----------------------------------------------------------
    def bounds(self, cols, lo, hi) -> None:
        self.xl[cols] = lo
        self.xu[cols] = hi

    def cost(self, col: int, value: float) -> None:
        self.c[col] += value

    def integer(self, col: int) -> None:
        """Require an integer value in this column. Use sparingly."""
        self.integrality[col] = 1

    def build(self) -> LPProblem:
        self._close()
        A = sp.csc_matrix(
            (self._vals, (self._rows, self._cols)),
            shape=(len(self._lo), self.n_x),
        )
        return LPProblem(self.c, A, np.array(self._lo), np.array(self._hi),
                         self.xl, self.xu, dict(self._blocks),
                         self.integrality if np.any(self.integrality) else None)


def solve_lp(problem: LPProblem, *, time_limit_s: float = 120.0,
             mip_gap: float = 1e-3) -> LPSolution:
    """Solve with HiGHS. Returns duals under the marginal-value convention.

    A mixed-integer problem is solved in two passes: branch and bound for the
    schedule, then the integer columns are **fixed at their solved values and the
    continuous relaxation re-solved** to recover duals. A MILP has no duals of
    its own -- the value function is not convex -- so this is the standard way to
    get a price out of one, and it is the price conditional on the commitment
    actually chosen, which is the economically meaningful one.
    """
    started = time.perf_counter()

    if problem.is_mip:
        mip = _branch_and_bound(problem, time_limit_s, mip_gap)
        if mip.x is None:
            return LPSolution(np.zeros(problem.n_x), float("nan"), False,
                              str(mip.message)[:80], time.perf_counter() - started,
                              np.zeros(problem.n_g), problem.blocks)
        fixed = _with_integers_fixed(problem, np.asarray(mip.x, dtype=float))
        res = _highs(fixed, time_limit_s)
        # keep the MILP's own objective and point; the second solve exists only
        # for its duals and can differ in the last digits
        x, fun = np.asarray(mip.x, dtype=float), float(mip.fun)
        ok = True
    else:
        res = _highs(problem, time_limit_s)
        x = np.asarray(res.x, dtype=float) if res.x is not None else np.zeros(problem.n_x)
        fun = float(res.fun) if res.fun is not None else float("nan")
        ok = bool(res.status == 0)

    elapsed = time.perf_counter() - started

    duals = np.zeros(problem.n_g)
    marg = getattr(getattr(res, "ineqlin", None), "marginals", None)
    if marg is not None and np.ndim(marg) == 1 and len(marg) == problem.n_g:
        duals = np.asarray(marg, dtype=float)

    return LPSolution(
        x=x, f=fun, success=ok, status=str(res.message)[:80],
        wall_time_s=elapsed, _duals=duals, _blocks=problem.blocks,
    )


def _branch_and_bound(problem: LPProblem, time_limit_s: float, gap: float):
    from scipy.optimize import Bounds, LinearConstraint, milp
    return milp(
        problem.c,
        constraints=LinearConstraint(sp.csc_matrix(problem.A), problem.lo, problem.hi),
        bounds=Bounds(problem.xl, problem.xu),
        integrality=problem.integrality,
        options={"time_limit": time_limit_s, "mip_rel_gap": gap},
    )


def _with_integers_fixed(problem: LPProblem, x: np.ndarray) -> LPProblem:
    """The same problem with every integer column pinned to its solved value."""
    xl = np.array(problem.xl, dtype=float, copy=True)
    xu = np.array(problem.xu, dtype=float, copy=True)
    idx = np.flatnonzero(problem.integrality)
    xl[idx] = xu[idx] = np.round(x[idx])
    return LPProblem(problem.c, problem.A, problem.lo, problem.hi, xl, xu,
                     problem.blocks, None)


def _highs(problem: LPProblem, time_limit_s: float):
    """`linprog` with the two-sided rows expressed as A_ub, so duals come back.

    `scipy.optimize.milp` accepts two-sided `LinearConstraint` directly but does
    not return duals even for a pure LP. `linprog` does, via `ineqlin.marginals`,
    but only takes one-sided rows -- so each two-sided row is split. Equalities
    stay in `A_eq`... except that then their duals live in a different array and
    the block slices no longer line up. Keeping *everything* in `A_ub` costs one
    extra row per equality and keeps one dual vector whose indices match the
    blocks exactly.
    """
    A = sp.csc_matrix(problem.A)
    finite_lo = np.isfinite(problem.lo)
    finite_hi = np.isfinite(problem.hi)

    # upper rows:  A x <= hi        lower rows: -A x <= -lo
    A_ub = sp.vstack([A, -A], format="csc")
    b_ub = np.concatenate([np.where(finite_hi, problem.hi, 1e20),
                           np.where(finite_lo, -problem.lo, 1e20)])

    res = linprog(
        problem.c, A_ub=A_ub, b_ub=b_ub,
        bounds=list(zip(problem.xl, problem.xu)),
        method="highs", options={"time_limit": time_limit_s},
    )
    # Fold the split rows back: for a two-sided row only one side can be active.
    #
    # Guarded, because this solve can fail. Fixing a MILP's integer columns at
    # `round(x)` can leave the continuous problem infeasible when the solver
    # reported a value a hair off integral, and HiGHS then returns no marginals
    # at all -- a zero-dimensional array rather than an empty one. Unguarded the
    # slice raises, which took down a 300-cell sweep on its 69th cell.
    #
    # Duals are a diagnostic here, not a decision input: the tracking inner layer
    # does not price energy. Losing them for one replan is acceptable; losing the
    # run is not.
    marg = getattr(getattr(res, "ineqlin", None), "marginals", None)
    if marg is not None and np.ndim(marg) == 1 and len(marg) == 2 * problem.n_g:
        marg = np.asarray(marg, dtype=float)
        res.ineqlin.marginals = marg[:problem.n_g] + marg[problem.n_g:]
    elif getattr(res, "ineqlin", None) is not None:
        res.ineqlin.marginals = np.zeros(problem.n_g)
    return res
