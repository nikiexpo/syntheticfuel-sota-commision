"""IPOPT through CasADi -- the reference backend.

This is the backend everything is developed against, and at M7 it becomes the
control in the experiment: the custom NLP solver is compared against it on
identical instances for iterations, wall time, KKT residual and objective
agreement. Keeping it plain matters more than making it fast.

Two settings are deliberate rather than inherited defaults:

`print_level = 0`. A receding-horizon planner solves hundreds of times per run,
and IPOPT's banner would bury everything else. Failures still surface -- they are
recorded in `SolveStats.status` and the caller decides what to do.

Warm starting is *off* by default. IPOPT's warm start needs consistent primal and
bound multipliers to be sound, and feeding it only a primal guess can leave it
worse off than a cold start. Passing `x0` still seeds the primal iterate, which is
the part that reliably helps on a receding horizon.
"""

from __future__ import annotations

import time

import casadi as ca
import numpy as np

from sfp.solvers.backend import NLP, Solution, SolveStats, SolverBackend, register_backend

#: IPOPT return strings that mean "this answer is usable".
_ACCEPTABLE = frozenset({"Solve_Succeeded", "Solved_To_Acceptable_Level"})


@register_backend
class IpoptBackend(SolverBackend):
    """CasADi's `nlpsol` with IPOPT."""

    name = "ipopt"
    provides_duals = True

    def __init__(
        self,
        *,
        max_iter: int = 500,
        tol: float = 1e-6,
        acceptable_tol: float = 1e-4,
        print_level: int = 0,
        warm_start: bool = False,
        linear_solver: str | None = None,
        **options,
    ) -> None:
        super().__init__(
            max_iter=max_iter, tol=tol, acceptable_tol=acceptable_tol,
            print_level=print_level, warm_start=warm_start,
            linear_solver=linear_solver, **options,
        )
        self._cache: dict[int, ca.Function] = {}

    # --- construction -----------------------------------------------------
    def _solver_options(self) -> dict:
        o = self.options
        ipopt = {
            "max_iter": int(o["max_iter"]),
            "tol": float(o["tol"]),
            "acceptable_tol": float(o["acceptable_tol"]),
            "print_level": int(o["print_level"]),
            "sb": "yes",
        }
        if o.get("linear_solver"):
            ipopt["linear_solver"] = o["linear_solver"]
        if o.get("warm_start"):
            ipopt.update(
                warm_start_init_point="yes",
                warm_start_bound_push=1e-9,
                warm_start_mult_bound_push=1e-9,
            )
        extra = {k: v for k, v in o.items()
                 if k not in {"max_iter", "tol", "acceptable_tol", "print_level",
                              "warm_start", "linear_solver"}}
        ipopt.update(extra)
        return {"ipopt": ipopt, "print_time": False}

    def _solver_for(self, nlp: NLP) -> ca.Function:
        """Build (once per problem structure) the CasADi solver object.

        Cached on `id(nlp)` because a receding-horizon controller rebuilds the
        same structure at every tick and constructing the solver dominates the
        cost of solving it. The planner reuses one `NLP` and shifts its data, so
        this cache hits on every tick after the first.
        """
        key = id(nlp)
        if key not in self._cache:
            self._cache[key] = ca.nlpsol(
                f"solver_{nlp.name}", "ipopt",
                {"x": nlp.x, "f": nlp.f, "g": nlp.g},
                self._solver_options(),
            )
        return self._cache[key]

    # --- solving ----------------------------------------------------------
    def solve(self, nlp: NLP, *, x0=None, lam_g0=None) -> Solution:
        started = time.perf_counter()
        solver = self._solver_for(nlp)

        guess = np.asarray(nlp.x0 if x0 is None else x0, dtype=float).ravel()
        if guess.size != nlp.n_x:
            raise ValueError(f"x0 has {guess.size} entries, expected {nlp.n_x}")
        # IPOPT will push an initial point onto its bounds anyway; doing it here
        # means a caller's warm start from a previous tick cannot be infeasible
        # merely because a bound moved.
        guess = np.clip(guess, nlp.lbx, nlp.ubx)

        args = {
            "x0": guess,
            "lbx": nlp.lbx, "ubx": nlp.ubx,
            "lbg": nlp.lbg, "ubg": nlp.ubg,
        }
        if lam_g0 is not None and self.options.get("warm_start"):
            args["lam_g0"] = np.asarray(lam_g0, dtype=float).ravel()

        stats = SolveStats()
        try:
            result = solver(**args)
            raw = solver.stats()
            stats.status = str(raw.get("return_status", "unknown"))
            stats.iterations = int(raw.get("iter_count", 0))
            stats.success = bool(raw.get("success", False)) or stats.status in _ACCEPTABLE
            x = np.asarray(result["x"]).astype(float).ravel()
            lam_g = np.asarray(result["lam_g"]).astype(float).ravel()
            lam_x = np.asarray(result["lam_x"]).astype(float).ravel()
        except RuntimeError as exc:
            # A failed solve is an expected event on a receding horizon -- the
            # caller falls back to the previous plan rather than crashing the
            # run. It must never be silent, though, so it is recorded.
            stats.status = "exception"
            stats.success = False
            stats.message = str(exc)
            x = guess
            lam_g = np.zeros(nlp.n_g)
            lam_x = np.zeros(nlp.n_x)

        return self._finish(nlp, x, lam_g, lam_x, stats, started)
