"""Solver backend tests.

The sign of a dual is the highest-stakes convention in this project: the whole
hierarchy is coordinated by lambda, and getting its sign backwards would not
raise anything, it would silently make the plant do the opposite of the right
thing at every hour. So the dual tests here check against finite differences of
the optimal value rather than against a remembered convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from sfp.solvers import NLPBuilder, available_backends, get_backend


def _qp(c: float) -> "object":
    """min (x-1)^2 + (y-2)^2 s.t. x + y == c. Optimum f* = (c-3)^2 / 2."""
    b = NLPBuilder("qp")
    x = b.variable("x", 1, lb=-10.0, ub=10.0, x0=0.0)
    y = b.variable("y", 1, lb=-10.0, ub=10.0, x0=0.0)
    b.constraint("sum", x + y, equals=c)
    b.minimise((x - 1) ** 2 + (y - 2) ** 2)
    return b.build()


# --- registry ---------------------------------------------------------------


def test_ipopt_is_registered():
    assert "ipopt" in available_backends()
    assert get_backend("ipopt").name == "ipopt"


def test_unknown_backend_names_what_is_available():
    with pytest.raises(KeyError, match="unknown solver backend"):
        get_backend("no_such_solver")


# --- correctness ------------------------------------------------------------


def test_qp_reaches_the_analytic_optimum():
    sol = get_backend("ipopt").solve(_qp(5.0))
    assert sol.success
    # stationarity gives y = x + 1, and x + y = 5
    assert sol.value("x")[0] == pytest.approx(2.0, abs=1e-6)
    assert sol.value("y")[0] == pytest.approx(3.0, abs=1e-6)
    assert sol.f == pytest.approx(2.0, abs=1e-6)


def test_solution_reports_measured_diagnostics_not_just_a_status():
    sol = get_backend("ipopt").solve(_qp(5.0))
    assert sol.stats.kkt_residual < 1e-8
    assert sol.stats.constraint_violation < 1e-9
    assert sol.stats.iterations > 0
    assert sol.stats.wall_time_s > 0.0
    assert sol.stats.backend == "ipopt"


# --- the sign of the dual ---------------------------------------------------


def test_dual_of_minimisation_matches_finite_difference():
    """For `min f s.t. g == c`, CasADi's lam satisfies d f*/dc = -lam."""
    backend = get_backend("ipopt")
    c, eps = 5.0, 1e-4
    lam = backend.solve(_qp(c)).dual("sum")[0]
    fd = (backend.solve(_qp(c + eps)).f - backend.solve(_qp(c - eps)).f) / (2 * eps)
    assert fd == pytest.approx(-lam, abs=1e-4)
    assert fd == pytest.approx(c - 3.0, abs=1e-4)


def test_dual_of_maximisation_is_marginal_value():
    """For a problem built with `maximise`, lam IS the marginal value.

    This is the convention the planner's price signal depends on. `maximise`
    stores `-J`, and the two sign flips cancel: d J*/dc = +lam.
    """
    def profit_problem(budget: float):
        b = NLPBuilder("max")
        p = b.variable("p", 1, lb=0.0, ub=10.0, x0=0.0)
        b.constraint("budget", p, lb=-1e9, ub=budget)
        b.maximise(3.0 * p)
        return b.build()

    backend = get_backend("ipopt")
    sol = backend.solve(profit_problem(4.0))
    lam = sol.dual("budget")[0]

    assert sol.value("p")[0] == pytest.approx(4.0, abs=1e-6)
    # profit J* = 3 * budget, so dJ*/d(budget) = 3
    eps = 1e-4
    fd = (-backend.solve(profit_problem(4.0 + eps)).f
          + backend.solve(profit_problem(4.0 - eps)).f) / (2 * eps)
    assert fd == pytest.approx(3.0, abs=1e-3)
    assert lam == pytest.approx(3.0, abs=1e-4), (
        "lam must be the marginal value of the constraint for a maximisation; "
        "a flipped sign here inverts the planner's price signal silently"
    )


def test_inactive_constraint_has_zero_dual():
    b = NLPBuilder("slack")
    p = b.variable("p", 1, lb=0.0, ub=2.0, x0=0.0)
    b.constraint("budget", p, lb=-1e9, ub=100.0)  # never binds
    b.maximise(3.0 * p)
    sol = get_backend("ipopt").solve(b.build())
    assert sol.value("p")[0] == pytest.approx(2.0, abs=1e-6)
    assert abs(sol.dual("budget")[0]) < 1e-6


# --- functions --------------------------------------------------------------


def test_nlp_functions_agree_with_the_expressions():
    nlp = _qp(5.0)
    fns = nlp.functions()
    v = np.array([2.0, 3.0])
    assert fns.n_x == 2 and fns.n_g == 1
    assert fns.f(v) == pytest.approx(2.0)
    assert fns.g(v) == pytest.approx(np.array([5.0]))
    assert fns.grad_f(v) == pytest.approx(np.array([2.0, 2.0]))
    assert fns.jac_g(v) == pytest.approx(np.array([[1.0, 1.0]]))
    assert fns.hess_lag(v, np.array([0.0]), 1.0) == pytest.approx(np.diag([2.0, 2.0]))


def test_functions_are_compiled_once():
    nlp = _qp(5.0)
    assert nlp.functions() is nlp.functions()


def test_kkt_residual_is_zero_at_the_solution():
    nlp = _qp(5.0)
    sol = get_backend("ipopt").solve(nlp)
    assert nlp.functions().kkt_residual(sol.x, sol.lam_g) < 1e-8


# --- builder bookkeeping ----------------------------------------------------


def test_named_blocks_address_the_right_slices():
    b = NLPBuilder("blocks")
    b.variable("a", 3, lb=0.0, ub=1.0, x0=0.25)
    b.variable("bb", 2, lb=-1.0, ub=1.0, x0=0.0)
    b.minimise(0.0)
    nlp = b.build()
    assert nlp.n_x == 5
    assert nlp.var_blocks["a"].slice == slice(0, 3)
    assert nlp.var_blocks["bb"].slice == slice(3, 5)
    assert nlp.x0[:3] == pytest.approx(0.25)


def test_bounds_broadcast_and_are_length_checked():
    b = NLPBuilder("bounds")
    b.variable("v", 3, lb=[0.0, 1.0, 2.0], ub=5.0)
    b.minimise(0.0)
    nlp = b.build()
    assert nlp.lbx == pytest.approx(np.array([0.0, 1.0, 2.0]))
    assert nlp.ubx == pytest.approx(np.array([5.0, 5.0, 5.0]))

    with pytest.raises(ValueError, match="expected 3 values"):
        bad = NLPBuilder("bad")
        bad.variable("v", 3, lb=[0.0, 1.0])


def test_initial_guess_is_clipped_into_the_box():
    b = NLPBuilder("clip")
    b.variable("v", 2, lb=0.0, ub=1.0, x0=5.0)
    b.minimise(0.0)
    assert b.build().x0 == pytest.approx(np.array([1.0, 1.0]))


def test_duplicate_names_are_rejected():
    b = NLPBuilder()
    b.variable("v")
    with pytest.raises(ValueError, match="duplicate variable block"):
        b.variable("v")

    b2 = NLPBuilder()
    v = b2.variable("v")
    b2.constraint("c", v, equals=0.0)
    with pytest.raises(ValueError, match="duplicate constraint block"):
        b2.constraint("c", v, equals=1.0)


def test_constraint_needs_bounds_and_rejects_both_forms():
    b = NLPBuilder()
    v = b.variable("v")
    with pytest.raises(ValueError, match="no bounds"):
        b.constraint("c", v)
    with pytest.raises(ValueError, match="either .equals. or"):
        b.constraint("d", v, equals=0.0, ub=1.0)


def test_build_requires_an_objective_and_variables():
    b = NLPBuilder()
    b.variable("v")
    with pytest.raises(ValueError, match="no objective"):
        b.build()

    b2 = NLPBuilder()
    b2.minimise(0.0)
    with pytest.raises(ValueError, match="no decision variables"):
        b2.build()


def test_unknown_block_name_lists_the_known_ones():
    sol = get_backend("ipopt").solve(_qp(5.0))
    with pytest.raises(KeyError, match="known blocks: sum"):
        sol.dual("not_a_constraint")
    with pytest.raises(KeyError, match="known blocks: x, y"):
        sol.value("not_a_variable")


# --- failure behaviour ------------------------------------------------------


def test_infeasible_problem_reports_failure_without_raising():
    """A receding-horizon controller must be able to fall back, not crash."""
    b = NLPBuilder("infeasible")
    v = b.variable("v", 1, lb=0.0, ub=1.0, x0=0.5)
    b.constraint("impossible", v, equals=5.0)  # outside the variable's own box
    b.minimise(v**2)
    sol = get_backend("ipopt", max_iter=50).solve(b.build())
    assert not sol.success
    assert sol.stats.status != ""
    assert np.all(np.isfinite(sol.x))


# --- the solver cache -------------------------------------------------------


def test_solver_cache_holds_only_the_current_problem():
    """A dict keyed on `id(nlp)` leaked and could return the wrong solver.

    Every replan builds a new NLP, so every replan added an entry that was never
    evicted, each holding a CasADi solver and its whole expression graph -- a
    three-day run reached 7.9 GB and was still climbing. And because the dict
    held the solver but not the NLP, an NLP could be collected and a new one
    allocated at the same address; weather is baked into the graph as constants,
    so a collision would have silently solved a different problem and reported
    success.
    """
    backend = get_backend("ipopt")
    first = _qp(5.0)
    backend.solve(first)
    assert backend._cached is not None
    assert backend._cached[0] is first

    second = _qp(6.0)
    backend.solve(second)
    assert backend._cached[0] is second, "cache must not keep the old problem"


def test_repeated_solves_of_one_problem_reuse_the_solver():
    backend = get_backend("ipopt")
    nlp = _qp(5.0)
    backend.solve(nlp)
    solver = backend._cached[1]
    backend.solve(nlp)
    assert backend._cached[1] is solver


def test_many_problems_do_not_accumulate_solvers():
    """The leak showed up as unbounded growth across a receding horizon."""
    backend = get_backend("ipopt")
    for c in range(20):
        backend.solve(_qp(float(c)))
    # exactly one entry, whatever the history
    assert backend._cached is not None
    assert isinstance(backend._cached, tuple) and len(backend._cached) == 2
