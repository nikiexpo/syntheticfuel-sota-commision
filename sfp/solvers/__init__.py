"""Optimisation backends.

Everything that solves an NLP goes through `SolverBackend`: problems are built
with `NLPBuilder`, handed to whichever backend is configured, and read back by
name. Nothing upstream imports CasADi's solver interface directly, so swapping
in another solver is a configuration change rather than a rewrite.

The outer dispatch MILP is a different animal and goes through `solvers.lp`.
"""

from sfp.solvers.backend import (
    NLP,
    NLPBuilder,
    NLPFunctions,
    Solution,
    SolveStats,
    SolverBackend,
    available_backends,
    get_backend,
    register_backend,
)
from sfp.solvers.ipopt_backend import IpoptBackend

__all__ = [
    "NLP",
    "NLPBuilder",
    "NLPFunctions",
    "Solution",
    "SolveStats",
    "SolverBackend",
    "IpoptBackend",
    "available_backends",
    "get_backend",
    "register_backend",
]
