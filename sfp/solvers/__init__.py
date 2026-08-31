"""Optimisation backends.

Everything in this project that solves an NLP goes through `SolverBackend`. The
planner and the NMPC build problems with `NLPBuilder`, hand them to whichever
backend is configured, and read the answer back by name. Nothing upstream of
this package imports CasADi's solver interface directly.

That indentation exists for one reason: at M7 a custom NLP solver replaces IPOPT
inside the inner NMPC, and the swap has to be a configuration change rather than
a rewrite. Building the interface first -- before the planner that will be its
first user -- is what keeps that promise cheap.
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
