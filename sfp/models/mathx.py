"""Math that works on both numpy floats and CasADi symbolics.

Every reduced-order model in `sfp.models` is written exactly once against this
module. The same source is then used three ways:

    simulator   numeric numpy, perturbed parameters, the "truth"
    estimator   CasADi symbolics inside an MHE problem
    controller  CasADi symbolics inside the NMPC / planner

Writing the model twice is the classic way to get a digital twin that quietly
disagrees with its own controller, so we do not do that.

Two flavours of the non-smooth operators are provided:

    fmax / fmin / clip          exact, kinked   -- fine for the simulator
    smooth_max / smooth_min     C-infinity      -- for anything an NLP differentiates

The smooth variants converge to the exact ones as `eps -> 0`; `eps` is in the
units of the quantity being limited, so pass something meaningful (e.g. 1e-3 W).
"""

from __future__ import annotations

import casadi as ca
import numpy as np

_SYM_TYPES = (ca.SX, ca.MX, ca.DM)


def is_sym(x) -> bool:
    """True if `x` is a CasADi expression rather than a plain number/array."""
    return isinstance(x, _SYM_TYPES)


def any_sym(*args) -> bool:
    """True if any argument is a CasADi expression."""
    return any(is_sym(a) for a in args)


# --- elementwise transcendentals ------------------------------------------
def exp(x):
    return ca.exp(x) if is_sym(x) else np.exp(x)


def log(x):
    return ca.log(x) if is_sym(x) else np.log(x)


def sqrt(x):
    return ca.sqrt(x) if is_sym(x) else np.sqrt(x)


def tanh(x):
    return ca.tanh(x) if is_sym(x) else np.tanh(x)


def sin(x):
    return ca.sin(x) if is_sym(x) else np.sin(x)


def cos(x):
    return ca.cos(x) if is_sym(x) else np.cos(x)


def arccos(x):
    return ca.acos(x) if is_sym(x) else np.arccos(x)


def arcsin(x):
    return ca.asin(x) if is_sym(x) else np.arcsin(x)


def fabs(x):
    return ca.fabs(x) if is_sym(x) else np.abs(x)


def power(x, p):
    return x**p


# --- non-smooth limiters (exact) ------------------------------------------
# These three are the hottest functions in the whole project: a one-day
# simulation calls them several million times. They therefore test the operand
# types directly rather than going through `any_sym`, which builds a generator
# and calls `any()` on every invocation -- that indirection alone accounted for
# roughly a quarter of total runtime before it was removed. Behaviour is
# identical; only the dispatch is cheaper.
def fmax(a, b):
    if isinstance(a, _SYM_TYPES) or isinstance(b, _SYM_TYPES):
        return ca.fmax(a, b)
    return np.maximum(a, b)


def fmin(a, b):
    if isinstance(a, _SYM_TYPES) or isinstance(b, _SYM_TYPES):
        return ca.fmin(a, b)
    return np.minimum(a, b)


def clip(x, lo, hi):
    """Clamp `x` into [lo, hi]."""
    if isinstance(x, _SYM_TYPES) or isinstance(lo, _SYM_TYPES) or isinstance(hi, _SYM_TYPES):
        return ca.fmin(ca.fmax(x, lo), hi)
    return np.minimum(np.maximum(x, lo), hi)


def if_else(condition, if_true, if_false):
    """Branch that works symbolically.

    For symbolic `condition` this becomes a CasADi `if_else` node; numerically
    it is `np.where`. Prefer the smooth helpers below inside an NLP -- a hard
    branch gives the solver a discontinuous derivative and it will suffer.
    """
    if any_sym(condition, if_true, if_false):
        return ca.if_else(condition, if_true, if_false)
    return np.where(condition, if_true, if_false)


# --- smooth limiters (differentiable) -------------------------------------
def smooth_max(a, b, eps: float = 1e-6):
    """C-infinity approximation of max(a, b).

    Uses the numerically stable form  0.5*(a+b+sqrt((a-b)^2 + eps^2)), which
    never overflows the way a log-sum-exp does for large arguments.
    """
    d = a - b
    return 0.5 * (a + b + sqrt(d * d + eps * eps))


def smooth_min(a, b, eps: float = 1e-6):
    """C-infinity approximation of min(a, b)."""
    d = a - b
    return 0.5 * (a + b - sqrt(d * d + eps * eps))


def smooth_clip(x, lo, hi, eps: float = 1e-6):
    """C-infinity clamp into [lo, hi]."""
    return smooth_min(smooth_max(x, lo, eps), hi, eps)


def smooth_abs(x, eps: float = 1e-6):
    """C-infinity approximation of |x|, exact away from the origin."""
    return sqrt(x * x + eps * eps)


def softplus(x, eps: float = 1e-6):
    """C-infinity approximation of max(x, 0)."""
    return smooth_max(x, 0.0, eps)


def smooth_step(x, width: float = 1e-3):
    """Smooth 0->1 transition centred at x = 0, reaching ~0.96 at x = 2*width."""
    return 0.5 * (1.0 + tanh(x / width))


# --- array helpers ---------------------------------------------------------
def vertcat(*args):
    """Stack into a column vector, symbolic or numeric."""
    if any_sym(*args):
        return ca.vertcat(*args)
    return np.asarray([float(np.asarray(a).squeeze()) for a in args], dtype=float)


def zeros_like_state(x, n: int):
    """An n-vector of zeros matching the symbolic-ness of `x`."""
    return ca.SX.zeros(n) if is_sym(x) else np.zeros(n)
