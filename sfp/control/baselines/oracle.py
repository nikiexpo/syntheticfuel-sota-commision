"""Perfect-foresight oracle -- the upper bound the controller is measured against.

Not a controller anyone could build. It is the same economic planner given the
*true* weather instead of a forecast, which makes it a bound on what any
forecast-driven strategy could have achieved, and lets the report quote **regret**:

    regret = J_oracle - J_controller

A bound is worth more than another baseline. Beating greedy says little if greedy
is bad; sitting close to the oracle says the remaining loss is forecast error
rather than bad scheduling, and that is the honest way to describe how good a
controller is.

What kind of bound this actually is
-----------------------------------
Two limits, both deliberate, and both of which make this a *conservative* bound
-- the true optimum is at least this good, so real regret is at least what gets
reported.

**The oracle re-plans.** It is not solved once over the whole run. Its horizon is
the planner's horizon, so at any moment it sees the truth for the next seven days
and nothing beyond. A genuine open-loop optimum over a ten-day run would need a
240-hour solve, which does not converge (see
`bookkeeping/04_PLANNER_TRACTABILITY.md`). So this is "perfect foresight over a
seven-day rolling window", not "perfect foresight over everything".

**It is still a relaxation, solved to a local optimum.** The commitment variables
are relaxed and rounded exactly as the planner's are, and IPOPT returns a local
solution to a non-convex problem. Nothing here proves global optimality.

Both caveats mean the gap to the true optimum is understated, never overstated.
Stated plainly because an "oracle" that is quietly not one would make every regret
number in the report wrong in the flattering direction.
"""

from __future__ import annotations

from sfp.control.base import ControlContext
from sfp.control.planner import EconomicPlanner


class PerfectForesightOracle(EconomicPlanner):
    """The economic planner, handed the truth instead of a forecast."""

    name = "oracle"
    description = (
        "Perfect-foresight bound: the economic planner solved against the true "
        "weather over a rolling seven-day window. Not achievable; bounds regret."
    )

    def reset(self, context: ControlContext) -> None:
        """Swap the degraded forecast for the truth before planning starts.

        `ControlContext.metadata['truth']` is set by the simulator. If it is
        missing this raises rather than silently falling back to the forecast --
        an oracle running on a forecast is not an oracle, it is the planner under
        a different name, and it would report a regret of approximately zero and
        look like a triumph.
        """
        truth = context.metadata.get("truth")
        if truth is None:
            raise ValueError(
                "PerfectForesightOracle needs the true weather in "
                "ControlContext.metadata['truth']; refusing to run against a "
                "forecast and report the result as a bound"
            )
        context = ControlContext(
            plant=context.plant,
            site=context.site,
            dt_s=context.dt_s,
            horizon_s=context.horizon_s,
            forecast=truth,
            economics=context.economics,
            metadata=dict(context.metadata),
        )
        super().reset(context)
