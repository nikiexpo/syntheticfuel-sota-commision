"""Perfect-foresight oracle -- the upper bound the controller is measured against.

Not a controller anyone could build. It is the same economic planner given the
*true* weather instead of a forecast, which makes it a bound on what any
forecast-driven strategy could have achieved, and lets the report quote **regret**:

    regret = J_oracle - J_controller

Beating greedy says little if greedy is bad; sitting close to the oracle says
the remaining loss is forecast error rather than bad scheduling.

Two deliberate limits, both of which make this a *conservative* bound -- the
true optimum is at least this good, so reported regret is understated:

**It re-plans.** Its horizon is the planner's, so it sees the truth for the next
seven days and nothing beyond. A genuine open-loop optimum over a ten-day run
needs a 240-hour solve, which does not converge. So this is perfect foresight
over a rolling seven-day window, not over everything.

**It is still a relaxation.** Commitment variables are relaxed and rounded as
the planner's are, and IPOPT returns a local solution to a non-convex problem.
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

        `ControlContext.metadata['truth']` is set by the simulator. Missing, this
        raises rather than falling back to the forecast: an oracle on a forecast
        is the planner under another name, reporting a regret of ~zero.
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
