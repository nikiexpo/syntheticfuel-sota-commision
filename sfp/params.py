"""Parameters that carry their own provenance.

The challenge brief asks entrants to distinguish "supplied data, measured data,
assumptions, and invented values". Rather than maintaining that as prose that
rots, every parameter in this project is declared in YAML with a mandatory
`provenance` tag, and `docs/ASSUMPTIONS.md` is generated from those tags. If a
number exists in the model, it is in the ledger.

Provenance levels, weakest evidence last:

    supplied    given by the challenge brief or by the user as a design input
    measured    from a real dataset (PVGIS irradiance, weather reanalysis)
    literature  taken from a cited paper, with the citation recorded
    assumed     engineering judgement, defensible, order-of-magnitude right
    invented    placeholder with no external basis -- must be flagged in the writeup

Usage:

    p = load_params("pv")
    p.eta_ref                 # -> 0.205, a plain float
    p.meta("eta_ref").source  # -> "Fraunhofer ISE Photovoltaics Report 2024"

For the truth simulator we need the plant to disagree with the controller's
model, so `ParamSet.perturb()` returns a copy with multiplicative noise applied
to exactly those parameters marked `uncertain: true`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

PARAM_DIR = Path(__file__).parent / "models" / "params"

PROVENANCE_LEVELS = ("supplied", "measured", "literature", "assumed", "invented")


class ParamError(ValueError):
    """Raised when a parameter file is malformed or a lookup fails."""


@dataclass(frozen=True)
class Param:
    """A single parameter value plus everything needed to defend it."""

    name: str
    value: float
    units: str
    provenance: str
    source: str = ""
    note: str = ""
    uncertain: bool = False
    sigma: float = 0.05  # relative std-dev used by `perturb`
    bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.provenance not in PROVENANCE_LEVELS:
            raise ParamError(
                f"parameter {self.name!r} has provenance {self.provenance!r}; "
                f"expected one of {PROVENANCE_LEVELS}"
            )
        if self.provenance in ("literature", "measured") and not self.source:
            raise ParamError(
                f"parameter {self.name!r} is marked {self.provenance!r} but has no source"
            )


@dataclass
class ParamSet:
    """A named group of parameters, attribute-accessible as plain floats."""

    name: str
    description: str = ""
    params: dict[str, Param] = field(default_factory=dict)

    def __getattr__(self, key: str) -> float:
        # only called when normal attribute lookup fails, so no recursion risk
        # for the real fields above
        try:
            return self.__dict__["params"][key].value
        except KeyError:
            raise AttributeError(
                f"{self.__dict__.get('name', '?')!r} has no parameter {key!r}"
            ) from None

    def __contains__(self, key: str) -> bool:
        return key in self.params

    def __iter__(self) -> Iterator[Param]:
        return iter(self.params.values())

    def meta(self, key: str) -> Param:
        """The full record for one parameter, including its citation."""
        if key not in self.params:
            raise ParamError(f"{self.name!r} has no parameter {key!r}")
        return self.params[key]

    def as_dict(self) -> dict[str, float]:
        """Plain {name: value} mapping."""
        return {k: p.value for k, p in self.params.items()}

    def override(self, **kwargs: float) -> "ParamSet":
        """A copy with some values replaced -- used for design sweeps and siting."""
        out = copy.deepcopy(self)
        for key, value in kwargs.items():
            if key not in out.params:
                raise ParamError(f"cannot override unknown parameter {key!r} in {self.name!r}")
            old = out.params[key]
            out.params[key] = Param(
                name=old.name,
                value=float(value),
                units=old.units,
                provenance="supplied",
                source="design input / sweep override",
                note=old.note,
                uncertain=old.uncertain,
                sigma=old.sigma,
                bounds=old.bounds,
            )
        return out

    def perturb(self, rng: np.random.Generator, scale: float = 1.0) -> "ParamSet":
        """A copy with lognormal noise on every parameter marked `uncertain`.

        This is what creates plant/model mismatch: the truth simulator runs on
        the perturbed set, the controller on the nominal one. Without it the
        closed-loop results are circular and mean nothing.

        `scale` multiplies every sigma, so scale=0 recovers the nominal set and
        scale=2 doubles the mismatch for stress tests.
        """
        out = copy.deepcopy(self)
        for key, p in self.params.items():
            if not p.uncertain or scale == 0.0:
                continue
            factor = float(np.exp(rng.normal(0.0, p.sigma * scale)))
            value = p.value * factor
            if p.bounds is not None:
                value = float(np.clip(value, p.bounds[0], p.bounds[1]))
            out.params[key] = Param(
                name=p.name,
                value=value,
                units=p.units,
                provenance=p.provenance,
                source=p.source,
                note=(p.note + " [perturbed for truth simulator]").strip(),
                uncertain=p.uncertain,
                sigma=p.sigma,
                bounds=p.bounds,
            )
        return out


def _parse_param(name: str, raw: Any) -> Param:
    if not isinstance(raw, dict):
        raise ParamError(
            f"parameter {name!r} must be a mapping with at least 'value', 'units' "
            f"and 'provenance'; got {type(raw).__name__}"
        )
    missing = {"value", "units", "provenance"} - set(raw)
    if missing:
        raise ParamError(f"parameter {name!r} is missing {sorted(missing)}")
    bounds = raw.get("bounds")
    return Param(
        name=name,
        value=float(raw["value"]),
        units=str(raw["units"]),
        provenance=str(raw["provenance"]),
        source=str(raw.get("source", "")),
        note=str(raw.get("note", "")),
        uncertain=bool(raw.get("uncertain", False)),
        sigma=float(raw.get("sigma", 0.05)),
        bounds=(float(bounds[0]), float(bounds[1])) if bounds else None,
    )


def load_params(name: str, directory: Path | None = None) -> ParamSet:
    """Load one parameter group, e.g. `load_params("pv")` -> sfp/models/params/pv.yaml."""
    directory = directory or PARAM_DIR
    path = directory / f"{name}.yaml"
    if not path.exists():
        raise ParamError(f"no parameter file at {path}")
    with path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    raw_params = doc.get("params") or {}
    return ParamSet(
        name=doc.get("name", name),
        description=doc.get("description", ""),
        params={k: _parse_param(k, v) for k, v in raw_params.items()},
    )


def load_all(directory: Path | None = None) -> dict[str, ParamSet]:
    """Every parameter group in the directory, keyed by file stem."""
    directory = directory or PARAM_DIR
    return {p.stem: load_params(p.stem, directory) for p in sorted(directory.glob("*.yaml"))}


def provenance_summary(sets: dict[str, ParamSet]) -> dict[str, int]:
    """Count parameters by provenance level -- the headline for the writeup."""
    counts = dict.fromkeys(PROVENANCE_LEVELS, 0)
    for ps in sets.values():
        for p in ps:
            counts[p.provenance] += 1
    return counts
