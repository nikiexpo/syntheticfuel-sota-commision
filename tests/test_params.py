"""The parameter/provenance system, which is also the assumptions ledger.

The brief asks submissions to distinguish supplied, measured, assumed and
invented values. That claim is only worth making if it is enforced, so these
tests enforce it: a parameter cannot claim to come from the literature without
naming a source, and the ledger cannot silently drift from the models.
"""

from __future__ import annotations

import numpy as np
import pytest

from sfp.params import (
    PROVENANCE_LEVELS,
    ParamError,
    ParamSet,
    load_all,
    load_params,
    provenance_summary,
)


def test_every_parameter_file_loads():
    sets = load_all()
    assert set(sets) >= {"pv", "battery", "aggregate", "economics"}
    for name, ps in sets.items():
        assert len(ps.params) > 0, name


def test_every_parameter_declares_a_known_provenance():
    for name, ps in load_all().items():
        for p in ps:
            assert p.provenance in PROVENANCE_LEVELS, f"{name}.{p.name}"


def test_cited_parameters_name_a_source():
    """`literature` and `measured` values must be traceable."""
    for name, ps in load_all().items():
        for p in ps:
            if p.provenance in ("literature", "measured"):
                assert p.source.strip(), f"{name}.{p.name} claims {p.provenance} without a source"


def test_every_parameter_has_units():
    for name, ps in load_all().items():
        for p in ps:
            assert p.units, f"{name}.{p.name} has no units"


def test_no_invented_parameters_without_explanation():
    """Invented values are allowed, but must say what they stand in for."""
    for name, ps in load_all().items():
        for p in ps:
            if p.provenance == "invented":
                assert p.note.strip(), f"{name}.{p.name} is invented with no note"


def test_provenance_summary_counts_everything():
    sets = load_all()
    counts = provenance_summary(sets)
    total = sum(len(ps.params) for ps in sets.values())
    assert sum(counts.values()) == total


def test_attribute_access_returns_floats():
    pv = load_params("pv")
    assert isinstance(pv.capacity_kwp, float)
    assert pv.eta_inverter > 0.9


def test_unknown_parameter_raises():
    pv = load_params("pv")
    with pytest.raises(AttributeError):
        _ = pv.definitely_not_a_parameter


def test_override_replaces_value_and_retags_provenance():
    pv = load_params("pv")
    modified = pv.override(capacity_kwp=1234.0)
    assert modified.capacity_kwp == 1234.0
    assert modified.meta("capacity_kwp").provenance == "supplied"
    # the original is untouched
    assert pv.capacity_kwp != 1234.0


def test_override_rejects_unknown_parameters():
    with pytest.raises(ParamError):
        load_params("pv").override(not_a_real_parameter=1.0)


def test_perturb_only_moves_uncertain_parameters():
    pv = load_params("pv")
    rng = np.random.default_rng(0)
    perturbed = pv.perturb(rng)
    for p in pv:
        if p.uncertain:
            continue
        assert perturbed.meta(p.name).value == p.value, p.name


def test_perturb_actually_perturbs():
    pv = load_params("pv")
    rng = np.random.default_rng(1)
    perturbed = pv.perturb(rng)
    uncertain = [p.name for p in pv if p.uncertain]
    assert uncertain, "expected some parameters to be marked uncertain"
    assert any(perturbed.meta(n).value != pv.meta(n).value for n in uncertain)


def test_perturb_with_zero_scale_is_the_identity():
    pv = load_params("pv")
    perturbed = pv.perturb(np.random.default_rng(2), scale=0.0)
    for p in pv:
        assert perturbed.meta(p.name).value == p.value


def test_perturb_respects_declared_bounds():
    pv = load_params("pv")
    for _ in range(50):
        perturbed = pv.perturb(np.random.default_rng(None), scale=5.0)
        for p in pv:
            bounds = p.bounds
            if bounds is None:
                continue
            value = perturbed.meta(p.name).value
            assert bounds[0] <= value <= bounds[1], p.name


def test_malformed_parameter_is_rejected():
    from sfp.params import _parse_param

    with pytest.raises(ParamError):
        _parse_param("x", {"value": 1.0, "units": "-"})  # no provenance
    with pytest.raises(ParamError):
        _parse_param("x", {"value": 1.0, "units": "-", "provenance": "vibes"})
    with pytest.raises(ParamError):
        _parse_param("x", 42.0)  # not a mapping


def test_assumptions_ledger_renders_and_covers_every_parameter():
    from sfp.report import assumptions

    text = assumptions.render()
    sets = load_all()
    for name, ps in sets.items():
        assert f"`{name}`" in text
        for p in ps:
            assert f"`{p.name}`" in text, f"{name}.{p.name} missing from the ledger"
