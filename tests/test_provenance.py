"""Unit tests for the GroundedSignal / SignalSource provenance primitive (#822)."""

import dataclasses

import pytest

from synapsekit import GroundedSignal, SignalSource
from synapsekit.provenance import GroundedSignal as ProvGroundedSignal


def test_external_signal_is_grounded():
    signal = GroundedSignal(
        value=0.92,
        source=SignalSource.EXTERNAL_OVERRIDE,
        provenance={"evaluator": "human_review", "eval_id": "eval-4471"},
    )
    assert signal.grounded is True
    assert signal.value == 0.92
    assert signal.provenance["evaluator"] == "human_review"


def test_self_reported_signal_is_not_grounded():
    signal = GroundedSignal(
        value=0.86,
        source=SignalSource.SELF_REPORTED,
        provenance={"origin": "bid_estimate_fallback"},
    )
    assert signal.grounded is False


def test_grounded_is_derived_read_only_not_settable():
    signal = GroundedSignal.self_reported(0.5)
    # Frozen dataclass: cannot mutate the source after the fact, and `grounded`
    # is a derived property with no backing field — neither can be set, so a
    # self-reported signal can never be relabelled as grounded.
    with pytest.raises(dataclasses.FrozenInstanceError):
        signal.source = SignalSource.EXTERNAL_OVERRIDE  # type: ignore[misc]
    # `grounded` is a derived property with no backing field on a frozen+slots
    # dataclass; assigning it is rejected (never silently accepted).
    with pytest.raises(Exception):
        signal.grounded = True  # type: ignore[misc]
    assert isinstance(type(signal).grounded, property)


def test_factory_helpers_pack_provenance_kwargs():
    external = GroundedSignal.external(0.7, evaluator="judge_model", eval_id="e1")
    assert external.source is SignalSource.EXTERNAL_OVERRIDE
    assert external.grounded is True
    assert external.provenance == {"evaluator": "judge_model", "eval_id": "e1"}

    internal = GroundedSignal.self_reported(0.7, origin="output_field")
    assert internal.source is SignalSource.SELF_REPORTED
    assert internal.grounded is False


def test_value_and_source_are_coerced():
    signal = GroundedSignal(value="0.5", source="external")  # type: ignore[arg-type]
    assert signal.value == 0.5
    assert signal.source is SignalSource.EXTERNAL_OVERRIDE

    assert GroundedSignal(value=1, source="self").source is SignalSource.SELF_REPORTED
    assert GroundedSignal(value=1, source="ungrounded").grounded is False
    assert GroundedSignal(value=1, source="grounded").grounded is True


def test_provenance_is_defensively_copied():
    provenance = {"k": "v"}
    signal = GroundedSignal(value=0.1, source=SignalSource.SELF_REPORTED, provenance=provenance)
    provenance["k"] = "mutated"
    assert signal.provenance["k"] == "v"


def test_unknown_source_raises():
    with pytest.raises(ValueError, match="Unknown signal source"):
        GroundedSignal(value=0.1, source="definitely-not-a-source")  # type: ignore[arg-type]


def test_to_dict_round_trip_shape():
    signal = GroundedSignal.external(0.9, evaluator="human")
    assert signal.to_dict() == {
        "value": 0.9,
        "source": "external_override",
        "grounded": True,
        "provenance": {"evaluator": "human"},
    }


def test_top_level_export_is_the_same_class():
    assert GroundedSignal is ProvGroundedSignal
