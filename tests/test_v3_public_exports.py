"""Public import surfaces for the v3 feature modules."""


def test_evaluation_features_are_exported() -> None:
    from synapsekit.evaluation import (
        FittedCalibration,
        RulerGroup,
        RulerScorer,
        SimulationHarness,
        calibrate,
        fit_calibration,
    )
    from synapsekit.evaluation.ruler import RulerGroup as GroupImplementation
    from synapsekit.evaluation.ruler import RulerScorer as ScorerImplementation
    from synapsekit.evaluation.ruler import calibrate as calibrate_implementation
    from synapsekit.evaluation.simulation import SimulationHarness as HarnessImplementation

    assert RulerGroup is GroupImplementation
    assert RulerScorer is ScorerImplementation
    assert calibrate is calibrate_implementation
    assert SimulationHarness is HarnessImplementation
    assert callable(fit_calibration)
    assert FittedCalibration.__name__ == "FittedCalibration"


def test_agent_negotiation_features_are_exported() -> None:
    from synapsekit.agents import (
        BoundedNegotiationPolicy,
        Negotiation,
        NegotiationMessage,
        NegotiationRole,
        NegotiationTerms,
    )
    from synapsekit.agents.negotiation import Message, Terms
    from synapsekit.agents.negotiation import Negotiation as NegotiationImplementation

    assert Negotiation is NegotiationImplementation
    assert NegotiationMessage is Message
    assert NegotiationTerms is Terms
    assert BoundedNegotiationPolicy.__name__ == "BoundedPolicy"
    assert NegotiationRole.BUYER.value == "buyer"


def test_mcp_scanner_features_are_exported() -> None:
    from synapsekit.mcp import ScanReport, ToolScanBlockedError, ToolScanLimitError, scan_tools
    from synapsekit.mcp.scanner import ScanReport as ReportImplementation
    from synapsekit.mcp.scanner import ToolScanBlockedError as BlockedImplementation
    from synapsekit.mcp.scanner import ToolScanLimitError as LimitImplementation
    from synapsekit.mcp.scanner import scan_tools as scan_implementation

    assert ScanReport is ReportImplementation
    assert ToolScanBlockedError is BlockedImplementation
    assert ToolScanLimitError is LimitImplementation
    assert scan_tools is scan_implementation


def test_deliberation_features_are_exported() -> None:
    from synapsekit.llm import DeliberationResult, OpenAIChatAdapter, deliberate
    from synapsekit.llm.deliberation import DeliberationResult as ResultImplementation
    from synapsekit.llm.deliberation import OpenAIChatAdapter as AdapterImplementation
    from synapsekit.llm.deliberation import deliberate as deliberate_implementation

    assert DeliberationResult is ResultImplementation
    assert OpenAIChatAdapter is AdapterImplementation
    assert deliberate is deliberate_implementation
