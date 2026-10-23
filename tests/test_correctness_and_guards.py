"""
Comprehensive correctness and guard tests for SilentFail Hardener.
Covers:
1. Retries with stable idempotency keys preventing second refunds on timeout.
2. Orders with pre-existing partial refunds where subsequent refund fails before mutation (not falsely claimed).
3. Read-back verification across timeout-after-success and empty-200 (both applied and not applied).
4. Full rejection path where all candidates are rejected and no winner is selected.
5. Incomplete / invalid metrics guard rejection.
"""

import pytest
from src.silentfail.models import Scenario, ExpectedOutcome, FaultConfig, PatchSpec
from src.silentfail.ledger import WorldState, CausalLedger
from src.silentfail.tools import ToolRegistry, FaultInjector
from src.silentfail.agent import NaiveRefundAgent
from src.silentfail.patches import PatchedToolRegistry
from src.silentfail.runner import EpisodeRunner, BenchmarkAggregator, VerifierAndSelector


def test_retry_with_idempotency_key_prevents_second_refund():
    """
    Test 1: Stable operation key prevents second refund mutation when agent retries after timeout.
    """
    initial = {"orders": [{"id": "5001", "status": "delivered", "total": 100.0, "refunded_amount": 0.0}]}
    ws = WorldState(initial)
    ledger = CausalLedger(scenario_id="sc_test_idemp_retry")
    
    # Fault on first call to issue_refund: timeout after real mutation
    injector = FaultInjector([FaultConfig(tool="issue_refund", call_index=0, type="timeout_after_success")])
    raw_tools = ToolRegistry(world_state=ws, ledger=ledger, fault_injector=injector)
    patch = PatchSpec(
        patch_id="patch_idempotency_key",
        name="Idempotency Key",
        type="wrapper.idempotency_key",
        description="Stable key reuse"
    )
    patched_tools = PatchedToolRegistry(raw_tools, patch=patch)

    # First attempt: completes in world state, returns timeout error to agent
    resp1 = patched_tools.issue_refund("5001", 50.0, op_index=0)
    assert resp1.get("error") == "timeout"
    assert ws.orders["5001"].refunded_amount == 50.0
    assert len(ws.refunds) == 1

    # Retry of the same logical operation (op_index=0): must NOT create a second refund
    resp2 = patched_tools.issue_refund("5001", 50.0, op_index=0)
    assert resp2.get("status") == "ok"
    # Ground truth world state must strictly have only 1 refund record of $50
    assert ws.orders["5001"].refunded_amount == 50.0
    assert len(ws.refunds) == 1


def test_existing_partial_refund_unapplied_new_refund_not_claimed_success():
    """
    Test 2: Order already has an existing partial refund. A subsequent refund fails BEFORE mutation.
    Wrapper must not falsely claim success just because cumulative refunded_amount > 0.
    """
    initial = {
        "orders": [{"id": "5002", "status": "partially_refunded", "total": 100.0, "refunded_amount": 40.0}],
        "refunds": [{"id": "rf_prior_1", "order_id": "5002", "amount": 40.0}]
    }
    ws = WorldState(initial)
    ledger = CausalLedger(scenario_id="sc_test_existing_partial")
    
    # Empty 200 with effect NOT applied (failed before mutation)
    injector = FaultInjector([FaultConfig(tool="issue_refund", call_index=0, type="empty_200", effect_applied=False)])
    raw_tools = ToolRegistry(world_state=ws, ledger=ledger, fault_injector=injector)
    patch = PatchSpec(
        patch_id="patch_read_back_verify",
        name="Read Back",
        type="wrapper.read_back_verify",
        description="Verify specific refund"
    )
    patched_tools = PatchedToolRegistry(raw_tools, patch=patch)

    # Agent requests a new $30 refund
    resp = patched_tools.issue_refund("5002", 30.0, op_index=0)
    
    # Wrapper must NOT claim success based on prior 40.0 refund
    assert resp.get("status") != "ok"
    assert resp.get("verified_by_readback") is False
    assert ws.orders["5002"].refunded_amount == 40.0
    assert len(ws.refunds) == 1


def test_read_back_verification_empty_200_applied_vs_not_applied():
    """
    Test 3: Read-back verification handles both empty-200 applied and empty-200 not applied.
    """
    patch = PatchSpec(
        patch_id="patch_read_back_verify",
        name="Read Back",
        type="wrapper.read_back_verify",
        description="Verify specific refund"
    )

    # Case A: Empty 200 with effect APPLIED
    ws_a = WorldState({"orders": [{"id": "6001", "status": "delivered", "total": 60.0, "refunded_amount": 0.0}]})
    reg_a = ToolRegistry(ws_a, CausalLedger(), FaultInjector([FaultConfig(tool="issue_refund", call_index=0, type="empty_200", effect_applied=True)]))
    patched_a = PatchedToolRegistry(reg_a, patch=patch)
    res_a = patched_a.issue_refund("6001", 60.0, op_index=0)
    assert res_a.get("status") == "ok"
    assert res_a.get("verified_by_readback") is True
    assert ws_a.orders["6001"].refunded_amount == 60.0

    # Case B: Empty 200 with effect NOT APPLIED
    ws_b = WorldState({"orders": [{"id": "6002", "status": "delivered", "total": 60.0, "refunded_amount": 0.0}]})
    reg_b = ToolRegistry(ws_b, CausalLedger(), FaultInjector([FaultConfig(tool="issue_refund", call_index=0, type="empty_200", effect_applied=False)]))
    patched_b = PatchedToolRegistry(reg_b, patch=patch)
    res_b = patched_b.issue_refund("6002", 60.0, op_index=0)
    assert res_b.get("status") == "error"
    assert res_b.get("verified_by_readback") is False
    assert ws_b.orders["6002"].refunded_amount == 0.0


def test_all_candidates_rejected_returns_no_winner():
    """
    Test 4: If all candidates fail the guards, winner must be None and reasons recorded.
    """
    baseline_metrics = {
        "correct_completion": [0.60, 0.45, 0.75],
        "legitimate_completion_rate": [0.90, 0.70, 0.98],
        "duplicate_action_rate": [0.20, 0.10, 0.35],
        "silent_failure_rate": [0.15, 0.07, 0.28],
        "missed_action_rate": [0.10, 0.05, 0.22],
        "incorrect_completion_rate": [0.40, 0.25, 0.55],
        "over_suppression_rate": [0.0, 0.0, 0.05],
        "mean_steps": 4.0
    }

    # Two bad candidates: one causes severe regression, another causes massive cost overrun
    candidate_specs = [
        PatchSpec(patch_id="bad_1", name="Bad 1", type="control.refuse_all", description="refuse all"),
        PatchSpec(patch_id="bad_2", name="Bad 2", type="wrapper.retry_policy", description="costly")
    ]
    candidate_metrics = {
        "bad_1": {
            "correct_completion": [0.10, 0.03, 0.25],  # Fails regression guard
            "legitimate_completion_rate": [0.0, 0.0, 0.05],  # Fails legitimate completion guard
            "duplicate_action_rate": [0.0, 0.0, 0.05],
            "silent_failure_rate": [0.15, 0.07, 0.28],
            "missed_action_rate": [0.60, 0.45, 0.75],  # Fails missed action guard
            "incorrect_completion_rate": [0.90, 0.75, 0.97],
            "over_suppression_rate": [0.50, 0.35, 0.65],  # Fails over-suppression guard
            "mean_steps": 4.0
        },
        "bad_2": {
            "correct_completion": [0.60, 0.45, 0.75],
            "legitimate_completion_rate": [0.90, 0.70, 0.98],
            "duplicate_action_rate": [0.20, 0.10, 0.35],
            "silent_failure_rate": [0.15, 0.07, 0.28],
            "missed_action_rate": [0.10, 0.05, 0.22],
            "incorrect_completion_rate": [0.40, 0.25, 0.55],
            "over_suppression_rate": [0.0, 0.0, 0.05],
            "mean_steps": 15.0  # Fails cost guard (> 1.8 * 4.0)
        }
    }

    verifier = VerifierAndSelector(tolerance=0.02, max_cost_factor=1.8)
    winner, race_summary = verifier.select_winner(baseline_metrics, candidate_metrics, candidate_specs)

    assert winner is None
    assert race_summary["bad_1"]["eligible"] is False
    assert len(race_summary["bad_1"]["rejection_reasons"]) > 0
    assert race_summary["bad_2"]["eligible"] is False
    assert any("Cost Guard" in r for r in race_summary["bad_2"]["rejection_reasons"])


def test_guard_rejects_incomplete_metrics():
    """
    Test 5: Incomplete candidate metrics are rejected by guard rather than defaulted to zero.
    """
    baseline_metrics = {
        "correct_completion": [0.60, 0.45, 0.75],
        "legitimate_completion_rate": [0.90, 0.70, 0.98],
        "duplicate_action_rate": [0.20, 0.10, 0.35],
        "silent_failure_rate": [0.15, 0.07, 0.28],
        "missed_action_rate": [0.10, 0.05, 0.22],
        "incorrect_completion_rate": [0.40, 0.25, 0.55],
        "over_suppression_rate": [0.0, 0.0, 0.05],
        "mean_steps": 4.0
    }
    incomplete_metrics = {
        "correct_completion": [0.65, 0.50, 0.78],
        # missing legitimate_completion_rate, duplicate_action_rate, etc.
    }
    verifier = VerifierAndSelector()
    is_eligible, reasons = verifier.evaluate_guards(baseline_metrics, incomplete_metrics)
    assert is_eligible is False
    assert any("Incomplete Metrics Guard" in r for r in reasons)
