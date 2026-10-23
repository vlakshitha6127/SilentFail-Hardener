"""
Comprehensive correctness and guard tests for SilentFail Hardener.
Covers:
1. Retries with stable idempotency keys preventing second refunds on timeout.
2. Orders with pre-existing partial refunds where subsequent refund fails before mutation (not falsely claimed).
3. Read-back verification across timeout-after-success and empty-200 (both applied and not applied).
4. Two equal-value legitimate partial refunds receive distinct operation IDs and both succeed.
5. Overlapping operations with the same amount receive different operation IDs and do not collide.
6. Timeout without mutation returns proper failure and does not claim success.
7. Full rejection path where all candidates are rejected and no winner is selected.
8. Incomplete / invalid metrics guard rejection.
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
    resp1 = patched_tools.issue_refund("5001", 50.0, operation_id="op_5001_req0")
    assert resp1.get("error") == "timeout"
    assert ws.orders["5001"].refunded_amount == 50.0
    assert len(ws.refunds) == 1

    # Retry of the same logical operation (same operation_id): must NOT create a second refund
    resp2 = patched_tools.issue_refund("5001", 50.0, operation_id="op_5001_req0")
    assert resp2.get("status") == "ok"
    # Ground truth world state must strictly have only 1 refund record of $50
    assert ws.orders["5001"].refunded_amount == 50.0
    assert len(ws.refunds) == 1


def test_two_equal_value_legitimate_partial_refunds():
    """
    Test 2: Two legitimate refund requests for the same order and same amount ($40 and $40)
    receive different operation IDs, distinct idempotency keys, and both succeed.
    """
    initial = {"orders": [{"id": "5003", "status": "delivered", "total": 100.0, "refunded_amount": 0.0}]}
    ws = WorldState(initial)
    ledger = CausalLedger(scenario_id="sc_two_equal_refunds")
    raw_tools = ToolRegistry(world_state=ws, ledger=ledger)
    patch = PatchSpec(
        patch_id="patch_combo_idemp_readback",
        name="Combo",
        type="combo.idempotency_and_read_back",
        description="Combo"
    )
    patched_tools = PatchedToolRegistry(raw_tools, patch=patch)

    # First legitimate refund of $40 (operation 1)
    res1 = patched_tools.issue_refund("5003", 40.0, operation_id="op_5003_part1")
    assert res1.get("status") == "ok"
    assert ws.orders["5003"].refunded_amount == 40.0
    assert len(ws.refunds) == 1

    # Second legitimate refund of equal amount $40 (operation 2)
    res2 = patched_tools.issue_refund("5003", 40.0, operation_id="op_5003_part2")
    assert res2.get("status") == "ok"
    assert ws.orders["5003"].refunded_amount == 80.0
    assert len(ws.refunds) == 2


def test_overlapping_operations_with_same_amount():
    """
    Test 3: Overlapping operations on the same order with identical amounts
    without explicit operation_id receive distinct generated operation IDs and do not collide.
    """
    initial = {"orders": [{"id": "5004", "status": "delivered", "total": 100.0, "refunded_amount": 0.0}]}
    ws = WorldState(initial)
    ledger = CausalLedger(scenario_id="sc_overlap_ops")
    raw_tools = ToolRegistry(world_state=ws, ledger=ledger)
    patch = PatchSpec(
        patch_id="patch_idempotency_key",
        name="Idempotency Key",
        type="wrapper.idempotency_key",
        description="Key"
    )
    patched_tools = PatchedToolRegistry(raw_tools, patch=patch)

    # Calling without explicit operation_id should automatically allocate distinct sequential operation IDs
    res1 = patched_tools.issue_refund("5004", 30.0)
    assert res1.get("status") == "ok"

    res2 = patched_tools.issue_refund("5004", 30.0)
    assert res2.get("status") == "ok"

    assert len(ws.refunds) == 2
    assert ws.orders["5004"].refunded_amount == 60.0


def test_timeout_after_success_resolved_by_readback_and_combo():
    """
    Test 4: On timeout after success, read_back_verify and combo wrappers accurately verify
    the specific refund by its operation key and report success.
    """
    initial = {"orders": [{"id": "5005", "status": "delivered", "total": 70.0, "refunded_amount": 0.0}]}
    ws = WorldState(initial)
    ledger = CausalLedger()
    faults = [FaultConfig(tool="issue_refund", call_index=0, type="timeout_after_success")]
    raw_tools = ToolRegistry(ws, ledger, FaultInjector(faults))
    patch = PatchSpec(
        patch_id="patch_read_back_verify",
        name="Read Back",
        type="wrapper.read_back_verify",
        description="Verify"
    )
    patched_tools = PatchedToolRegistry(raw_tools, patch=patch)

    res = patched_tools.issue_refund("5005", 70.0, operation_id="op_5005_main")
    assert res.get("status") == "ok"
    assert res.get("verified_by_readback") is True
    assert ws.orders["5005"].refunded_amount == 70.0


def test_timeout_without_mutation():
    """
    Test 5: On timeout where mutation did NOT occur (empty_200 with effect_applied=False),
    read-back returns honest error and does NOT claim success.
    """
    initial = {"orders": [{"id": "5006", "status": "delivered", "total": 70.0, "refunded_amount": 0.0}]}
    ws = WorldState(initial)
    ledger = CausalLedger()
    faults = [FaultConfig(tool="issue_refund", call_index=0, type="empty_200", effect_applied=False)]
    raw_tools = ToolRegistry(ws, ledger, FaultInjector(faults))
    patch = PatchSpec(
        patch_id="patch_read_back_verify",
        name="Read Back",
        type="wrapper.read_back_verify",
        description="Verify"
    )
    patched_tools = PatchedToolRegistry(raw_tools, patch=patch)

    res = patched_tools.issue_refund("5006", 70.0, operation_id="op_5006_main")
    assert res.get("status") == "error"
    assert res.get("verified_by_readback") is False
    assert ws.orders["5006"].refunded_amount == 0.0


def test_existing_partial_refund_unapplied_new_refund_not_claimed_success():
    """
    Test 6: Order already has an existing partial refund. A subsequent refund fails BEFORE mutation.
    Wrapper must not falsely claim success just because cumulative refunded_amount > 0.
    """
    initial = {
        "orders": [{"id": "5002", "status": "partially_refunded", "total": 100.0, "refunded_amount": 40.0}],
        "refunds": [{"id": "rf_prior_1", "order_id": "5002", "amount": 40.0, "idempotency_key": "prior_key"}]
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
    resp = patched_tools.issue_refund("5002", 30.0, operation_id="op_5002_new")
    
    # Wrapper must NOT claim success based on prior 40.0 refund
    assert resp.get("status") != "ok"
    assert resp.get("verified_by_readback") is False
    assert ws.orders["5002"].refunded_amount == 40.0
    assert len(ws.refunds) == 1


def test_all_candidates_rejected_returns_no_winner():
    """
    Test 7: If all candidates fail the guards, winner must be None and reasons recorded.
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
    Test 8: Incomplete candidate metrics are rejected by guard rather than defaulted to zero.
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
    }
    verifier = VerifierAndSelector()
    is_eligible, reasons = verifier.evaluate_guards(baseline_metrics, incomplete_metrics)
    assert is_eligible is False
    assert any("Incomplete Metrics Guard" in r for r in reasons)


def test_llm_tool_call_operation_identity_contract():
    """
    Test 9: LLM tool-calling path follows the documented operation-identity contract.
    Retries that supply the same operation_id reuse the stable key.
    Independent calls with different operation_ids receive distinct keys.
    """
    initial = {"orders": [{"id": "7001", "status": "delivered", "total": 100.0, "refunded_amount": 0.0}]}
    ws = WorldState(initial)
    ledger = CausalLedger()
    raw_tools = ToolRegistry(ws, ledger)
    patch = PatchSpec(
        patch_id="patch_idempotency_key",
        name="Idempotency Key",
        type="wrapper.idempotency_key",
        description="Key"
    )
    patched_tools = PatchedToolRegistry(raw_tools, patch=patch)

    # LLM issues first refund with explicit operation_id 'call_1'
    res1 = patched_tools.issue_refund("7001", 20.0, operation_id="call_1")
    assert res1.get("status") == "ok"
    key1 = ws.refunds[res1["refund_id"]].idempotency_key

    # LLM retries with identical operation_id 'call_1'
    res1_retry = patched_tools.issue_refund("7001", 20.0, operation_id="call_1")
    assert res1_retry.get("status") == "ok"
    # Mutation must NOT duplicate
    assert len(ws.refunds) == 1

    # LLM issues independent second refund with different operation_id 'call_2'
    res2 = patched_tools.issue_refund("7001", 20.0, operation_id="call_2")
    assert res2.get("status") == "ok"
    key2 = ws.refunds[res2["refund_id"]].idempotency_key

    assert len(ws.refunds) == 2
    assert key1 != key2


def test_generated_regression_test_handles_winner_and_no_winner():
    """
    Test 10: Regression test generator uses actual selected winner,
    and returns a clean notice without claiming a patch passed when winner is None.
    """
    from src.silentfail.orchestrator import Orchestrator
    orch = Orchestrator()

    # Case A: Real winner
    winner = PatchSpec(
        patch_id="patch_custom_winner",
        name="Custom Hardened Wrapper",
        type="wrapper.retry_policy",
        description="A real verified patch"
    )
    code_with_winner = orch.generate_regression_test_file(winner)
    assert 'patch_id="patch_custom_winner"' in code_with_winner
    assert 'type="wrapper.retry_policy"' in code_with_winner

    # Case B: No winner passed
    code_no_winner = orch.generate_regression_test_file(None)
    assert "No candidate patch passed all 4 verifier guards" in code_no_winner
    assert "patch_id=" not in code_no_winner
