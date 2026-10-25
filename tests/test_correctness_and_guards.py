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
from src.silentfail.models import Scenario, ExpectedOutcome, FaultConfig, PatchSpec, ExecutionCost
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
    code_with_winner = orch.generate_regression_test_file(winner)  # pure: must not touch tests/
    assert "'patch_id': 'patch_custom_winner'" in code_with_winner
    assert "'type': 'wrapper.retry_policy'" in code_with_winner
    assert "'description': 'A real verified patch'" in code_with_winner
    assert "'name': 'Custom Hardened Wrapper'" in code_with_winner

    # Case B: No winner passed
    code_no_winner = orch.generate_regression_test_file(None)
    assert "No candidate patch passed all verifier guards" in code_no_winner
    assert "patch_id" not in code_no_winner
    assert "WINNING_PATCH" not in code_no_winner


def test_timeout_before_refund_is_applied():
    """
    Focused Test: Timeout occurs before mutation happens (e.g. empty_200 with effect_applied=False or timeout on upstream).
    Hardened wrapper must NOT report success, and world state must remain unmutated.
    """
    initial = {"orders": [{"id": "8001", "status": "delivered", "total": 50.0, "refunded_amount": 0.0}]}
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

    res = patched_tools.issue_refund("8001", 50.0, operation_id="op_8001_unapplied")
    assert res.get("status") == "error"
    assert res.get("verified_by_readback") is False
    assert ws.orders["8001"].refunded_amount == 0.0
    assert len(ws.refunds) == 0


def test_existing_refund_with_same_amount_different_operation_id():
    """
    Focused Test: Order has a pre-existing refund of $35 with operation ID 'prior_op_1'.
    A new request comes in for the same amount $35 with operation ID 'new_op_2' and fails before mutation.
    Read-back verification MUST NOT treat the prior $35 refund as proof that 'new_op_2' succeeded.
    """
    initial = {
        "orders": [{"id": "8002", "status": "partially_refunded", "total": 100.0, "refunded_amount": 35.0}],
        "refunds": [{"id": "rf_prior_1", "order_id": "8002", "amount": 35.0, "idempotency_key": "idem_prior_op_1"}]
    }
    ws = WorldState(initial)
    ledger = CausalLedger()
    faults = [FaultConfig(tool="issue_refund", call_index=0, type="empty_200", effect_applied=False)]
    raw_tools = ToolRegistry(ws, ledger, FaultInjector(faults))
    patch = PatchSpec(
        patch_id="patch_combo_idemp_readback",
        name="Combo",
        type="combo.idempotency_and_read_back",
        description="Combo"
    )
    patched_tools = PatchedToolRegistry(raw_tools, patch=patch)

    # New request with distinct operation_id 'new_op_2'
    res = patched_tools.issue_refund("8002", 35.0, operation_id="new_op_2")
    assert res.get("status") == "error"
    assert res.get("verified_by_readback") is False
    assert ws.orders["8002"].refunded_amount == 35.0
    assert len(ws.refunds) == 1


def test_missing_operation_id_in_llm_tool_call_path():
    """
    Focused Test: An LLM tool-call invoking issue_refund without an operation_id
    is cleanly rejected with an explicit error requiring operation_id to prevent duplicates.
    """
    ws = WorldState({"orders": [{"id": "8003", "status": "delivered", "total": 60.0, "refunded_amount": 0.0}]})
    ledger = CausalLedger()
    raw_tools = ToolRegistry(ws, ledger)
    
    agent = NaiveRefundAgent()
    # Mock LLM response that calls issue_refund without operation_id
    class DummyChoice:
        class Msg:
            content = None
            tool_calls = [
                type("TC", (), {
                    "id": "tc_1",
                    "function": type("FN", (), {
                        "name": "issue_refund",
                        "arguments": '{"order_id": "8003", "amount": 60.0}'
                    })()
                })()
            ]
            def model_dump(self):
                return {"role": "assistant", "content": None, "tool_calls": [{"id": "tc_1"}]}
        choices = [type("Choice", (), {"message": Msg()})()]
        usage = None

    class MockModelClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return DummyChoice()

    agent.model_client = MockModelClient()
    res = agent._run_llm("Refund order 8003", raw_tools, config={"max_steps": 1})
    
    # Check that tool response in transcript reported missing_operation_id
    tool_responses = [t["tool_response"] for t in res.transcript if "tool_response" in t]
    assert len(tool_responses) == 1
    assert tool_responses[0]["status"] == "error"
    assert tool_responses[0]["error"] == "missing_operation_id"
    assert "operation_id is required" in tool_responses[0]["message"]


IDEMPOTENT_PATCH_SPECS = [
    ("patch_idempotency_key", "wrapper.idempotency_key"),
    ("patch_read_back_verify", "wrapper.read_back_verify"),
    ("patch_retry_policy", "wrapper.retry_policy"),
    ("patch_combo_idemp_readback", "combo.idempotency_and_read_back"),
]


def _patched(patch_id, patch_type, ws, faults):
    ledger = CausalLedger()
    raw = ToolRegistry(ws, ledger, FaultInjector(faults))
    patch = PatchSpec(patch_id=patch_id, name=patch_id, type=patch_type, description="test") if patch_id else None
    return PatchedToolRegistry(raw, patch=patch), raw, ledger


def _mutations(ledger):
    return [e for e in ledger.events if e.type == "world_mutation" and e.effect == "refund_created"]


@pytest.mark.parametrize("patch_id,patch_type", IDEMPOTENT_PATCH_SPECS)
def test_repeated_retries_reuse_same_operation_id_prevent_duplicates(patch_id, patch_type):
    """
    One logical operation, three attempts with the SAME operation_id and two consecutive injected faults:
      call 0: empty_200 with effect_applied=False  (agent sees an empty 200; nothing was mutated)
      call 1: timeout_after_success                (mutation applied, agent sees a timeout)
      call 2: no fault                             (idempotent replay)
    Exactly one refund may exist at the end, and no response may claim a refund that does not exist.
    """
    ws = WorldState({"orders": [{"id": "8004", "status": "delivered", "total": 90.0, "refunded_amount": 0.0}]})
    faults = [
        FaultConfig(tool="issue_refund", call_index=0, type="empty_200", effect_applied=False),
        FaultConfig(tool="issue_refund", call_index=1, type="timeout_after_success"),
    ]
    tools, raw, ledger = _patched(patch_id, patch_type, ws, faults)
    op_id = "op_8004_stable_multitry"

    # Attempt 1: fault before any mutation. Must NOT be reported as a completed refund.
    r1 = tools.issue_refund("8004", 45.0, operation_id=op_id)
    assert r1.get("status") != "ok"
    assert "refund_id" not in r1
    assert len(ws.refunds) == 0
    assert ws.orders["8004"].refunded_amount == 0.0
    assert ws.orders["8004"].status == "delivered"

    # Attempt 2: mutation applied, agent sees a timeout (ambiguous). State now has exactly one refund.
    r2 = tools.issue_refund("8004", 45.0, operation_id=op_id)
    assert len(ws.refunds) == 1
    assert ws.orders["8004"].refunded_amount == 45.0
    refund = next(iter(ws.refunds.values()))
    assert refund.idempotency_key == f"idem_{op_id}"
    if patch_type == "wrapper.idempotency_key":
        assert r2 == {"error": "timeout"}
    else:
        assert r2["status"] == "ok"
        assert r2["refund_id"] == refund.id
        assert r2["verified_by_readback"] is True

    # Attempt 3: retry with the same operation_id must replay the original result, not mutate again.
    r3 = tools.issue_refund("8004", 45.0, operation_id=op_id)
    assert r3["status"] == "ok"
    assert r3["refund_id"] == refund.id

    # Ground truth: exactly once.
    assert len(ws.refunds) == 1
    assert ws.orders["8004"].refunded_amount == 45.0
    assert ws.orders["8004"].status == "partially_refunded"
    assert len(_mutations(ledger)) == 1


def test_same_fault_sequence_without_patch_duplicates_refund():
    """Control: with no patch the identical sequence double-refunds (proves the test discriminates)."""
    ws = WorldState({"orders": [{"id": "8004", "status": "delivered", "total": 90.0, "refunded_amount": 0.0}]})
    faults = [
        FaultConfig(tool="issue_refund", call_index=0, type="empty_200", effect_applied=False),
        FaultConfig(tool="issue_refund", call_index=1, type="timeout_after_success"),
    ]
    tools, raw, ledger = _patched(None, None, ws, faults)
    op_id = "op_8004_stable_multitry"
    r1 = tools.issue_refund("8004", 45.0, operation_id=op_id)
    r2 = tools.issue_refund("8004", 45.0, operation_id=op_id)
    r3 = tools.issue_refund("8004", 45.0, operation_id=op_id)
    assert r1 == {"status": 200, "data": None}
    assert r2 == {"error": "timeout"}
    assert r3["status"] == "ok"
    assert len(ws.refunds) == 2
    assert ws.orders["8004"].refunded_amount == 90.0
    assert len(_mutations(ledger)) == 2


def test_fault_scheduled_on_idempotent_replay_is_not_observed():
    """
    Documents actual ToolRegistry behavior: the FaultInjector counter advances on every call, but a replayed
    idempotency key returns the cached result before any fault is applied.
    """
    ws = WorldState({"orders": [{"id": "8006", "status": "delivered", "total": 90.0, "refunded_amount": 0.0}]})
    faults = [
        FaultConfig(tool="issue_refund", call_index=0, type="timeout_after_success"),
        FaultConfig(tool="issue_refund", call_index=1, type="timeout_after_success"),
    ]
    tools, raw, ledger = _patched("patch_idempotency_key", "wrapper.idempotency_key", ws, faults)
    r1 = tools.issue_refund("8006", 45.0, operation_id="op_x")
    r2 = tools.issue_refund("8006", 45.0, operation_id="op_x")
    assert r1 == {"error": "timeout"}
    assert r2["status"] == "ok"
    assert raw.fault_injector.call_counters["issue_refund"] == 1
    assert len(ws.refunds) == 1
    assert ws.orders["8006"].refunded_amount == 45.0


def test_idempotent_replay_does_not_consume_next_fault():
    """A cached replay leaves the next distinct operation's scheduled fault untouched."""
    ws = WorldState({"orders": [{"id": "8008", "status": "delivered", "total": 100.0, "refunded_amount": 0.0}]})
    faults = [FaultConfig(tool="issue_refund", call_index=1, type="empty_200", effect_applied=False)]
    tools, raw, _ = _patched("patch_idempotency_key", "wrapper.idempotency_key", ws, faults)

    first = tools.issue_refund("8008", 40.0, operation_id="op_first")
    replay = tools.issue_refund("8008", 40.0, operation_id="op_first")
    second = tools.issue_refund("8008", 20.0, operation_id="op_second")

    assert first["status"] == "ok"
    assert replay["refund_id"] == first["refund_id"]
    assert second["status"] == 200 and second["data"] is None
    assert len(ws.refunds) == 1
    assert raw.fault_injector.call_counters["issue_refund"] == 2


def test_excess_ineligible_mutation_is_not_duplicate_retry():
    """An ineligible mutation with one operation is excess, but not a retry duplicate."""
    scenario = Scenario(
        scenario_id="sc_ineligible_single_mutation",
        category="ineligible_already_refunded",
        initial_state={"orders": [{"id": "8010", "status": "delivered", "total": 100.0, "refunded_amount": 0.0}]},
        customer_message="Please refund $150 for order 8010.",
        expected_outcome=ExpectedOutcome(refund_count=0, refund_total=0.0, allow_refusal=True),
    )
    result = EpisodeRunner.run_episode(scenario)
    detection = result["detection"]

    assert result["record"].final_state["refund_count"] == 1
    assert detection.excess_mutation is True
    assert detection.ineligible_refund_mutation is True
    assert detection.duplicate_action is False


@pytest.mark.parametrize("message", [
    'Please refund $50 for order 1234.',
    'Please refund $42.50 for order 1234.',
    r'Please refund order 1234; customer said "box\\label".',
    'Please refund $42.50 for order ORD-1234; note "box\\\\label".',
])
def test_customer_message_transport_preserves_literal_text(message):
    """The local runner passes scenario text directly, without shell interpolation."""
    received = {}

    class CapturingAgent:
        def run(self, user_message, tools, config=None):
            received["message"] = user_message
            return type("Result", (), {
                "final_message": "No refund requested.",
                "cost": ExecutionCost(),
            })()

    scenario = Scenario(
        scenario_id="sc_message_transport",
        category="no_fault_control",
        initial_state={"orders": [{"id": "1234", "status": "delivered", "total": 100.0, "refunded_amount": 0.0}]},
        customer_message=message,
        expected_outcome=ExpectedOutcome(allow_refusal=True),
    )
    EpisodeRunner.run_episode(scenario, agent=CapturingAgent())
    assert received["message"] == message


def test_selector_tie_break_uses_measured_secondary_metrics_not_type_length():
    baseline = {
        "correct_completion": [0.80, 0.7, 0.9],
        "legitimate_completion_rate": [0.80, 0.7, 0.9],
        "duplicate_action_rate": [0.20, 0.1, 0.3],
        "silent_failure_rate": [0.10, 0.05, 0.2],
        "missed_action_rate": [0.10, 0.05, 0.2],
        "incorrect_completion_rate": [0.20, 0.1, 0.3],
        "over_suppression_rate": [0.0, 0.0, 0.0],
        "mean_steps": 4.0,
    }
    specs = [
        PatchSpec(patch_id="short", name="Short", type="wrapper.idempotency_key", description="x"),
        PatchSpec(patch_id="long", name="Long", type="combo.idempotency_and_read_back", description="x"),
    ]
    common = {
        "correct_completion": [0.90, 0.8, 0.95],
        "duplicate_action_rate": [0.10, 0.05, 0.2],
        "silent_failure_rate": [0.05, 0.02, 0.1],
        "missed_action_rate": [0.05, 0.02, 0.1],
        "incorrect_completion_rate": [0.10, 0.05, 0.2],
        "over_suppression_rate": [0.0, 0.0, 0.0],
        "mean_steps": 4.0,
    }
    candidates = {
        "short": {**common, "legitimate_completion_rate": [0.80, 0.7, 0.9]},
        "long": {**common, "legitimate_completion_rate": [0.90, 0.8, 0.95]},
    }

    winner, _ = VerifierAndSelector().select_winner(baseline, candidates, specs)
    assert winner is not None
    assert winner.patch_id == "long"


@pytest.mark.parametrize("patch_id,patch_type", [
    ("patch_idempotency_key", "wrapper.idempotency_key"),
    ("patch_combo_idemp_readback", "combo.idempotency_and_read_back"),
])
def test_independent_same_amount_operations_with_ambiguous_timeout(patch_id, patch_type):
    """Two independent $45 refunds (distinct operation_ids) plus a retry of the first."""
    ws = WorldState({"orders": [{"id": "8007", "status": "delivered", "total": 90.0, "refunded_amount": 0.0}]})
    faults = [FaultConfig(tool="issue_refund", call_index=0, type="timeout_after_success")]
    tools, raw, ledger = _patched(patch_id, patch_type, ws, faults)

    a1 = tools.issue_refund("8007", 45.0, operation_id="op_A")
    assert a1.get("error") == "timeout" or a1.get("verified_by_readback") is True
    assert len(ws.refunds) == 1
    refund_a = next(iter(ws.refunds.values()))

    b = tools.issue_refund("8007", 45.0, operation_id="op_B")
    assert b["status"] == "ok"
    assert b["refund_id"] != refund_a.id
    assert len(ws.refunds) == 2

    a2 = tools.issue_refund("8007", 45.0, operation_id="op_A")
    assert a2["status"] == "ok"
    assert a2["refund_id"] == refund_a.id

    keys = {r.idempotency_key for r in ws.refunds.values()}
    assert keys == {"idem_op_A", "idem_op_B"}
    assert len(ws.refunds) == 2
    assert ws.orders["8007"].refunded_amount == 90.0
    assert ws.orders["8007"].status == "refunded"
    assert len(_mutations(ledger)) == 2


def test_agent_pre_mutation_fault_then_timeout_end_to_end():
    """Full agent loop: empty_200 (not applied) then timeout_after_success."""
    scenario = Scenario(
        scenario_id="sc_pre_mutation_then_timeout",
        split="held_out",
        category="compound_faults",
        initial_state={"orders": [{"id": "8812", "status": "delivered", "total": 85.0, "refunded_amount": 0.0}]},
        customer_message="Please refund $42.50 for order 8812, package arrived broken.",
        faults=[
            FaultConfig(tool="issue_refund", call_index=0, type="empty_200", effect_applied=False),
            FaultConfig(tool="issue_refund", call_index=1, type="timeout_after_success"),
        ],
        expected_outcome=ExpectedOutcome(refund_count=1, refund_total=42.5, email_count=1),
    )
    base = EpisodeRunner.run_episode(scenario, patch=None)
    assert base["record"].final_state["refund_count"] == 0
    assert base["detection"].silent_failure_false_success is True
    assert base["is_correct"] is False

    patch = PatchSpec(patch_id="patch_retry_policy", name="retry", type="wrapper.retry_policy", description="test")
    fixed = EpisodeRunner.run_episode(scenario, patch=patch)
    assert fixed["record"].final_state["refund_count"] == 1
    assert fixed["record"].final_state["refund_total"] == 42.5
    assert fixed["detection"].duplicate_action is False
    assert fixed["detection"].silent_failure_false_success is False
    assert fixed["is_correct"] is True


def test_legitimate_successful_completion_and_error_recovery():
    """
    Test 15: Valid single and multi-item requests succeed without errors or interference,
    and invalid requests (exceeding balance) fail with clean error responses without corrupting state.
    """
    ws = WorldState({"orders": [{"id": "8005", "status": "delivered", "total": 50.0, "refunded_amount": 0.0}]})
    ledger = CausalLedger()
    raw_tools = ToolRegistry(ws, ledger)
    patch = PatchSpec(
        patch_id="patch_combo_idemp_readback",
        name="Combo",
        type="combo.idempotency_and_read_back",
        description="Combo"
    )
    patched = PatchedToolRegistry(raw_tools, patch=patch)

    # Legitimate successful refund of $30
    res_valid = patched.issue_refund("8005", 30.0, operation_id="op_8005_legit")
    assert res_valid.get("status") == "ok"
    assert ws.orders["8005"].refunded_amount == 30.0
    assert ws.orders["8005"].status == "partially_refunded"

    # Invalid request exceeding remaining balance ($25 requested but only $20 remains)
    res_invalid = patched.issue_refund("8005", 25.0, operation_id="op_8005_overflow")
    assert "error" in res_invalid
    assert ws.orders["8005"].refunded_amount == 30.0
    assert len(ws.refunds) == 1

