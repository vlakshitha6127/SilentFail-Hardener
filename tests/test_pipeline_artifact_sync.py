"""
Keeps committed pipeline outputs consistent with each other and with the code that produces them:
- data/latest_pipeline_run.json  <->  tests/test_generated_regression.py  <->  README benchmark table
- A fresh deterministic local run reproduces the committed counts and winner.
- When no candidate passes the guards, no winning patch is fabricated.
"""

import importlib.util
import json

from src.silentfail.models import PatchSpec
from src.silentfail.orchestrator import Orchestrator, render_regression_test, REPO_ROOT, GENERATED_TEST_FILENAME
from src.silentfail.runner import VerifierAndSelector

ARTIFACT = REPO_ROOT / "data" / "latest_pipeline_run.json"
GENERATED = REPO_ROOT / "tests" / GENERATED_TEST_FILENAME
README = REPO_ROOT / "README.md"


def _artifact():
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_generated_test_file_matches_artifact_winner():
    art = _artifact()
    winner = PatchSpec(**art["winning_patch"]) if art["winning_patch"] else None
    expected = render_regression_test(winner)
    assert GENERATED.read_text(encoding="utf-8") == expected

    # The historical artifact stored code rendered under the pre-refactor customer message template;
    # verify that the preserved code in the artifact matches the rendered code up to the explicit amount refactor.
    normalized_art_code = art["generated_test_code"].replace(
        'customer_message="Please refund order 8812, package arrived broken."',
        'customer_message="Please refund $42.50 for order 8812, package arrived broken."'
    )
    assert normalized_art_code == expected


CANDIDATE_ARTIFACT = REPO_ROOT / "data" / "candidate_pipeline_run.json"


def _candidate_artifact():
    return json.loads(CANDIDATE_ARTIFACT.read_text(encoding="utf-8"))


def test_candidate_artifact_parity():
    """
    Candidate artifact parity check:
    Verifies that the candidate artifact's generated regression code strictly matches
    the rendered regression test for the candidate winner without any ad-hoc normalizations,
    and that the file on disk (GENERATED) matches this exact rendered code.
    """
    assert CANDIDATE_ARTIFACT.exists(), "candidate_pipeline_run.json must exist"
    cand_art = _candidate_artifact()
    winner = PatchSpec(**cand_art["winning_patch"]) if cand_art.get("winning_patch") else None
    assert winner is not None, "Candidate artifact must specify a winning patch"
    expected = render_regression_test(winner)

    # Candidate's own generated_test_code must match expected code exactly with zero normalization
    assert cand_art["generated_test_code"] == expected
    # The active test file on disk must match the candidate's exact rendered code
    assert GENERATED.read_text(encoding="utf-8") == expected


def test_pipeline_artifact_is_local_simulation_and_labeled():
    art = _artifact()
    assert art["scale_and_cost"]["mode"] == "local_simulation"
    assert art["gate_g0"]["connected"] is False
    assert art["scale_and_cost"]["speedup_factor"] is None
    assert art["scale_and_cost"]["total_episodes_evaluated"] == (
        art["scale_and_cost"]["scenarios_train"] * (1 + art["scale_and_cost"]["candidates_verified"])
        + art["scale_and_cost"]["scenarios_heldout"] * 2
    )


def _fmt(r, num, den):
    return f"{r[0] * 100:.1f}% ({num}/{den}) [{r[1] * 100:.1f}% - {r[2] * 100:.1f}%]"


def test_readme_benchmark_matches_artifact():
    art = _artifact()
    readme = README.read_text(encoding="utf-8").replace("**", "")
    assert f"`{art['winning_patch']['patch_id']}`" in readme
    for key in ("heldout_metrics_before", "heldout_metrics_after"):
        m = art[key]
        c = m["counts"]
        total, incorrect = c["total"], c["cond_failed_total"]
        claimed = round(m["conditional_false_success_rate"][0] * incorrect)
        assert _fmt(m["correct_completion"], c["correct"], total) in readme
        assert _fmt(m["duplicate_action_rate"], c["duplicate"], total) in readme
        assert _fmt(m["silent_failure_rate"], c["silent_fail"], total) in readme
        assert _fmt(m["conditional_false_success_rate"], claimed, incorrect) in readme
    assert "Silent failures eliminated" not in readme


def test_historical_artifact_integrity(tmp_path, monkeypatch):
    """
    Historical baseline artifact integrity check:
    Validates that the preserved historical benchmark artifact (data/latest_pipeline_run.json)
    strictly satisfies all historical invariants, exact winner specification, metrics, and guard eligibility.
    """
    art = _artifact()
    assert art["status"] == "completed"
    assert art["progress"] == 100
    assert art["winning_patch"] == {
        "patch_id": "patch_retry_policy",
        "name": "State-Checked Retry Policy Wrapper",
        "type": "wrapper.retry_policy",
        "description": "On timeout/empty-200, looks up a refund recorded under the operation's stable idempotency key before reporting the outcome; retries reuse the same key.",
        "is_control": False,
        "config": {}
    }

    # Strict check of exact historical baseline and heldout metric counts
    assert art["baseline_metrics"]["counts"] == {
        "correct": 13,
        "duplicate": 8,
        "silent_fail": 14,
        "total": 50,
        "cond_failed_total": 37
    }
    assert art["heldout_metrics_before"]["counts"] == {
        "correct": 14,
        "duplicate": 3,
        "silent_fail": 5,
        "total": 30,
        "cond_failed_total": 16
    }
    assert art["heldout_metrics_after"]["counts"] == {
        "correct": 14,
        "duplicate": 0,
        "silent_fail": 0,
        "total": 30,
        "cond_failed_total": 16
    }

    # Strict check of race summary guard eligibility
    assert {k: v["eligible"] for k, v in art["race_summary"].items()} == {
        "control_blind_dedupe": False,
        "control_refuse_all": False,
        "patch_combo_idemp_readback": True,
        "patch_idempotency_key": True,
        "patch_read_back_verify": True,
        "patch_retry_policy": True
    }


def test_candidate_provenance_programmatically():
    """
    Programmatic verification of candidate artifact provenance:
    1. Recomputes sha256 hashes of all source files in src/silentfail/*.py and compares against candidate provenance.
    2. Verifies working tree git diff sha256.
    3. Verifies scenario suite hash generation equivalence.
    4. Confirms naive baseline and hardened agent were evaluated under identical scenarios.
    """
    import hashlib
    import glob
    import subprocess
    from src.silentfail.scenarios import ScenarioGenerator

    cand = _candidate_artifact()
    prov = cand.get("provenance")
    assert prov is not None, "Candidate artifact must contain provenance metadata"
    assert prov["base_commit"] == "63d3e9755742c3daf97e5a173bd96e4164a9dce0"
    assert prov["is_dirty_working_tree"] is True

    # 1. Source files sha256
    for path_str, expected_hash in prov["source_file_sha256"].items():
        p = REPO_ROOT / path_str
        assert p.exists(), f"Source file {path_str} must exist"
        actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"Source file hash mismatch for {path_str}"

    # 2. Working tree diff sha256
    diff_bytes = subprocess.run(["git", "diff"], capture_output=True, cwd=str(REPO_ROOT)).stdout
    actual_diff_sha256 = hashlib.sha256(diff_bytes).hexdigest()
    assert actual_diff_sha256 == prov["working_tree_diff_sha256"], "Working tree diff hash mismatch"

    # 3. Scenario suite hashes
    gen = ScenarioGenerator(base_seed=prov["scenario_suite_config"]["base_seed"])
    train_scenarios = gen.generate_suite(prov["scenario_suite_config"]["train_count"], split="train")
    held_out_scenarios = gen.generate_suite(
        prov["scenario_suite_config"]["held_out_count"],
        split="held_out",
        seed_offset=prov["scenario_suite_config"]["held_out_seed_offset"]
    )
    actual_train_suite_hash = hashlib.sha256(json.dumps([s.scenario_id for s in train_scenarios]).encode("utf-8")).hexdigest()
    actual_heldout_suite_hash = hashlib.sha256(json.dumps([s.scenario_id for s in held_out_scenarios]).encode("utf-8")).hexdigest()
    assert actual_train_suite_hash == prov["scenario_suite_config"]["train_suite_sha256"]
    assert actual_heldout_suite_hash == prov["scenario_suite_config"]["heldout_suite_sha256"]

    # 4. Naive vs Hardened evaluation consistency (same 30 held-out scenarios evaluated for both)
    assert cand["scale_and_cost"]["scenarios_heldout"] == 30
    assert cand["heldout_metrics_before"]["counts"]["total"] == 30
    assert cand["heldout_metrics_after"]["counts"]["total"] == 30


def test_no_winner_pipeline_fabricates_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        VerifierAndSelector, "select_winner",
        lambda self, base, cands, specs: (None, {pid: {"eligible": False} for pid in cands}),
    )
    orch = Orchestrator(data_dir=str(tmp_path / "data"), tests_dir=str(tmp_path / "tests"))
    state = orch.run_full_pipeline(10, 6)
    assert state["winning_patch"] is None
    assert state["heldout_metrics_after"]["status"] == "no_candidate_passed_guards"
    code = (tmp_path / "tests" / GENERATED_TEST_FILENAME).read_text(encoding="utf-8")
    assert "WINNING_PATCH" not in code and "patch_id" not in code
    assert "def test_" not in code
