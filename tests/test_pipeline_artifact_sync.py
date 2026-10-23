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
    assert art["generated_test_code"] == expected


def test_generated_test_module_uses_exact_winning_patch():
    art = _artifact()
    assert art["winning_patch"] is not None, "committed run must have a winner for this check"
    spec = importlib.util.spec_from_file_location("generated_regression_module", GENERATED)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.WINNING_PATCH.model_dump() == art["winning_patch"]


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


def test_fresh_local_run_reproduces_committed_results(tmp_path):
    art = _artifact()
    orch = Orchestrator(data_dir=str(tmp_path / "data"), tests_dir=str(tmp_path / "tests"))
    state = orch.run_full_pipeline(art["scale_and_cost"]["scenarios_train"], art["scale_and_cost"]["scenarios_heldout"])
    state = json.loads(json.dumps(state, default=str))  # normalize tuples -> lists like the saved artifact
    assert state["winning_patch"] == art["winning_patch"]
    for key in ("baseline_metrics", "heldout_metrics_before", "heldout_metrics_after"):
        assert state[key] == art[key]
    assert {k: v["eligible"] for k, v in state["race_summary"].items()} == {
        k: v["eligible"] for k, v in art["race_summary"].items()
    }
    written = (tmp_path / "tests" / GENERATED_TEST_FILENAME).read_text(encoding="utf-8")
    assert written == state["generated_test_code"] == art["generated_test_code"]
    assert json.loads((tmp_path / "data" / "latest_pipeline_run.json").read_text(encoding="utf-8"))["winning_patch"] == art["winning_patch"]


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
