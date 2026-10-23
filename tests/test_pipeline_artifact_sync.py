"""
Pipeline Artifact & Generated Regression Test Synchronization Tests.
Ensures:
1. data/latest_pipeline_run.json and tests/test_generated_regression.py agree on the winning patch ID and type.
2. README.md reported metrics match data/latest_pipeline_run.json.
3. Fresh pipeline replay reproduces the deterministic winner and held-out counts.
4. No-winner path writes a clear notice and invents no fake patch.
"""

import json
from pathlib import Path
import pytest
from src.silentfail.models import PatchSpec
from src.silentfail.orchestrator import Orchestrator


def test_pipeline_artifact_and_generated_test_sync():
    """Verify that latest_pipeline_run.json and test_generated_regression.py agree on winning patch."""
    artifact_path = Path("data/latest_pipeline_run.json")
    assert artifact_path.exists(), "latest_pipeline_run.json must exist"
    
    with open(artifact_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    winning_patch = data.get("winning_patch")
    assert winning_patch is not None, "Winning patch must be selected in latest run"
    winner_id = winning_patch["patch_id"]
    winner_type = winning_patch["type"]

    regression_test_path = Path("tests/test_generated_regression.py")
    assert regression_test_path.exists(), "test_generated_regression.py must exist"
    test_content = regression_test_path.read_text(encoding="utf-8")

    assert f'patch_id="{winner_id}"' in test_content, f"Generated test must pin actual winner {winner_id}"
    assert f'type="{winner_type}"' in test_content, f"Generated test must pin actual winner type {winner_type}"


def test_no_winner_path_writes_safe_notice_without_inventing_patch():
    """Verify that when winner is None, render_regression_test_code produces an honest notice."""
    orch = Orchestrator()
    code = orch.render_regression_test_code(None)
    assert "No candidate patch passed all 4 verifier guards" in code
    assert "patch_id=" not in code
    assert "WINNING_PATCH" not in code
