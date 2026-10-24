"""
Nebius AI Cloud & NVIDIA Nemotron Integration Client with Local Fallback.
Sections 11, 14, 16 of the Project Spec.

Features:
- OpenAI-compatible client interface pointing to Nebius Token Factory / NVIDIA endpoints
- Real connectivity test & throughput measurement (Gate G0)
- Graceful deterministic local fallback with clear logging if credentials or quotas block remote calls
- Structured output parsing for diagnosis and patch proposal (Ultra tier)
- Structured claim extraction (Super tier)
"""

import os
import json
import time
import re
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from src.silentfail.models import PatchSpec


def sanitize_error_message(err: Any, sensitive_tokens: Optional[List[str]] = None) -> str:
    """
    Sanitizes error messages to prevent leakage of credentials, bearer tokens,
    API keys, or authorization headers.
    """
    err_str = str(err)
    if not err_str:
        return f"{type(err).__name__}" if err is not None else "Unknown error"

    # Redact explicit sensitive strings if supplied
    if sensitive_tokens:
        for token in sensitive_tokens:
            if token and len(token) > 4:
                err_str = err_str.replace(token, "[REDACTED_API_KEY]")

    # Redact common credential patterns (Bearer tokens, Authorization headers, api_key=...)
    err_str = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]{8,}", r"\1[REDACTED_TOKEN]", err_str, flags=re.IGNORECASE)
    err_str = re.sub(r"((?:api[_-]?key|password|secret|authorization)\s*[:=]\s*['\"]?)[A-Za-z0-9_\-\.]{8,}(['\"]?)", r"\1[REDACTED_CREDENTIAL]\2", err_str, flags=re.IGNORECASE)
    err_str = re.sub(r"(v1\.[A-Za-z0-9_\-]{20,})", "[REDACTED_TOKEN_FACTORY_KEY]", err_str)

    return f"{type(err).__name__}: {err_str}"

# Allowed patch types matching PatchSpec Literal in models.py
ALLOWED_PATCH_TYPES = {
    "wrapper.idempotency_key",
    "wrapper.read_back_verify",
    "wrapper.retry_policy",
    "combo.idempotency_and_read_back",
    "control.refuse_all",
    "control.blind_dedupe"
}

ALLOWED_PATCH_SPEC_FIELDS = {"patch_id", "name", "type", "description", "is_control", "config"}

PATCH_PROPOSAL_SYSTEM_PROMPT = """You are an automated root-cause reliability engineer analyzing failing tool execution ledgers in SilentFail Hardener.
Your goal is to propose declarative tool wrapper patches from a strictly constrained set of supported patch types.

ALLOWED PATCH TYPES (Only choose from these exact strings):
1. "wrapper.idempotency_key"
   Semantics: Generates a deterministic idempotency key per logical operation to guarantee tool call deduplication and prevent duplicate mutations.
2. "wrapper.read_back_verify"
   Semantics: Queries world/order state (read-back verification) before and after mutations to verify real state before claiming success or retrying.
3. "wrapper.retry_policy"
   Semantics: On timeout or empty responses, checks world state before retrying mutating actions; ensures retries reuse stable operation IDs.
4. "combo.idempotency_and_read_back"
   Semantics: Combines deterministic idempotency key generation with active state read-back for compound fault tolerance.
5. "control.refuse_all"
   Semantics: Negative control patch that refuses all mutations (used solely for guard baseline verification).
6. "control.blind_dedupe"
   Semantics: Control patch that naively blocks repeated calls on the same order.

STRICT CONTRACT RULES:
- Output MUST be a valid JSON array of objects. Do not include markdown explanation outside the JSON.
- Every patch object MUST ONLY include the following allowed fields:
  - "patch_id": string (unique identifier, e.g. "patch_model_idempotency")
  - "name": string (descriptive name)
  - "type": string (MUST be one of the 6 allowed patch types above)
  - "description": string (explanation of mechanism)
  - "is_control": boolean (false for hardening patches, true for controls)
  - "config": object (optional, default empty dict {})
- PROHIBITED:
  - Do NOT invent new categories (e.g., "tool_contract", "protocol", "agent_policy", "observability", "safety_net" are strictly forbidden).
  - Do NOT include executable Python code or pseudo-code.
  - Do NOT include extra schema fields.
- If no supported patch is justified by the provided failure ledgers, output an empty JSON array: []

MINIMAL VALID EXAMPLE:
[
  {
    "patch_id": "patch_model_idempotency",
    "name": "Deterministic Idempotency Key Wrapper",
    "type": "wrapper.idempotency_key",
    "description": "Injects a stable idempotency key for the logical operation to prevent duplicate side effects.",
    "is_control": false,
    "config": {}
  }
]
"""


class NebiusClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = os.getenv("NEBIUS_API_KEY", "") if api_key is None else api_key
        self.base_url = base_url or os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
        self.nano_model = os.getenv("NEMOTRON_NANO_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
        self.super_model = os.getenv("NEMOTRON_SUPER_MODEL", "nvidia/nemotron-3-super-120b-a12b")
        self.ultra_model = os.getenv("NEMOTRON_ULTRA_MODEL", "nvidia/Nemotron-3-Ultra-550b-a55b")
        
        self.is_live = False
        self.client: Optional[OpenAI] = None
        
        if self.api_key and self.api_key != "your_nebius_or_nvidia_api_key_here":
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                self.is_live = True
            except Exception:
                self.is_live = False

    def probe_gate_g0(self) -> Dict[str, Any]:
        """
        Gate G0 check (Section 16):
        Measures Nemotron throughput and verifies connectivity if real credentials exist.
        Never fabricates latency or throughput on local simulation fallback.
        """
        if not self.is_live or not self.client:
            return {
                "status": "local_simulation",
                "connected": False,
                "reason": "NEBIUS_API_KEY not configured or offline. Running on local simulation engine.",
                "endpoint": self.base_url,
                "measured_rps": None,
                "latency_ms": None,
                "note": "Local deterministic simulation mode; remote Nebius call was not performed."
            }

        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.nano_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
                temperature=0.0
            )
            elapsed = time.time() - t0
            return {
                "status": "connected",
                "connected": True,
                "model": self.nano_model,
                "latency_ms": round(elapsed * 1000, 2),
                "measured_rps": round(1.0 / max(0.001, elapsed), 2),
                "response": resp.choices[0].message.content
            }
        except Exception as e:
            return {
                "status": "local_simulation",
                "connected": False,
                "reason": f"Remote Nebius endpoint error: {str(e)}. Operating in local simulation mode.",
                "endpoint": self.base_url,
                "measured_rps": None,
                "latency_ms": None,
                "note": "Local deterministic simulation mode; remote Nebius call failed."
            }

    def diagnose_and_propose_patches(self, failing_ledgers: List[Dict[str, Any]]) -> Tuple[List[PatchSpec], Dict[str, Any]]:
        """
        Ultra diagnosis and patch proposer (Section 10).
        If live LLM available, queries Nemotron Ultra with strict JSON schema.
        Otherwise uses deterministic diagnosis and generates candidate patches.
        
        Returns:
            Tuple of (candidate_patch_specs, proposal_metadata)
            where proposal_metadata contains:
            - source: "nemotron_ultra" or "deterministic_fallback"
            - model: model ID used or None
            - model_outcome: "accepted", "empty", "rejected", "request_error", "parse_error", or "not_attempted"
            - raw_proposals_count: count of candidates proposed by model
            - accepted_candidates_count: count of valid candidate patches accepted
            - fallback_candidates_count: count of deterministic fallback patches supplied
            - rejected_proposals: list of details for any invalid/unsupported proposals
            - error: model call error message if any
            - parse_error: JSON/schema parsing error message if any
        """
        from src.silentfail.patches import get_default_candidate_patches

        metadata: Dict[str, Any] = {
            "source": "deterministic_fallback",
            "model": self.ultra_model if (self.is_live and self.client) else None,
            "model_outcome": "not_attempted",
            "raw_proposals_count": 0,
            "accepted_candidates_count": 0,
            "fallback_candidates_count": 0,
            "rejected_proposals": [],
            "error": None,
            "parse_error": None
        }

        if self.is_live and self.client:
            raw_content = ""
            try:
                user_content = (
                    "Analyze the following failing execution ledgers from agent runs and propose compatible candidate patches:\n"
                    f"{json.dumps(failing_ledgers[:5], indent=2)}\n\n"
                    "Remember to respond with a JSON array of PatchSpec objects matching the supported patch types only."
                )
                resp = self.client.chat.completions.create(
                    model=self.ultra_model,
                    messages=[
                        {"role": "system", "content": PATCH_PROPOSAL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.0
                )
                raw_content = resp.choices[0].message.content or ""
            except Exception as e:
                # Never log credentials or API keys; capture safe sanitized error string
                safe_err = sanitize_error_message(e, sensitive_tokens=[self.api_key] if self.api_key else None)
                metadata["error"] = f"Model call failed: {safe_err}"
                metadata["model_outcome"] = "request_error"

            if metadata["error"] is None:
                if raw_content:
                    start_idx = raw_content.find("[")
                    end_idx = raw_content.rfind("]")
                    if start_idx != -1 and end_idx != -1:
                        try:
                            candidates_data = json.loads(raw_content[start_idx:end_idx+1])
                            if isinstance(candidates_data, list):
                                metadata["raw_proposals_count"] = len(candidates_data)
                                valid_patches: List[PatchSpec] = []

                                # If model returned an empty list [], it concluded no supported patch is justified
                                if len(candidates_data) == 0:
                                    metadata["model_outcome"] = "empty"
                                    metadata["parse_error"] = None
                                else:
                                    for idx, item in enumerate(candidates_data):
                                        if not isinstance(item, dict):
                                            metadata["rejected_proposals"].append({
                                                "index": idx,
                                                "reason": "Item is not a JSON object",
                                                "raw": str(item)[:100]
                                            })
                                            continue
                                        
                                        # Reject extra unknown fields
                                        extra_fields = set(item.keys()) - ALLOWED_PATCH_SPEC_FIELDS
                                        if extra_fields:
                                            metadata["rejected_proposals"].append({
                                                "index": idx,
                                                "patch_id": item.get("patch_id"),
                                                "reason": f"Prohibited extra fields in PatchSpec: {sorted(list(extra_fields))}"
                                            })
                                            continue

                                        # Reject non-empty configuration: wrappers in patches.py do not consume config keys
                                        config_val = item.get("config", {})
                                        if not isinstance(config_val, dict):
                                            metadata["rejected_proposals"].append({
                                                "index": idx,
                                                "patch_id": item.get("patch_id"),
                                                "reason": f"Patch config must be a dict, got {type(config_val).__name__}"
                                            })
                                            continue
                                        if config_val:
                                            metadata["rejected_proposals"].append({
                                                "index": idx,
                                                "patch_id": item.get("patch_id"),
                                                "reason": f"Prohibited non-empty config: wrappers do not consume config keys, got {sorted(list(config_val.keys()))}"
                                            })
                                            continue

                                        patch_type = item.get("type")
                                        if patch_type not in ALLOWED_PATCH_TYPES:
                                            metadata["rejected_proposals"].append({
                                                "index": idx,
                                                "patch_id": item.get("patch_id"),
                                                "reason": f"Unsupported or missing patch type: {patch_type}",
                                                "allowed_types": sorted(list(ALLOWED_PATCH_TYPES))
                                            })
                                            continue
                                        try:
                                            patch_spec = PatchSpec(**item)
                                            valid_patches.append(patch_spec)
                                        except Exception as val_err:
                                            metadata["rejected_proposals"].append({
                                                "index": idx,
                                                "patch_id": item.get("patch_id"),
                                                "reason": f"PatchSpec validation failed: {str(val_err)}"
                                            })

                                    metadata["accepted_candidates_count"] = len(valid_patches)

                                    # If at least one valid candidate was proposed, use them
                                    if valid_patches:
                                        # Ensure required negative control candidates are included so guards are verified
                                        control_defaults = [p for p in get_default_candidate_patches() if p.is_control]
                                        existing_ids = {p.patch_id for p in valid_patches}
                                        for ctrl in control_defaults:
                                            if ctrl.patch_id not in existing_ids:
                                                valid_patches.append(ctrl)

                                        metadata["source"] = "nemotron_ultra"
                                        metadata["model_outcome"] = "accepted"
                                        return valid_patches, metadata
                                    else:
                                        metadata["model_outcome"] = "rejected"
                                        metadata["parse_error"] = "No valid PatchSpec candidates passed schema and type validation."
                            else:
                                metadata["model_outcome"] = "parse_error"
                                metadata["parse_error"] = f"Expected JSON list but got {type(candidates_data).__name__}"
                        except json.JSONDecodeError as jde:
                            metadata["model_outcome"] = "parse_error"
                            metadata["parse_error"] = f"JSON parse error: {str(jde)}"
                    else:
                        metadata["model_outcome"] = "parse_error"
                        metadata["parse_error"] = "No JSON array found in model response."
                else:
                    metadata["model_outcome"] = "parse_error"
                    metadata["parse_error"] = "Empty or null response content received from model."

        # High-assurance deterministic fallback
        default_patches = get_default_candidate_patches()
        metadata["fallback_candidates_count"] = len(default_patches)
        return default_patches, metadata
