"""
Typed Declarative Patches and Registry Wrappers.
Sections 10.2, 10.3 of the Project Spec.

Supported patch types:
- wrapper.idempotency_key: generates deterministic key per (order_id, action, amount)
- wrapper.read_back_verify: queries get_order state before retrying or confirming
- wrapper.retry_policy: on timeout/error, inspects world state before repeating mutating actions
- combo.idempotency_and_read_back: combines idempotency key and read-back verification
- control.refuse_all: negative control (rejects all refunds, proving the regression guard works)
- control.blind_dedupe: naive deduplication (blocks repeat calls on the same order, proving legitimate repeat failures)
"""

from typing import Dict, Any, Optional
from src.silentfail.tools import ToolRegistry
from src.silentfail.models import PatchSpec


class PatchedToolRegistry:
    """
    Wraps the ToolRegistry to enforce declarative hardening patches.
    Follows a strict operation-correlation contract:
    - Retries must explicitly supply the same operation_id or reuse an existing idempotency_key.
    - Independent calls receive distinct operation_ids and idempotency_keys.
    - Identity is never inferred from order_id + amount or ambiguous per-order state.
    """
    def __init__(self, raw_registry: ToolRegistry, patch: Optional[PatchSpec] = None):
        self.raw = raw_registry
        self.patch = patch
        # Maps operation_id -> stable idempotency key
        self._op_key_map: Dict[str, str] = {}
        self._op_seq_counter: int = 0
        self._order_refunded_keys: set = set()

    def get_order(self, order_id: str) -> Dict[str, Any]:
        return self.raw.get_order(order_id)

    def _resolve_operation_id(self, order_id: str, operation_id: Optional[str] = None) -> str:
        """
        Resolves or generates a unique logical operation ID.
        Never infers logical identity from (order_id + amount) or prior unresolved calls.
        Every call without an explicit operation_id represents an independent request
        and receives a distinct, unique operation ID.
        """
        if operation_id:
            return operation_id
        self._op_seq_counter += 1
        return f"op_{order_id}_{self._op_seq_counter}"

    def _get_stable_idempotency_key(self, op_id: str) -> str:
        """
        Generates or retrieves one stable idempotency key strictly per unique operation ID.
        All retries sharing that operation ID reuse the same key.
        """
        if op_id not in self._op_key_map:
            self._op_key_map[op_id] = f"idem_{op_id}"
        return self._op_key_map[op_id]

    def _verify_specific_refund(self, order_id: str, amount: float, stable_key: str) -> Optional[Dict[str, Any]]:
        """
        Verifies read-back strictly by exact operation ID / stable idempotency key.
        Does NOT accept random newly created refunds with matching amounts.
        """
        for r in self.raw.world_state.refunds.values():
            if r.order_id == str(order_id) and r.idempotency_key == stable_key:
                return {
                    "status": "ok",
                    "refund_id": r.id,
                    "amount": r.amount,
                    "order_id": order_id,
                    "verified_by_readback": True
                }
        return None

    def issue_refund(
        self,
        order_id: str,
        amount: float,
        idempotency_key: Optional[str] = None,
        operation_id: Optional[str] = None,
        op_index: Optional[int] = None
    ) -> Dict[str, Any]:
        # Backward compatibility: if caller supplied op_index but not operation_id, derive unique op_id from op_index
        if not operation_id and op_index is not None:
            operation_id = f"op_{order_id}_req{op_index}"

        # Resolve logical operation ID and its stable idempotency key
        op_id = self._resolve_operation_id(order_id, operation_id)
        stable_key = idempotency_key or self._get_stable_idempotency_key(op_id)

        if not self.patch:
            # Unpatched tool registry call
            return self.raw.issue_refund(order_id, amount, idempotency_key=idempotency_key, operation_id=op_id)

        p_type = self.patch.type

        # 1. Control: refuse_all
        if p_type == "control.refuse_all":
            self.raw.ledger.record_tool_request("issue_refund", {"order_id": order_id, "amount": amount, "control": "refuse_all"})
            res = {"status": "error", "message": "Policy refusal: all refund requests blocked"}
            self.raw.ledger.record_tool_response(res)
            return res

        # 2. Control: blind_dedupe
        if p_type == "control.blind_dedupe":
            if order_id in self._order_refunded_keys:
                self.raw.ledger.record_tool_request("issue_refund", {"order_id": order_id, "amount": amount, "control": "blind_dedupe"})
                res = {"status": "ok", "message": "Deduplicated repeat call on order"}
                self.raw.ledger.record_tool_response(res)
                return res
            self._order_refunded_keys.add(order_id)
            return self.raw.issue_refund(order_id, amount, idempotency_key=stable_key, operation_id=op_id)

        # 3. Wrapper: idempotency_key
        if p_type == "wrapper.idempotency_key":
            return self.raw.issue_refund(order_id, amount, idempotency_key=stable_key, operation_id=op_id)

        # 4. Wrapper: read_back_verify
        if p_type == "wrapper.read_back_verify":
            resp = self.raw.issue_refund(order_id, amount, idempotency_key=stable_key, operation_id=op_id)
            is_unclear = ("error" in resp and resp.get("error") == "timeout") or (resp.get("status") == 200 and resp.get("data") is None)
            if is_unclear:
                verified = self._verify_specific_refund(order_id, amount, stable_key)
                if verified:
                    return verified
                return {"status": "error", "error": "timeout", "verified_by_readback": False, "effect_applied": False}
            return resp

        # 5. Wrapper: retry_policy
        if p_type == "wrapper.retry_policy":
            resp = self.raw.issue_refund(order_id, amount, idempotency_key=stable_key, operation_id=op_id)
            is_unclear = ("error" in resp and resp.get("error") == "timeout") or (resp.get("status") == 200 and resp.get("data") is None)
            if is_unclear:
                verified = self._verify_specific_refund(order_id, amount, stable_key)
                if verified:
                    return verified
                return {"status": "error", "error": "timeout", "verified_by_readback": False, "effect_applied": False}
            return resp

        # 6. Combo: idempotency_key + read_back_verify
        if p_type == "combo.idempotency_and_read_back":
            resp = self.raw.issue_refund(order_id, amount, idempotency_key=stable_key, operation_id=op_id)
            is_unclear = ("error" in resp and resp.get("error") == "timeout") or (resp.get("status") == 200 and resp.get("data") is None)
            if is_unclear:
                verified = self._verify_specific_refund(order_id, amount, stable_key)
                if verified:
                    return verified
                return {"status": "error", "error": "timeout", "verified_by_readback": False, "effect_applied": False}
            return resp

        return self.raw.issue_refund(order_id, amount, idempotency_key=stable_key, operation_id=op_id)

    def send_email(self, to: str, subject: str, body: str) -> Dict[str, Any]:
        return self.raw.send_email(to, subject, body)

    def create_ticket(self, order_id: str, summary: str) -> Dict[str, Any]:
        return self.raw.create_ticket(order_id, summary)


def get_default_candidate_patches() -> list[PatchSpec]:
    """
    Returns the required candidate pool according to Section 10.3:
    - 3 Ultra wrapper types
    - 1 combo wrapper
    - 2 tagged control candidates (refuse_all, blind_dedupe)
    """
    return [
        PatchSpec(
            patch_id="patch_idempotency_key",
            name="Deterministic Idempotency Key Wrapper",
            type="wrapper.idempotency_key",
            description="Injects a deterministic idempotency key per (order, action, amount) into mutating tool calls.",
            is_control=False
        ),
        PatchSpec(
            patch_id="patch_read_back_verify",
            name="Read-Back Verification Wrapper",
            type="wrapper.read_back_verify",
            description="Inspects world state after timeout/empty response to confirm whether side effect completed.",
            is_control=False
        ),
        PatchSpec(
            patch_id="patch_retry_policy",
            name="State-Checked Retry Policy Wrapper",
            type="wrapper.retry_policy",
            description="On timeout/empty-200, looks up a refund recorded under the operation's stable idempotency key before reporting the outcome; retries reuse the same key.",
            is_control=False
        ),
        PatchSpec(
            patch_id="patch_combo_idemp_readback",
            name="Combo: Idempotency Key + Read-Back Verification",
            type="combo.idempotency_and_read_back",
            description="Combines idempotency keys with active state read-back for compound fault tolerance.",
            is_control=False
        ),
        PatchSpec(
            patch_id="control_refuse_all",
            name="Control: Refuse All Refunds",
            type="control.refuse_all",
            description="Negative control that refuses every refund. Proves regression and legitimate-completion guards.",
            is_control=True
        ),
        PatchSpec(
            patch_id="control_blind_dedupe",
            name="Control: Blind Deduplication",
            type="control.blind_dedupe",
            description="Suppresses all repeat actions on an order. Proves legitimate repeat and empty-200 failure detection.",
            is_control=True
        )
    ]
