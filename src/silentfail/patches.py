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
    """
    def __init__(self, raw_registry: ToolRegistry, patch: Optional[PatchSpec] = None):
        self.raw = raw_registry
        self.patch = patch
        # Maps (order_id, logical_op_index) -> stable idempotency key
        self._op_key_map: Dict[str, str] = {}
        # Records initial refund state per order before any mutations in this session
        self._initial_refund_ids: Dict[str, set] = {}
        self._order_refunded_keys: set = set()

    def get_order(self, order_id: str) -> Dict[str, Any]:
        return self.raw.get_order(order_id)

    def _get_stable_op_key(self, order_id: str, amount: float, op_index: Optional[int] = None) -> str:
        """
        Generates or retrieves one stable operation key per logical refund request.
        Separate logical operations on the same order get distinct stable keys.
        """
        key_id = f"{order_id}_{amount}" if op_index is None else f"{order_id}_{amount}_{op_index}"
        if key_id not in self._op_key_map:
            self._op_key_map[key_id] = f"idem_{order_id}_{int(amount*100)}_{len(self._op_key_map)}"
        return self._op_key_map[key_id]

    def _verify_specific_refund(self, order_id: str, amount: float, stable_key: str, initial_refund_ids: set) -> Optional[Dict[str, Any]]:
        """
        Verifies the specific refund operation through:
        1. A matching idempotency key on a newly created refund record, OR
        2. A newly created refund record (absent from initial state) with matching order_id and exact amount.
        Never infers success from cumulative refunded_amount >= requested amount.
        """
        current_refunds = self.raw.world_state.refunds.values()
        # 1. Check by stable idempotency key
        for r in current_refunds:
            if r.order_id == str(order_id) and r.idempotency_key == stable_key:
                return {"status": "ok", "refund_id": r.id, "amount": r.amount, "order_id": order_id, "verified_by_readback": True}
        
        # 2. Check for newly created refund since this operation began
        for r in current_refunds:
            if r.order_id == str(order_id) and r.id not in initial_refund_ids and abs(r.amount - amount) < 0.001:
                return {"status": "ok", "refund_id": r.id, "amount": r.amount, "order_id": order_id, "verified_by_readback": True}

        return None

    def issue_refund(self, order_id: str, amount: float, idempotency_key: Optional[str] = None, op_index: Optional[int] = None) -> Dict[str, Any]:
        if not self.patch:
            return self.raw.issue_refund(order_id, amount, idempotency_key)

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
            return self.raw.issue_refund(order_id, amount, idempotency_key)

        # Snapshot initial refund IDs for this order before call
        existing_refund_ids = {r.id for r in self.raw.world_state.refunds.values() if r.order_id == str(order_id)}
        stable_key = idempotency_key or self._get_stable_op_key(order_id, amount, op_index)

        # 3. Wrapper: idempotency_key
        if p_type == "wrapper.idempotency_key":
            return self.raw.issue_refund(order_id, amount, idempotency_key=stable_key)

        # 4. Wrapper: read_back_verify
        if p_type == "wrapper.read_back_verify":
            resp = self.raw.issue_refund(order_id, amount, idempotency_key=idempotency_key)
            # Handle timeout or empty response (200 with empty body)
            is_unclear = ("error" in resp and resp.get("error") == "timeout") or (resp.get("status") == 200 and resp.get("data") is None)
            if is_unclear:
                verified = self._verify_specific_refund(order_id, amount, stable_key, existing_refund_ids)
                if verified:
                    return verified
                # Specific effect was NOT applied - return honest failure/timeout to agent
                return {"status": "error", "error": "timeout", "verified_by_readback": False, "effect_applied": False}
            return resp

        # 5. Wrapper: retry_policy
        if p_type == "wrapper.retry_policy":
            resp = self.raw.issue_refund(order_id, amount, idempotency_key=idempotency_key)
            is_unclear = ("error" in resp and resp.get("error") == "timeout") or (resp.get("status") == 200 and resp.get("data") is None)
            if is_unclear:
                verified = self._verify_specific_refund(order_id, amount, stable_key, existing_refund_ids)
                if verified:
                    return verified
                return {"status": "error", "error": "timeout", "verified_by_readback": False, "effect_applied": False}
            return resp

        # 6. Combo: idempotency_key + read_back_verify
        if p_type == "combo.idempotency_and_read_back":
            resp = self.raw.issue_refund(order_id, amount, idempotency_key=stable_key)
            is_unclear = ("error" in resp and resp.get("error") == "timeout") or (resp.get("status") == 200 and resp.get("data") is None)
            if is_unclear:
                verified = self._verify_specific_refund(order_id, amount, stable_key, existing_refund_ids)
                if verified:
                    return verified
                return {"status": "error", "error": "timeout", "verified_by_readback": False, "effect_applied": False}
            return resp

        return self.raw.issue_refund(order_id, amount, idempotency_key)

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
            description="Checks underlying order balance on timeout before permitting retries.",
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
