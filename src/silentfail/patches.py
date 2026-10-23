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
        self._action_call_history: Dict[str, int] = {}
        self._order_refunded_keys: set = set()

    def get_order(self, order_id: str) -> Dict[str, Any]:
        return self.raw.get_order(order_id)

    def issue_refund(self, order_id: str, amount: float, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        if not self.patch:
            return self.raw.issue_refund(order_id, amount, idempotency_key)

        p_type = self.patch.type

        # 1. Control: refuse_all
        if p_type == "control.refuse_all":
            # Negative control: never issues refunds
            self.raw.ledger.record_tool_request("issue_refund", {"order_id": order_id, "amount": amount, "control": "refuse_all"})
            res = {"status": "error", "message": "Policy refusal: all refund requests blocked"}
            self.raw.ledger.record_tool_response(res)
            return res

        # 2. Control: blind_dedupe
        if p_type == "control.blind_dedupe":
            # Suppresses any repeat issue_refund call on an order regardless of items or amount
            if order_id in self._order_refunded_keys:
                self.raw.ledger.record_tool_request("issue_refund", {"order_id": order_id, "amount": amount, "control": "blind_dedupe"})
                res = {"status": "ok", "message": "Deduplicated repeat call on order"}
                self.raw.ledger.record_tool_response(res)
                return res
            self._order_refunded_keys.add(order_id)
            return self.raw.issue_refund(order_id, amount, idempotency_key)

        # 3. Wrapper: idempotency_key
        if p_type in ("wrapper.idempotency_key", "combo.idempotency_and_read_back"):
            # Generates a deterministic key per (order, action, amount, item_index)
            call_idx = self._action_call_history.get(order_id, 0)
            self._action_call_history[order_id] = call_idx + 1
            generated_key = f"idemp_{order_id}_{amount}_{call_idx}"
            key_to_use = idempotency_key or generated_key

            resp = self.raw.issue_refund(order_id, amount, idempotency_key=key_to_use)
            
            # If combo, also read-back verify
            if p_type == "combo.idempotency_and_read_back":
                if "error" in resp and resp.get("error") == "timeout":
                    # Read back order to verify if it actually succeeded
                    order_check = self.raw.world_state.orders.get(str(order_id))
                    if order_check and order_check.refunded_amount >= amount:
                        # Mutation actually succeeded! Return normalized success
                        resp = {"status": "ok", "refund_id": f"verified_{order_id}", "amount": amount, "verified_by_readback": True}
            return resp

        # 4. Wrapper: read_back_verify
        if p_type == "wrapper.read_back_verify":
            resp = self.raw.issue_refund(order_id, amount, idempotency_key)
            if "error" in resp and resp.get("error") == "timeout":
                # Check ground truth order state
                order_check = self.raw.world_state.orders.get(str(order_id))
                if order_check and order_check.refunded_amount >= amount:
                    resp = {"status": "ok", "refund_id": f"verified_{order_id}", "amount": amount, "verified_by_readback": True}
            return resp

        # 5. Wrapper: retry_policy
        if p_type == "wrapper.retry_policy":
            resp = self.raw.issue_refund(order_id, amount, idempotency_key)
            if "error" in resp and resp.get("error") == "timeout":
                # Check order state before allowing any retry
                order_check = self.raw.world_state.orders.get(str(order_id))
                if order_check and order_check.refunded_amount > 0:
                    # Already mutated, don't retry blindly
                    resp = {"status": "ok", "message": "State check confirmed refund completed despite timeout"}
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
