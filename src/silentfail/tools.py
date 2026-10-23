"""
Mock Tools and Fault Injection implementation.
Sections 5.4, 6.0 of the Project Spec.

Tools:
- get_order(order_id)
- issue_refund(order_id, amount, idempotency_key=None)
- send_email(to, subject, body)
- create_ticket(order_id, summary)

Faults:
- timeout_after_success: action completes, agent sees {"error": "timeout"}
- empty_200: agent sees HTTP 200 with empty/null body, effect may or may not be applied
- duplicate_webhook: action completes once, second event simulation
"""

from typing import Dict, Any, List, Optional
from src.silentfail.models import FaultConfig, Refund, Email, Ticket
from src.silentfail.ledger import WorldState, CausalLedger
import uuid


class FaultInjector:
    def __init__(self, faults: Optional[List[FaultConfig]] = None):
        self.faults = faults or []
        self.call_counters: Dict[str, int] = {}

    def get_fault_for_call(self, tool_name: str) -> Optional[FaultConfig]:
        idx = self.call_counters.get(tool_name, 0)
        self.call_counters[tool_name] = idx + 1
        for f in self.faults:
            if f.tool == tool_name and f.call_index == idx:
                return f
        return None


class ToolRegistry:
    def __init__(self, world_state: WorldState, ledger: CausalLedger, fault_injector: Optional[FaultInjector] = None):
        self.world_state = world_state
        self.ledger = ledger
        self.fault_injector = fault_injector or FaultInjector()
        self._refund_key_cache: Dict[str, Dict[str, Any]] = {}

    def get_order(self, order_id: str) -> Dict[str, Any]:
        self.ledger.record_tool_request("get_order", {"order_id": order_id})
        fault = self.fault_injector.get_fault_for_call("get_order")
        
        order = self.world_state.orders.get(str(order_id))
        real_result = order.model_dump() if order else {"error": f"Order {order_id} not found"}
        
        # Apply fault if any
        if fault and fault.type == "timeout_after_success":
            observed = {"error": "timeout"}
            self.ledger.record_tool_response(observed, fault="timeout_after_success")
            return observed
        elif fault and fault.type == "empty_200":
            observed = {"status": 200, "data": None}
            self.ledger.record_tool_response(observed, fault="empty_200")
            return observed
            
        self.ledger.record_tool_response(real_result)
        return real_result

    def issue_refund(self, order_id: str, amount: float, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        self.ledger.record_tool_request("issue_refund", {
            "order_id": order_id,
            "amount": amount,
            "idempotency_key": idempotency_key
        })
        fault = self.fault_injector.get_fault_for_call("issue_refund")
        
        # Idempotency check at tool layer (if caller supplied a key)
        if idempotency_key and idempotency_key in self._refund_key_cache:
            cached_result = self._refund_key_cache[idempotency_key]
            # No world mutation performed for duplicate key!
            self.ledger.record_tool_response(cached_result)
            return cached_result

        # Determine real effect
        order = self.world_state.orders.get(str(order_id))
        if not order:
            err = {"error": f"Order {order_id} not found"}
            self.ledger.record_tool_response(err)
            return err
            
        remaining_balance = round(order.total - order.refunded_amount, 2)
        if amount > remaining_balance or remaining_balance <= 0:
            err = {"error": "Order already fully refunded or requested amount exceeds remaining balance", "remaining_balance": remaining_balance}
            self.ledger.record_tool_response(err)
            return err

        # Check fault conditions that suppress mutation
        apply_mutation = True
        if fault and fault.type == "empty_200" and fault.effect_applied is False:
            apply_mutation = False

        if apply_mutation:
            ref_id = f"rf_{uuid.uuid4().hex[:6]}"
            new_refund = Refund(id=ref_id, order_id=str(order_id), amount=amount, idempotency_key=idempotency_key)
            self.world_state.refunds[ref_id] = new_refund
            order.refunded_amount = round(order.refunded_amount + amount, 2)
            if order.refunded_amount >= order.total:
                order.status = "refunded"
            else:
                order.status = "partially_refunded"
                
            self.ledger.record_world_mutation("refund_created", details={"order_id": order_id, "amount": amount}, refund_id=ref_id, amount=amount)
            real_result = {"status": "ok", "refund_id": ref_id, "amount": amount, "order_id": order_id}
            if idempotency_key:
                self._refund_key_cache[idempotency_key] = real_result
        else:
            real_result = {"status": "skipped", "reason": "fault_injected_no_effect"}

        # Determine what the agent observes
        if fault:
            if fault.type == "timeout_after_success":
                observed = {"error": "timeout"}
                self.ledger.record_tool_response(observed, fault="timeout_after_success")
                return observed
            elif fault.type == "empty_200":
                observed = {"status": 200, "data": None}
                self.ledger.record_tool_response(observed, fault="empty_200")
                return observed
            elif fault.type == "duplicate_webhook":
                observed = {"status": "ok", "refund_id": real_result.get("refund_id"), "event": "webhook.refund.completed"}
                self.ledger.record_tool_response(observed, fault="duplicate_webhook")
                return observed

        self.ledger.record_tool_response(real_result)
        return real_result

    def send_email(self, to: str, subject: str, body: str) -> Dict[str, Any]:
        self.ledger.record_tool_request("send_email", {"to": to, "subject": subject, "body": body})
        fault = self.fault_injector.get_fault_for_call("send_email")
        
        apply_mutation = True
        if fault and fault.type == "empty_200" and fault.effect_applied is False:
            apply_mutation = False

        em_id = f"em_{uuid.uuid4().hex[:6]}"
        if apply_mutation:
            new_em = Email(id=em_id, to=to, subject=subject, body=body)
            self.world_state.emails.append(new_em)
            self.ledger.record_world_mutation("email_sent", details={"to": to, "subject": subject})
            real_result = {"status": "ok", "email_id": em_id}
        else:
            real_result = {"status": "skipped", "reason": "fault_injected_no_effect"}

        if fault:
            if fault.type == "timeout_after_success":
                observed = {"error": "timeout"}
                self.ledger.record_tool_response(observed, fault="timeout_after_success")
                return observed
            elif fault.type == "empty_200":
                observed = {"status": 200, "data": None}
                self.ledger.record_tool_response(observed, fault="empty_200")
                return observed

        self.ledger.record_tool_response(real_result)
        return real_result

    def create_ticket(self, order_id: str, summary: str) -> Dict[str, Any]:
        self.ledger.record_tool_request("create_ticket", {"order_id": order_id, "summary": summary})
        fault = self.fault_injector.get_fault_for_call("create_ticket")
        
        tk_id = f"tk_{uuid.uuid4().hex[:6]}"
        new_tk = Ticket(id=tk_id, order_id=str(order_id), summary=summary)
        self.world_state.tickets.append(new_tk)
        self.ledger.record_world_mutation("ticket_created", details={"order_id": order_id, "summary": summary})
        real_result = {"status": "ok", "ticket_id": tk_id}
        
        if fault and fault.type == "timeout_after_success":
            observed = {"error": "timeout"}
            self.ledger.record_tool_response(observed, fault="timeout_after_success")
            return observed

        self.ledger.record_tool_response(real_result)
        return real_result
