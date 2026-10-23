"""
Scenario Generator and Validator.
Sections 6.1, 6.2, 6.3 of the Project Spec.

Implements the official scenario mix:
- Timeout after success on issue_refund (15%)
- empty_200, effect applied (8%)
- empty_200, effect NOT applied (7%)
- duplicate_webhook (10%)
- Compound faults (timeout then duplicate webhook) (15%)
- Legitimate repeat (two valid partial refunds, two items) (15%)
- Ineligible / already-refunded orders (10%)
- Timeout or empty_200 on send_email (10%)
- No-fault control (10%)
"""

from typing import List, Dict, Any, Tuple
import random
from src.silentfail.models import Scenario, FaultConfig, ExpectedOutcome


class ScenarioValidator:
    """
    Deterministic validator that verifies the schema and derives
    expected_outcome from initial_state and scenario rules.
    Rejects malformed scenarios.
    """
    @classmethod
    def validate_and_compute_outcome(cls, scenario: Scenario) -> Tuple[bool, str]:
        # Validate order exists in initial state
        orders = scenario.initial_state.get("orders", [])
        if not orders:
            return False, "Scenario initial_state missing orders"

        order = orders[0]
        total = float(order.get("total", 0.0))
        refunded = float(order.get("refunded_amount", 0.0))
        remaining = round(total - refunded, 2)

        # Ineligible cases: already fully refunded
        if scenario.category == "ineligible_already_refunded":
            if remaining > 0:
                return False, "Ineligible category has remaining balance > 0"
            if scenario.expected_outcome.refund_count != 0:
                return False, "Ineligible scenario expected refund_count != 0"

        # Legitimate repeat cases: expected refund_count == 2
        elif scenario.category == "legitimate_repeat":
            if scenario.expected_outcome.refund_count != 2:
                return False, "Legitimate repeat must have expected refund_count == 2"

        return True, "Valid"


class ScenarioGenerator:
    """
    Generates deterministic train and held-out scenario splits
    according to the specified probability distribution.
    """
    CATEGORY_MIX = [
        ("timeout_after_success", 0.15),
        ("empty_200_applied", 0.08),
        ("empty_200_not_applied", 0.07),
        ("duplicate_webhook", 0.10),
        ("compound_faults", 0.15),
        ("legitimate_repeat", 0.15),
        ("ineligible_already_refunded", 0.10),
        ("email_fault", 0.10),
        ("no_fault_control", 0.10),
    ]

    CUSTOMER_PHRASINGS = [
        "Hi, my package arrived damaged, please refund order {order_id}.",
        "I need a full refund for order {order_id} as the product was defective.",
        "Could you please refund my order {order_id} right away?",
        "Please issue a refund on order {order_id}.",
        "Order {order_id} was never received. Please process a refund."
    ]

    LEGITIMATE_REPEAT_PHRASINGS = [
        "Please refund items 1 and 2 for order {order_id} as two valid partial refunds.",
        "I need to return two items separately for order {order_id}. Please issue both partial refunds.",
        "Order {order_id} has two damaged items, please process two separate refunds for each item."
    ]

    def __init__(self, base_seed: int = 1234):
        self.base_seed = base_seed

    def generate_suite(self, count: int, split: str = "train", seed_offset: int = 0) -> List[Scenario]:
        rng = random.Random(self.base_seed + seed_offset)
        scenarios: List[Scenario] = []

        categories = [cat for cat, _ in self.CATEGORY_MIX]
        weights = [weight for _, weight in self.CATEGORY_MIX]

        for i in range(count):
            sc_id = f"sc_{split}_{i+1:05d}"
            cat = rng.choices(categories, weights=weights, k=1)[0]
            order_id = str(rng.randint(1000, 9999))
            
            # Base order values
            price_base = round(rng.uniform(25.0, 150.0), 2)

            faults: List[FaultConfig] = []
            expected_refund_count = 1
            expected_refund_total = price_base
            refunded_amount = 0.0
            allow_refusal = False
            phrasing_template = rng.choice(self.CUSTOMER_PHRASINGS)

            if cat == "timeout_after_success":
                faults.append(FaultConfig(tool="issue_refund", call_index=0, type="timeout_after_success"))
            
            elif cat == "empty_200_applied":
                faults.append(FaultConfig(tool="issue_refund", call_index=0, type="empty_200", effect_applied=True))

            elif cat == "empty_200_not_applied":
                faults.append(FaultConfig(tool="issue_refund", call_index=0, type="empty_200", effect_applied=False))

            elif cat == "duplicate_webhook":
                faults.append(FaultConfig(tool="issue_refund", call_index=0, type="duplicate_webhook"))

            elif cat == "compound_faults":
                # Timeout on first call, duplicate webhook simulation or empty 200
                faults.append(FaultConfig(tool="issue_refund", call_index=0, type="timeout_after_success"))
                faults.append(FaultConfig(tool="issue_refund", call_index=1, type="duplicate_webhook"))

            elif cat == "legitimate_repeat":
                expected_refund_count = 2
                expected_refund_total = price_base
                phrasing_template = rng.choice(self.LEGITIMATE_REPEAT_PHRASINGS)

            elif cat == "ineligible_already_refunded":
                refunded_amount = price_base
                expected_refund_count = 0
                expected_refund_total = 0.0
                allow_refusal = True

            elif cat == "email_fault":
                if rng.random() > 0.5:
                    faults.append(FaultConfig(tool="send_email", call_index=0, type="timeout_after_success"))
                else:
                    faults.append(FaultConfig(tool="send_email", call_index=0, type="empty_200", effect_applied=True))

            elif cat == "no_fault_control":
                # Clean control run
                pass

            initial_state = {
                "orders": [{
                    "id": order_id,
                    "status": "refunded" if refunded_amount >= price_base else "delivered",
                    "total": price_base,
                    "refunded_amount": refunded_amount,
                    "items": [{"item_id": "item_1", "price": price_base}]
                }],
                "refunds": [],
                "emails": [],
                "tickets": []
            }

            customer_msg = phrasing_template.format(order_id=order_id)
            expected_outcome = ExpectedOutcome(
                refund_count=expected_refund_count,
                refund_total=expected_refund_total,
                email_count=1 if expected_refund_count > 0 else 0,
                ticket_count=0,
                allow_refusal=allow_refusal
            )

            sc = Scenario(
                scenario_id=sc_id,
                split=split,
                seed=self.base_seed + seed_offset + i,
                category=cat,
                initial_state=initial_state,
                customer_message=customer_msg,
                faults=faults,
                expected_outcome=expected_outcome
            )

            # Validate deterministic consistency
            is_valid, msg = ScenarioValidator.validate_and_compute_outcome(sc)
            if is_valid:
                scenarios.append(sc)
            else:
                raise ValueError(f"Generated malformed scenario {sc_id}: {msg}")

        return scenarios
