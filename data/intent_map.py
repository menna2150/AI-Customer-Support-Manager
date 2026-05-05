"""Map Bitext (category, intent) to our internal Intent / Priority labels.

Bitext columns: flags, instruction, category, intent, response.
Categories: ACCOUNT, CANCELLATION_FEE, CONTACT, DELIVERY, FEEDBACK, INVOICE,
NEWSLETTER, ORDER, PAYMENT, REFUND, SHIPPING_ADDRESS, SUBSCRIPTION.
"""
from __future__ import annotations
from typing import Tuple

from core.state import Intent, Priority

_BITEXT_INTENT_TO_OURS: dict[str, Intent] = {
    # billing
    "check_invoice": "billing",
    "check_invoices": "billing",
    "get_invoice": "billing",
    "check_payment_methods": "billing",
    "payment_issue": "billing",
    "check_refund_policy": "billing",
    "get_refund": "billing",
    "track_refund": "billing",
    "check_cancellation_fee": "billing",
    # account access
    "create_account": "account_access",
    "delete_account": "account_access",
    "edit_account": "account_access",
    "switch_account": "account_access",
    "recover_password": "account_access",
    "registration_problems": "account_access",
    # cancellation
    "cancel_order": "cancellation",
    # complaint
    "complaint": "complaint",
    "review": "complaint",
    # general inquiry / order ops
    "place_order": "general_inquiry",
    "change_order": "general_inquiry",
    "track_order": "general_inquiry",
    "delivery_options": "general_inquiry",
    "delivery_period": "general_inquiry",
    "set_up_shipping_address": "general_inquiry",
    "change_shipping_address": "general_inquiry",
    "newsletter_subscription": "general_inquiry",
    # contact
    "contact_customer_service": "general_inquiry",
    "contact_human_agent": "other",
}

_CATEGORY_TO_OURS: dict[str, Intent] = {
    "ACCOUNT": "account_access",
    "INVOICE": "billing",
    "PAYMENT": "billing",
    "REFUND": "billing",
    "CANCELLATION_FEE": "billing",
    "SUBSCRIPTION": "billing",
    "ORDER": "general_inquiry",
    "DELIVERY": "general_inquiry",
    "SHIPPING_ADDRESS": "general_inquiry",
    "NEWSLETTER": "general_inquiry",
    "CONTACT": "general_inquiry",
    "FEEDBACK": "complaint",
}


def to_intent(bitext_intent: str | None, category: str | None) -> Intent:
    """Best-effort mapping; falls back to 'other'."""
    if bitext_intent and bitext_intent in _BITEXT_INTENT_TO_OURS:
        return _BITEXT_INTENT_TO_OURS[bitext_intent]
    if category and category.upper() in _CATEGORY_TO_OURS:
        return _CATEGORY_TO_OURS[category.upper()]
    return "other"


def heuristic_priority(intent: Intent, instruction: str) -> Priority:
    """Coarse priority heuristic when the dataset doesn't supply one."""
    text = (instruction or "").lower()
    urgent_signals = ("urgent", "asap", "immediately", "right now", "broken", "not working")
    if any(s in text for s in urgent_signals):
        return "urgent"
    if intent in ("billing", "account_access", "cancellation"):
        return "high"
    if intent == "complaint":
        return "high"
    return "medium"


def split_label(bitext_intent: str | None, category: str | None) -> Tuple[Intent, str]:
    intent = to_intent(bitext_intent, category)
    label = bitext_intent or (category or "unknown").lower()
    return intent, label
