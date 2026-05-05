from data.intent_map import heuristic_priority, to_intent


def test_intent_map_billing():
    assert to_intent("get_refund", "REFUND") == "billing"
    assert to_intent("check_invoices", "INVOICE") == "billing"


def test_intent_map_account_access():
    assert to_intent("recover_password", "ACCOUNT") == "account_access"
    assert to_intent("delete_account", "ACCOUNT") == "account_access"


def test_intent_map_cancellation():
    assert to_intent("cancel_order", "ORDER") == "cancellation"


def test_intent_map_falls_back_to_category():
    assert to_intent("nonexistent_intent", "REFUND") == "billing"


def test_intent_map_unknown_returns_other():
    assert to_intent(None, None) == "other"
    assert to_intent("???", "???") == "other"


def test_heuristic_priority_urgent_signal():
    assert heuristic_priority("billing", "I need this fixed ASAP") == "urgent"
    assert heuristic_priority("technical_issue", "everything is broken") == "urgent"


def test_heuristic_priority_default():
    assert heuristic_priority("billing", "could you confirm the policy?") == "high"
    assert heuristic_priority("general_inquiry", "just curious") == "medium"
