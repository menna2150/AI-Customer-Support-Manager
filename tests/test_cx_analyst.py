from agents import cx_analyst
from core.state import CXInsights, Classification


def test_cx_high_churn_for_frustrated_billing(fake_llm, sample_state):
    sample_state.classification = Classification(
        intent="billing",
        priority="urgent",
        sentiment="frustrated",
        language="en",
        summary="threatening chargeback",
        confidence=0.95,
    )
    sample_state.final_route = "escalated"

    fake_llm.respond(
        CXInsights,
        CXInsights(
            churn_risk="high",
            satisfaction_signal="frustrated",
            tags=["double_charge", "refund_request", "human_requested"],
            notes="Third billing complaint in 30 days — investigate root cause.",
        ),
    )

    out = cx_analyst.run(sample_state)

    assert out.cx.churn_risk == "high"
    assert "double_charge" in out.cx.tags
