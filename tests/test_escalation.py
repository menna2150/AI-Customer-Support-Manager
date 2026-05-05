from agents import escalation
from core.state import Classification, EscalationPacket, Plan


def test_escalation_packet_built(fake_llm, sample_state):
    sample_state.classification = Classification(
        intent="billing",
        priority="urgent",
        sentiment="frustrated",
        language="en",
        summary="double charge complaint",
        confidence=0.9,
    )
    sample_state.plan = Plan(route="escalate", needs_kb=False, rationale="Frustrated, refund edge case.")

    fake_llm.respond(
        EscalationPacket,
        EscalationPacket(
            summary="Customer Sara reports being double-charged for the third time...",
            suggested_next_steps=[
                "Verify charges in Stripe",
                "Issue refund if duplicate confirmed",
                "Reply within 1 hour",
            ],
            customer_facing_acknowledgement="Hi Sara — I've routed this to our billing team...",
            severity="urgent",
        ),
    )

    out = escalation.run(sample_state)

    assert out.escalation is not None
    assert out.escalation.severity == "urgent"
    assert len(out.escalation.suggested_next_steps) == 3
