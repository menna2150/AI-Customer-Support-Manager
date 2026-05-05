from agents import classifier
from core.state import Classification


def test_classifier_populates_classification(fake_llm, sample_state):
    fake_llm.respond(
        Classification,
        Classification(
            intent="billing",
            priority="medium",
            sentiment="neutral",
            language="en",
            summary="Customer wants a refund.",
            confidence=0.91,
        ),
    )

    out = classifier.run(sample_state)

    assert out.classification is not None
    assert out.classification.intent == "billing"
    assert out.classification.priority == "medium"
    assert out.classification.confidence == 0.91
    assert any("classifier" in line for line in out.trace)


def test_classifier_passes_ticket_body_to_llm(fake_llm, make_state_factory):
    fake_llm.respond(
        Classification,
        Classification(
            intent="technical_issue",
            priority="high",
            sentiment="frustrated",
            language="en",
            summary="App is broken.",
            confidence=0.8,
        ),
    )
    state = make_state_factory(body="UNIQUE-BODY-MARKER everything is broken")
    classifier.run(state)

    calls = fake_llm.calls_for(Classification)
    assert len(calls) == 1
    _, _system, user = calls[0]
    assert "UNIQUE-BODY-MARKER" in user
