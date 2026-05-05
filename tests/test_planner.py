import pytest

from agents import planner
from core.state import Classification, Plan


def _classified(state, **kw):
    defaults = dict(
        intent="billing",
        priority="medium",
        sentiment="neutral",
        language="en",
        summary="refund",
        confidence=0.9,
    )
    defaults.update(kw)
    state.classification = Classification(**defaults)
    return state


def test_planner_requires_classification(fake_llm, sample_state):
    with pytest.raises(AssertionError):
        planner.run(sample_state)


def test_planner_routes_to_kb(fake_llm, sample_state):
    _classified(sample_state)
    fake_llm.respond(
        Plan,
        Plan(route="resolve_with_kb", needs_kb=True, rationale="Standard refund policy question."),
    )
    out = planner.run(sample_state)
    assert out.plan.route == "resolve_with_kb"
    assert out.plan.needs_kb is True


def test_planner_can_escalate(fake_llm, sample_state):
    _classified(sample_state, sentiment="frustrated", priority="urgent")
    fake_llm.respond(
        Plan,
        Plan(route="escalate", needs_kb=False, rationale="Frustrated + urgent."),
    )
    out = planner.run(sample_state)
    assert out.plan.route == "escalate"
