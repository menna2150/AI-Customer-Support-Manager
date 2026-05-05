"""Shared fixtures: project on sys.path, FakeLLM, sample TicketState, stubbed retriever."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib  # noqa: E402
from typing import Any, Callable  # noqa: E402

import pytest  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from core.state import (  # noqa: E402
    Customer,
    KBChunk,
    Ticket,
    TicketState,
)


# ---------------------------------------------------------------------------
# FakeLLM: stand-in for core.llm.structured
# ---------------------------------------------------------------------------

class FakeLLM:
    """Records every structured() call and returns canned responses by schema name.

    Usage:
        fake.respond(Classification, Classification(...))     # static
        fake.respond(QAResult, [revise_result, pass_result])  # queue, popped per call
        fake.respond(Plan, lambda system, user, schema: ...)  # callable
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self._responders: dict[str, Any] = {}

    def respond(self, schema_cls: type[BaseModel], value: Any) -> None:
        self._responders[schema_cls.__name__] = value

    def structured(
        self,
        system: str,
        user: str,
        schema: type[BaseModel],
        **_: Any,
    ) -> BaseModel:
        name = schema.__name__
        self.calls.append((name, system, user))
        if name not in self._responders:
            raise AssertionError(
                f"FakeLLM: no response registered for schema '{name}'. "
                f"Call fake_llm.respond({name}, ...) in your test."
            )
        r = self._responders[name]
        if isinstance(r, list):
            if not r:
                raise AssertionError(f"FakeLLM: queue exhausted for '{name}'")
            return r.pop(0)
        if callable(r) and not isinstance(r, BaseModel):
            return r(system, user, schema)
        return r

    def calls_for(self, schema_cls: type[BaseModel]) -> list[tuple[str, str, str]]:
        return [c for c in self.calls if c[0] == schema_cls.__name__]


_AGENT_MODULES = (
    "agents.classifier",
    "agents.planner",
    "agents.resolver",
    "agents.qa",
    "agents.escalation",
    "agents.cx_analyst",
)


@pytest.fixture(autouse=True)
def _disable_classifier_few_shot(monkeypatch):
    """Tests must not hit the network for the Bitext few-shot block."""
    import agents.classifier as clf

    monkeypatch.setattr(clf, "_few_shot_block", lambda: "")


@pytest.fixture
def fake_llm(monkeypatch) -> FakeLLM:
    """Patch `structured` everywhere it's bound (each agent does `from core.llm import structured`)."""
    fake = FakeLLM()
    import core.llm

    monkeypatch.setattr(core.llm, "structured", fake.structured)
    for mod_name in _AGENT_MODULES:
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "structured"):
            monkeypatch.setattr(mod, "structured", fake.structured)
    return fake


# ---------------------------------------------------------------------------
# Stub retriever (no Chroma needed in tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_retrieve(monkeypatch):
    """Replace `agents.rag.retrieve` with a stub. Returns a setter for canned chunks."""
    import agents.rag as rag_mod

    holder: dict[str, list[KBChunk]] = {"chunks": []}
    captured: dict[str, Any] = {}

    def _retrieve(query: str, k: int = 4, *, intent: str | None = None) -> list[KBChunk]:
        captured["query"] = query
        captured["k"] = k
        captured["intent"] = intent
        return list(holder["chunks"])

    monkeypatch.setattr(rag_mod, "retrieve", _retrieve)

    class Handle:
        def set(self, chunks: list[KBChunk]) -> None:
            holder["chunks"] = chunks

        @property
        def last_query(self) -> str | None:
            return captured.get("query")

        @property
        def last_k(self) -> int | None:
            return captured.get("k")

        @property
        def last_intent(self) -> str | None:
            return captured.get("intent")

    return Handle()


# ---------------------------------------------------------------------------
# Sample factories
# ---------------------------------------------------------------------------

def make_state(
    body: str = "I want a refund for my annual plan, only 10 days in.",
    **overrides,
) -> TicketState:
    customer = Customer(
        id=overrides.get("customer_id", "U-1"),
        name=overrides.get("customer_name", "Test User"),
        plan=overrides.get("plan", "annual_pro"),
        tenure_days=overrides.get("tenure_days", 10),
        history_summary=overrides.get("history_summary", None),
    )
    ticket = Ticket(
        id=overrides.get("ticket_id", "T-TEST"),
        channel=overrides.get("channel", "email"),
        subject=overrides.get("subject", "test"),
        body=body,
        customer=customer,
        conversation_history=overrides.get("conversation_history", []),
    )
    return TicketState(ticket=ticket)


@pytest.fixture
def sample_state() -> TicketState:
    return make_state()


@pytest.fixture
def make_state_factory():
    return make_state
