from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

Priority = Literal["low", "medium", "high", "urgent"]
Sentiment = Literal["positive", "neutral", "negative", "frustrated"]
Intent = Literal[
    "billing",
    "technical_issue",
    "account_access",
    "feature_request",
    "complaint",
    "general_inquiry",
    "cancellation",
    "other",
]
Route = Literal["resolve_with_kb", "resolve_direct", "escalate"]
QAVerdict = Literal["pass", "revise", "escalate"]


class Customer(BaseModel):
    id: str
    name: Optional[str] = None
    plan: Optional[str] = None
    tenure_days: Optional[int] = None
    history_summary: Optional[str] = None


class Ticket(BaseModel):
    id: str
    channel: Literal["email", "chat", "web", "api"] = "web"
    subject: Optional[str] = None
    body: str
    customer: Customer
    conversation_history: list[str] = Field(default_factory=list)


class Classification(BaseModel):
    intent: Intent
    priority: Priority
    sentiment: Sentiment
    language: str = "en"
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)


class Plan(BaseModel):
    route: Route
    needs_kb: bool
    rationale: str


class KBChunk(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float


class DraftReply(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)


class QAResult(BaseModel):
    verdict: QAVerdict
    issues: list[str] = Field(default_factory=list)
    feedback: Optional[str] = None


class EscalationPacket(BaseModel):
    summary: str
    suggested_next_steps: list[str]
    customer_facing_acknowledgement: str
    severity: Priority


class CXInsights(BaseModel):
    churn_risk: Literal["low", "medium", "high"]
    satisfaction_signal: Sentiment
    tags: list[str] = Field(default_factory=list)
    notes: Optional[str] = None


class LLMCall(BaseModel):
    """One LLM round-trip — emitted by core.llm.structured for telemetry."""
    agent: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    attempt: int = 1
    error: Optional[str] = None


class TicketState(BaseModel):
    ticket: Ticket

    classification: Optional[Classification] = None
    plan: Optional[Plan] = None
    kb_chunks: list[KBChunk] = Field(default_factory=list)
    draft: Optional[DraftReply] = None
    qa: Optional[QAResult] = None
    escalation: Optional[EscalationPacket] = None
    cx: Optional[CXInsights] = None

    qa_attempts: int = 0
    final_response: Optional[str] = None
    final_route: Optional[Literal["auto_reply", "escalated"]] = None
    trace: list[str] = Field(default_factory=list)

    llm_calls: list[LLMCall] = Field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(c.cost_usd for c in self.llm_calls), 6)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.llm_calls)

    @property
    def total_latency_ms(self) -> int:
        return sum(c.latency_ms for c in self.llm_calls)

    def log(self, agent: str, msg: str) -> None:
        self.trace.append(f"[{agent}] {msg}")
