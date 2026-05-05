"""Classifier Agent — extract intent, priority, sentiment.

Improves accuracy via few-shot examples drawn from the real Bitext dataset.
The examples are loaded once and cached.
"""
from __future__ import annotations
from functools import lru_cache

from core.llm import structured
from core.state import Classification, TicketState

BASE_SYSTEM = """You are a triage classifier for a SaaS customer support team.
Read one ticket and produce a structured classification.

Guidelines:
- intent: pick the SINGLE best label.
- priority: 'urgent' = service unusable, security, or revenue-impacting; 'high' = blocked workflow;
  'medium' = degraded but workable; 'low' = informational or feature request.
- sentiment: 'frustrated' if there are explicit signs of anger, threats to leave, or repeated attempts;
  'negative' for unhappy but composed; 'neutral' for matter-of-fact; 'positive' for praise.
- confidence: how sure you are about intent + priority together.
- summary: one sentence describing what the customer wants.
- language: ISO code (e.g. 'en', 'ar', 'es'). Detect from the body.

Below are real, labeled examples — use them as calibration. Match style/intent labels carefully."""


@lru_cache(maxsize=1)
def _few_shot_block() -> str:
    """Build the few-shot block lazily so tests don't need the dataset."""
    try:
        from data.bitext_loader import load_few_shot_examples

        examples = load_few_shot_examples(per_intent=1)
    except Exception:
        return ""

    seen_intents: set[str] = set()
    lines: list[str] = []
    for ex in examples:
        if ex["intent"] in seen_intents:
            continue
        seen_intents.add(ex["intent"])
        lines.append(
            f"Example — message: \"{ex['instruction']}\" -> intent: {ex['intent']}"
        )
        if len(seen_intents) >= 8:
            break
    if not lines:
        return ""
    return "\n\nCalibration examples:\n" + "\n".join(lines)


def _system_prompt() -> str:
    return BASE_SYSTEM + _few_shot_block()


def run(state: TicketState) -> TicketState:
    t = state.ticket
    history = "\n".join(t.conversation_history) or "(none)"
    user = f"""Ticket id: {t.id}
Channel: {t.channel}
Subject: {t.subject or "(no subject)"}
Customer plan: {t.customer.plan or "unknown"}, tenure_days: {t.customer.tenure_days or "unknown"}
Customer history: {t.customer.history_summary or "(none)"}
Prior conversation:
{history}

Body:
\"\"\"{t.body}\"\"\""""

    cls = structured(_system_prompt(), user, Classification, state=state, agent="classifier")
    state.classification = cls
    state.log(
        "classifier",
        f"intent={cls.intent} priority={cls.priority} sentiment={cls.sentiment} "
        f"conf={cls.confidence:.2f}",
    )
    return state
