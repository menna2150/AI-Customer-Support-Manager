"""QA Agent — validate accuracy, tone, and policy of the draft. May loop back to Resolver.

Two-layer check:
  1. Programmatic grounding gate (core.grounding) — catches fabricated numbers
     and unsupported policy claims with deterministic substring matching. Forces
     `revise` if any claim isn't grounded in retrieved chunks.
  2. LLM verdict — catches the things the heuristic misses (tone, language,
     leaked PII). Defaults to 'pass'.
"""
from __future__ import annotations

from core.grounding import check_grounding
from core.llm import structured
from core.state import QAResult, TicketState

SYSTEM = """You are a quality reviewer. Judge a draft reply against the source KB snippets.

DEFAULT TO 'pass'. Approve drafts that are factually consistent with the KB and not rude.
DO NOT revise for style preferences (could be more empathetic, slightly different phrasing,
brevity). The Resolver's job is to be correct, not perfect.

Verdict rules:
- 'pass'     : reply is factually consistent with the KB and not openly rude. THIS IS THE DEFAULT.
- 'revise'   : there is a SPECIFIC, NAMEABLE problem. State the exact issue.
- 'escalate' : a legal/security/compliance issue is raised, or the request is genuinely outside
              what the KB can answer safely. Use sparingly.

ONLY these are valid revision triggers:
  1. The reply states a number, date, percentage, or named policy that does NOT appear in any KB snippet.
  2. The reply omits a refund/cancellation step that the KB explicitly requires.
  3. The reply discloses another customer's data or internal system details.
  4. The reply is in the wrong language (must match the customer's language).
  5. The reply is openly rude, dismissive, or blames the customer.

NOT valid revision reasons:
  - "Could be more empathetic" / "tone could be warmer"
  - "Vague" / "lacks specific guidance" — only revise if a SPECIFIC required step is missing
  - "Could mention X" — only revise if X is REQUIRED
  - Slight phrasing differences from the KB"""


def _format_chunks(state: TicketState) -> str:
    if not state.kb_chunks:
        return "(no KB snippets — verify the reply is purely generic)"
    return "\n\n".join(f"[{c.doc_id}] {c.title}: {c.text}" for c in state.kb_chunks)


def run(state: TicketState) -> TicketState:
    assert state.draft is not None, "QA requires a draft"

    # 1. Programmatic grounding gate — runs before the LLM.
    grounding_issues: list[str] = []
    if state.kb_chunks:
        grounding_issues = check_grounding(
            state.draft.text, [c.text for c in state.kb_chunks]
        )

    if grounding_issues:
        state.qa = QAResult(
            verdict="revise",
            issues=grounding_issues,
            feedback=(
                "The grounding check found claims in the draft that aren't supported by any "
                "retrieved KB snippet. Rewrite the reply using ONLY information that appears "
                "verbatim in the snippets, or omit the unsupported claim."
            ),
        )
        state.qa_attempts += 1
        state.log(
            "qa",
            f"verdict=revise (grounding gate) attempts={state.qa_attempts} "
            f"issues={len(grounding_issues)}",
        )
        return state

    # 2. LLM verdict — for things the heuristic misses (tone, language, leaked data).
    cls = state.classification
    user = f"""Customer language: {cls.language if cls else "en"}
Customer sentiment: {cls.sentiment if cls else "neutral"}
Original ticket body:
\"\"\"{state.ticket.body}\"\"\"

KB snippets used:
{_format_chunks(state)}

Draft reply:
\"\"\"{state.draft.text}\"\"\"
Cited doc_ids: {state.draft.citations}

The grounding gate already verified that every number/policy claim in the draft is
supported by the KB. You only need to judge tone, language match, and any leaked PII.

Review the draft and emit a verdict."""

    qa = structured(SYSTEM, user, QAResult, state=state, agent="qa")
    state.qa = qa
    state.qa_attempts += 1
    state.log(
        "qa",
        f"verdict={qa.verdict} attempts={state.qa_attempts} issues={qa.issues}",
    )
    return state
