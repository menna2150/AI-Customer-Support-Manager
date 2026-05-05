"""Heuristic grounding check for the QA agent.

For each sentence in the draft reply, we look for **factual claims** — numbers,
dates, percentages, currency amounts, named policies — and verify they appear
in at least one of the retrieved KB chunks.

This is intentionally simple (substring matching after normalisation). It will
miss paraphrased numbers ("twelve" vs "12") but it's a strong floor: if the
Resolver invents a "60-day refund window" that's not in any chunk, this catches it.
"""
from __future__ import annotations
import re

# Sentences that mention any of these tokens are treated as factual claims worth verifying.
_POLICY_KEYWORDS = re.compile(
    r"\b(refund|cancel(?:lation)?|policy|fee|charge|plan|tier|sso|2fa|"
    r"day|days|week|weeks|month|months|year|years|hour|hours|"
    r"percent|%|free|paid|trial|pro|enterprise|invoice|receipt|"
    r"reset|password|account|api|rate.?limit)\b",
    re.I,
)

# A "number" is any standalone integer or percentage. We deliberately ignore
# tiny single-digit numbers that show up in step lists ("1.", "2.", "3.").
_NUMBER_PATTERN = re.compile(r"\b(\d{2,}|[2-9])\b")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def find_factual_claims(draft: str) -> list[str]:
    """Sentences worth checking — those that contain numbers OR policy keywords."""
    out: list[str] = []
    for sent in split_sentences(draft):
        if _NUMBER_PATTERN.search(sent) or _POLICY_KEYWORDS.search(sent):
            out.append(sent)
    return out


def check_grounding(draft: str, chunk_texts: list[str]) -> list[str]:
    """Return a list of human-readable problems where draft claims aren't grounded.

    Empty list => everything in the draft is supported by at least one chunk.
    """
    if not draft or not chunk_texts:
        return []

    haystack = _normalize(" ".join(chunk_texts))
    issues: list[str] = []

    for sent in find_factual_claims(draft):
        norm_sent = _normalize(sent)

        # 1. Every standalone number in the sentence must appear in the chunks.
        numbers = _NUMBER_PATTERN.findall(sent)
        for n in numbers:
            if n not in haystack:
                snippet = sent[:90] + ("..." if len(sent) > 90 else "")
                issues.append(
                    f"Reply contains number '{n}' not found in any KB snippet: \"{snippet}\""
                )
                break  # one issue per sentence is enough

        if numbers and any(f"'{n}'" in i for n in numbers for i in issues[-1:]):
            continue  # already flagged this sentence

        # 2. Sentences with policy keywords but no numbers — require keyword overlap with chunks.
        if not numbers and _POLICY_KEYWORDS.search(sent):
            words = set(re.findall(r"[a-z]{4,}", norm_sent))
            stopwords = {
                "your", "please", "thank", "with", "from", "this", "that", "have",
                "will", "would", "could", "should", "team", "help", "happy", "support",
                "customer", "regards", "kind", "best", "issue", "follow",
            }
            salient = words - stopwords
            if salient:
                hits = sum(1 for w in salient if w in haystack)
                if hits / len(salient) < 0.30:
                    snippet = sent[:90] + ("..." if len(sent) > 90 else "")
                    issues.append(
                        f"Reply makes a policy claim with no clear support in KB: \"{snippet}\""
                    )

    return issues
