# AI Customer Support Manager

A multi-agent system that triages, drafts replies to, or escalates customer support tickets. Built with LangGraph, evaluated on real Bitext data, served through a Streamlit UI.

---

## What this is

When a customer message comes in, seven specialised AI agents process it together:

1. **Classifier** — extracts intent, priority, sentiment, language
2. **Planner** — decides whether to answer with the knowledge base, answer directly, or hand off to a human
3. **RAG** — retrieves relevant snippets from the knowledge base, filtered by intent
4. **Resolver** — drafts a customer-facing reply grounded in the retrieved snippets
5. **QA** — reviews the draft for accuracy, tone, and policy; can loop back to the Resolver
6. **Escalation** — when a human is needed, builds a handoff packet (summary, next steps, customer acknowledgement)
7. **CX Analyst** — tags churn risk and satisfaction signals for analytics

Shared state flows between agents as a single typed `TicketState` (Pydantic). Routing is conditional, with a bounded QA retry loop.

## Who this is for

This is a tool for a **support team** or its manager — not for the customer. The customer only ever sees the final reply. The team uses the Streamlit UI to monitor what the AI is doing, test new types of tickets, and inspect escalations.

## Architecture

```
ticket -> Classifier -> Planner -> [ RAG -> Resolver -> QA ]  -> reply to customer
                              \                          \
                               -> Escalation -> handoff to human
                                                                 \
                                                                  -> CX Analyst (always runs)
```

| Layer | Tech |
|---|---|
| Orchestration | LangGraph (StateGraph with conditional edges + QA loop) |
| LLM | Groq (free tier, Llama family) or Anthropic Claude — selected via `LLM_PROVIDER` |
| Structured outputs | Provider-native tool use, returning Pydantic models |
| Vector store | ChromaDB (persistent, local) |
| Embeddings | sentence-transformers MiniLM (also used by the offline kNN classifier) |
| UI | Streamlit |

## Real data

The system uses the public **Bitext Customer Support** dataset (~27k labelled tickets) for three things:

1. **Eval tickets** — held-out test split, sampled by `data/bitext_loader.load_eval_tickets`.
2. **Few-shot calibration** — one real example per intent injected into the Classifier system prompt.
3. **Knowledge base cases** — when ingesting with `--with-bitext`, real (instruction, response) pairs are stored as KB chunks tagged with intent metadata, so RAG can match incoming tickets against canonical past resolutions.

The CSV is downloaded once on first use and cached under `data/cache/`.

## Measured accuracy

Two evaluation paths are wired up:

- **Offline kNN classifier** — sentence-transformers embeddings + cosine-similarity vote against a deterministic Bitext train split. Runs without any API key.
- **LLM classifier** — the actual Classifier agent against the same held-out set, for direct comparison.

```
python -m eval.classifier_eval_knn --n 50
```

On a 50-ticket held-out sample with seed=7, the offline kNN classifier achieves **98.0% accuracy** (49/50). Confidence is well-calibrated: the single miss had confidence 0.55 while every correct prediction scored 1.00. Per-intent breakdown:

| intent | n | correct | accuracy |
|---|---:|---:|---:|
| account_access | 8 | 8 | 100% |
| billing | 14 | 14 | 100% |
| cancellation | 2 | 2 | 100% |
| complaint | 1 | 1 | 100% |
| general_inquiry | 22 | 21 | 95% |
| other | 3 | 3 | 100% |

For the LLM eval, set a Groq or Anthropic key and run:

```
python -m eval.classifier_eval --n 50
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env
# Edit .env and set GROQ_API_KEY (free) or ANTHROPIC_API_KEY (paid).

# Build the knowledge base index. --with-bitext folds in real (q, a) cases.
python -m knowledge_base.ingest --with-bitext --max-per-intent 60
```

`.env` highlights:

```
LLM_PROVIDER=groq                          # or "anthropic"
GROQ_API_KEY=gsk_...                       # https://console.groq.com/keys
GROQ_MODEL=llama-3.3-70b-versatile         # or llama-3.1-8b-instant for a smaller free quota
ANTHROPIC_API_KEY=
CLAUDE_MODEL=claude-sonnet-4-6
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DIR=./knowledge_base/store
KB_COLLECTION=support_kb
```

## Run

### Streamlit UI

```powershell
streamlit run app.py
```

The sidebar has six one-click examples (refund question, frustrated customer, technical issue, praise, SSO setup, Arabic ticket) plus a form for writing your own ticket. The main panel shows the customer message, a plain-English step-by-step list of what each agent did, the final reply or escalation packet, and customer-health metrics. Toggle "Show technical details" in the sidebar to see classification JSON, plan, retrieved KB chunks, QA verdict, and the full trace.

### CLI

```powershell
python main.py                              # interactive
python main.py --ticket-id T001             # bundled sample
python main.py --all                        # every sample
```

### No-LLM demo

Runs Classifier (offline kNN) + rule-based Planner + RAG retrieval. No API key needed.

```powershell
python demo_no_llm.py --n 3
python demo_no_llm.py --ticket "I need a refund for the annual plan I bought yesterday"
```

## Tests

31 unit tests covering each agent and every conditional edge in the graph. The suite mocks the LLM (`core.llm.structured`), the retriever (`agents.rag.retrieve`), and the Bitext few-shot loader, so it runs offline in under a second with no API key.

```powershell
pip install -r requirements-dev.txt
pytest
```

## Project layout

```
.
+-- agents/                seven specialised agents
+-- core/
|   +-- state.py           Pydantic TicketState
|   +-- llm.py             provider-agnostic LLM wrapper with retry
|   +-- graph.py           LangGraph workflow
|   +-- knn_classifier.py  offline embedding-kNN (no LLM required)
+-- data/
|   +-- bitext_loader.py   downloads + parses Bitext (cached)
|   +-- intent_map.py      Bitext labels -> our Intent enum
|   +-- sample_tickets.json
|   +-- cache/             gitignored
+-- eval/
|   +-- classifier_eval.py     LLM-based eval
|   +-- classifier_eval_knn.py offline kNN eval
+-- knowledge_base/
|   +-- docs/              FAQ markdown sources
|   +-- store/             Chroma persistent index (gitignored)
|   +-- ingest.py          supports --with-bitext
|   +-- retriever.py       hybrid intent-filtered + unfiltered merge
+-- tests/
+-- app.py                 Streamlit UI
+-- main.py                CLI
+-- demo_no_llm.py         no-LLM demo
+-- examples/walkthrough.py
+-- config.py
```

## Honest limitations

This project is at proof-of-concept maturity. Concrete gaps before it could handle real customer tickets:

- **Over-escalation on small models.** The Planner prompt is conservative ("if unsure, escalate") and combined with a smaller model (Llama 8B) it escalates more than needed. Larger models (Llama 70B, Claude Sonnet) escalate less.
- **No real hallucination guard.** The QA agent checks tone and obvious errors but does not verify that every factual claim in the draft maps to a retrieved chunk.
- **No memory.** Each ticket is processed independently; the same customer asking three times in a row is treated as three strangers.
- **No tool use.** The Resolver only drafts text. It cannot look up an account, issue a refund, or update a ticket system.
- **No PII redaction, no cost tracking, no audit trail.** All required for handling real customer data.
- **Knowledge base is small.** Five policy markdown docs + 360 Bitext examples. A real product needs hundreds of policy chunks, properly versioned.

The architecture supports each of these as additive changes — no rewrite needed.

## Provider notes

- **Groq free tier** — fast (sub-second per call), generous daily quota, but Llama models occasionally produce malformed JSON in tool calls. The LLM wrapper retries with bumped temperature on `BadRequestError` / `ValidationError`.
- **Anthropic** — Claude is more reliable on tool use and language nuance, but paid.
- **Offline mode** — `demo_no_llm.py` and `eval/classifier_eval_knn.py` run with no provider at all, using sentence-transformers embeddings.
