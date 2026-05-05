"""Unified LLM wrapper. Two providers, same interface.

  • Groq      — free tier, Llama 3.x via OpenAI-compatible endpoint
  • Anthropic — Claude (paid)

`structured(system, user, schema)` returns a Pydantic instance, retrying once
on schema-validation failure (Llama can occasionally emit malformed JSON).

Telemetry: when called with `state=` and `agent=`, records an `LLMCall` into
`state.llm_calls` for cost / token / latency tracking.
"""
from __future__ import annotations
import json
import time
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    LLM_RETRIES,
)
from core.state import LLMCall, TicketState

T = TypeVar("T", bound=BaseModel)


# Approximate USD per 1M tokens (input, output). Groq's free tier is $0.
PRICING: dict[str, tuple[float, float]] = {
    # Groq — free tier
    "llama-3.3-70b-versatile": (0.0, 0.0),
    "llama-3.1-8b-instant": (0.0, 0.0),
    "gemma2-9b-it": (0.0, 0.0),
    "meta-llama/llama-4-scout-17b-16e-instruct": (0.0, 0.0),
    "meta-llama/llama-4-maverick-17b-128e-instruct": (0.0, 0.0),
    # Anthropic (approximate, public pricing)
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-7": (15.0, 75.0),
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_price, out_price = PRICING.get(model, (0.0, 0.0))
    return (prompt_tokens / 1_000_000) * in_price + (completion_tokens / 1_000_000) * out_price


# ---------------------------------------------------------------------------
# Lazy clients
# ---------------------------------------------------------------------------

_anthropic_client = None
_groq_client = None


def _anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        from anthropic import Anthropic

        _anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def _groq():
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set — get one at https://console.groq.com")
        from openai import OpenAI

        _groq_client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
    return _groq_client


# ---------------------------------------------------------------------------
# chat() — free-form text response
# ---------------------------------------------------------------------------

def chat(system: str, user: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
    if LLM_PROVIDER == "anthropic":
        resp = _anthropic().messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    resp = _groq().chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# structured() — Pydantic-shaped response via tool-use
# ---------------------------------------------------------------------------

def _call_anthropic_tool(
    system: str, user: str, schema: Type[T], *, max_tokens: int, temperature: float
) -> tuple[T, dict[str, int]]:
    tool_name = f"emit_{schema.__name__.lower()}"
    json_schema = schema.model_json_schema()
    resp = _anthropic().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        tools=[
            {
                "name": tool_name,
                "description": f"Return a {schema.__name__} object.",
                "input_schema": json_schema,
            }
        ],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": user}],
    )
    usage = {
        "prompt_tokens": getattr(resp.usage, "input_tokens", 0),
        "completion_tokens": getattr(resp.usage, "output_tokens", 0),
    }
    for block in resp.content:
        if block.type == "tool_use" and block.name == tool_name:
            return schema.model_validate(block.input), usage
    raise RuntimeError(f"Anthropic did not emit tool {tool_name}: {resp.content!r}")


def _call_groq_tool(
    system: str, user: str, schema: Type[T], *, max_tokens: int, temperature: float
) -> tuple[T, dict[str, int]]:
    tool_name = f"emit_{schema.__name__.lower()}"
    json_schema = schema.model_json_schema()
    resp = _groq().chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": f"Return a {schema.__name__} object.",
                    "parameters": json_schema,
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": tool_name}},
    )
    usage = {
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
        "completion_tokens": getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
    }
    msg = resp.choices[0].message
    if not msg.tool_calls:
        raise RuntimeError(f"Groq did not call tool {tool_name}; message: {msg!r}")
    args_json = msg.tool_calls[0].function.arguments
    try:
        args = json.loads(args_json)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Groq returned invalid JSON for {tool_name}: {e}; raw={args_json!r}")
    return schema.model_validate(args), usage


def _record(
    state: TicketState | None,
    agent: str | None,
    provider: str,
    model: str,
    usage: dict[str, int],
    latency_ms: int,
    attempt: int,
    error: str | None = None,
):
    if state is None or agent is None:
        return
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    state.llm_calls.append(
        LLMCall(
            agent=agent,
            provider=provider,
            model=model,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=pt + ct,
            latency_ms=latency_ms,
            cost_usd=round(estimate_cost_usd(model, pt, ct), 8),
            attempt=attempt,
            error=error,
        )
    )


def structured(
    system: str,
    user: str,
    schema: Type[T],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    state: Any = None,
    agent: str | None = None,
) -> T:
    last_err: Exception | None = None
    provider = LLM_PROVIDER
    model = CLAUDE_MODEL if provider == "anthropic" else GROQ_MODEL

    for attempt in range(LLM_RETRIES + 1):
        t = temperature + (0.1 * attempt)
        start = time.perf_counter()
        try:
            if provider == "anthropic":
                result, usage = _call_anthropic_tool(
                    system, user, schema, max_tokens=max_tokens, temperature=t
                )
            else:
                result, usage = _call_groq_tool(
                    system, user, schema, max_tokens=max_tokens, temperature=t
                )
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            _record(state, agent, provider, model, usage, elapsed_ms, attempt + 1)
            return result
        except (ValidationError, RuntimeError, json.JSONDecodeError) as e:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            _record(state, agent, provider, model, {}, elapsed_ms, attempt + 1, error=str(e)[:200])
            last_err = e
            if attempt == LLM_RETRIES:
                raise
        except Exception as e:
            cls_name = type(e).__name__
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            if cls_name in {"BadRequestError", "APIStatusError", "APIError"}:
                _record(state, agent, provider, model, {}, elapsed_ms, attempt + 1, error=str(e)[:200])
                last_err = e
                if attempt == LLM_RETRIES:
                    raise
            else:
                _record(state, agent, provider, model, {}, elapsed_ms, attempt + 1, error=str(e)[:200])
                raise
    assert last_err is not None
    raise last_err
