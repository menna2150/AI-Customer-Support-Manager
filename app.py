"""Streamlit UI — clean version.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from core.graph import run_ticket
from core.state import Customer, Ticket, TicketState

st.set_page_config(
    page_title="AI Customer Support",
    layout="wide",
    initial_sidebar_state="expanded",
)


EXAMPLES = [
    (
        "Refund question",
        "Refund for annual plan",
        "I subscribed to the annual plan 12 days ago and changed my mind. Can I get a refund?",
        "annual_pro",
    ),
    (
        "Frustrated customer",
        "Third time complaining",
        "This is the THIRD time I'm writing about being double-charged. I want my money back today "
        "or I'm disputing with my bank. Get me a real human.",
        "monthly_pro",
    ),
    (
        "Technical issue",
        "Sync error",
        "I keep getting sync error code SYNC-409 in our marketing workspace, signed out and back "
        "in but it's still broken. We're launching tomorrow.",
        "team",
    ),
    (
        "Praise",
        "Loving the new dashboard",
        "Just wanted to say the new dashboard is awesome - my team is way faster now. Keep it up!",
        "team",
    ),
    (
        "SSO setup",
        "SAML SSO with Okta",
        "We're rolling out the product to our org and need SAML SSO with Okta. What's the setup process?",
        "enterprise",
    ),
    (
        "Arabic ticket",
        "Refund in Arabic",
        "أريد إلغاء اشتراكي السنوي واسترداد المبلغ، لقد مر أسبوع فقط على الاشتراك.",
        "annual_pro",
    ),
]


def make_state(body: str, subject: str = "Ticket", plan: str = "monthly_pro") -> TicketState:
    return TicketState(
        ticket=Ticket(
            id="UI-LIVE",
            channel="web",
            subject=subject,
            body=body,
            customer=Customer(id="UI-USER", plan=plan),
        )
    )


def step_line(state: TicketState, agent: str) -> str | None:
    if agent == "classifier" and state.classification:
        c = state.classification
        return (
            f"Understood the ticket. Topic: {c.intent.replace('_', ' ')}. "
            f"Customer mood: {c.sentiment}. Priority: {c.priority}."
        )
    if agent == "planner" and state.plan:
        readable = {
            "resolve_with_kb": "answer using the knowledge base",
            "resolve_direct": "answer directly without the knowledge base",
            "escalate": "send to a human teammate",
        }.get(state.plan.route, state.plan.route)
        return f"Decided strategy: {readable}."
    if agent == "rag" and state.kb_chunks:
        return f"Searched the knowledge base. Found {len(state.kb_chunks)} relevant articles."
    if agent == "resolver" and state.draft:
        return f"Wrote a reply ({len(state.draft.text)} characters)."
    if agent == "qa" and state.qa:
        verdict = {
            "pass": "passed - reply approved",
            "revise": "needs revision - sending back",
            "escalate": "escalating to a human",
        }.get(state.qa.verdict, state.qa.verdict)
        return f"Quality check: {verdict}."
    if agent == "escalation" and state.escalation:
        return f"Built handoff packet for a human (severity: {state.escalation.severity})."
    if agent == "cx_analyst" and state.cx:
        return f"Analyzed customer health. Churn risk: {state.cx.churn_risk}."
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("AI Customer Support")
st.sidebar.caption("Pick an example or write your own ticket.")

if "selected_state" not in st.session_state:
    st.session_state.selected_state = None
if "show_details" not in st.session_state:
    st.session_state.show_details = False

st.sidebar.markdown("**Examples**")
for label, subject, body, plan in EXAMPLES:
    if st.sidebar.button(label, use_container_width=True, key=f"ex_{label}"):
        st.session_state.selected_state = make_state(body, subject, plan)

st.sidebar.markdown("---")
st.sidebar.markdown("**Write your own**")
with st.sidebar.form("custom_form", clear_on_submit=False):
    custom_body = st.text_area("Customer message", height=120, placeholder="Type a ticket...")
    custom_subject = st.text_input("Subject (optional)", value="")
    custom_plan = st.selectbox(
        "Customer plan",
        ["free", "monthly_pro", "annual_pro", "team", "enterprise"],
        index=1,
    )
    submitted = st.form_submit_button("Run", type="primary", use_container_width=True)
    if submitted and custom_body.strip():
        st.session_state.selected_state = make_state(
            custom_body.strip(), custom_subject.strip() or "Ticket", custom_plan
        )

st.sidebar.markdown("---")
st.session_state.show_details = st.sidebar.toggle(
    "Show technical details", value=st.session_state.show_details
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("AI Customer Support")
st.caption("Seven agents process a ticket: classify, plan, retrieve, draft, review, escalate if needed, log analytics.")

state_to_run: TicketState | None = st.session_state.selected_state

if state_to_run is None:
    st.info("Click an example on the left, or write your own and press Run.")
    st.stop()


# Customer message
t = state_to_run.ticket
with st.chat_message("user"):
    if t.subject:
        st.markdown(f"**{t.subject}**")
    st.markdown(t.body)
    st.caption(f"Customer {t.customer.id}, plan {t.customer.plan or 'unknown'}, channel {t.channel}")

# Run pipeline
with st.spinner("Running..."):
    try:
        final = run_ticket(state_to_run)
    except Exception as e:
        st.error("Pipeline failed.")
        st.exception(e)
        st.stop()


# Plain-English progress
st.markdown("### What the AI did")
sequence = ["classifier", "planner", "rag", "resolver", "qa", "escalation", "cx_analyst"]
for a in sequence:
    line = step_line(final, a)
    if line:
        st.markdown(f"- {line}")

st.markdown("---")

# Outcome
if final.final_route == "auto_reply" and final.draft:
    st.markdown("### Reply that will be sent to the customer")
    with st.chat_message("assistant"):
        st.markdown(final.draft.text)
        if final.draft.citations:
            st.caption("Based on: " + ", ".join(f"`{c}`" for c in final.draft.citations))
    st.success("Auto-approved by quality check.")

elif final.final_route == "escalated" and final.escalation:
    e = final.escalation
    st.markdown(f"### Escalated to a human (severity: {e.severity})")

    st.markdown("**Short message sent to the customer right away:**")
    with st.chat_message("assistant"):
        st.markdown(e.customer_facing_acknowledgement)

    with st.container(border=True):
        st.markdown("**Briefing for the human teammate**")
        st.markdown(e.summary)
        st.markdown("**Suggested next steps:**")
        for i, s in enumerate(e.suggested_next_steps, 1):
            st.markdown(f"{i}. {s}")

# Customer health
if final.cx:
    cx = final.cx
    st.markdown("### Customer health")
    cols = st.columns(3)
    cols[0].metric("Churn risk", cx.churn_risk)
    cols[1].metric("Mood signal", cx.satisfaction_signal)
    cols[2].metric("Tags", ", ".join(cx.tags) if cx.tags else "-")
    if cx.notes:
        st.caption(cx.notes)

# Telemetry
if final.llm_calls:
    st.markdown("### Run cost")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LLM calls", len(final.llm_calls))
    c2.metric("Total tokens", f"{final.total_tokens:,}")
    c3.metric("Latency", f"{final.total_latency_ms / 1000:.1f}s")
    cost = final.total_cost_usd
    c4.metric("Cost", "free" if cost == 0 else f"${cost:.5f}")

# Technical details
if st.session_state.show_details:
    st.markdown("---")
    st.markdown("## Technical details")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Classification", "Plan", "Knowledge base", "QA", "Trace", "LLM calls"]
    )
    with tab1:
        if final.classification:
            st.json(final.classification.model_dump())
        else:
            st.write("(none)")
    with tab2:
        if final.plan:
            st.json(final.plan.model_dump())
        else:
            st.write("(none)")
    with tab3:
        if final.kb_chunks:
            for i, ch in enumerate(final.kb_chunks):
                with st.expander(f"{i+1}. [{ch.doc_id}] {ch.title}  -  score {ch.score:.2f}"):
                    st.code(ch.text, language="markdown")
        else:
            st.write("(no knowledge base lookup happened)")
    with tab4:
        if final.qa:
            st.json(final.qa.model_dump())
            st.caption(f"QA attempts: {final.qa_attempts}")
        else:
            st.write("(QA was skipped)")
    with tab5:
        st.code("\n".join(final.trace), language="text")
    with tab6:
        if final.llm_calls:
            rows = [
                {
                    "agent": c.agent,
                    "model": c.model,
                    "tokens_in": c.prompt_tokens,
                    "tokens_out": c.completion_tokens,
                    "latency_ms": c.latency_ms,
                    "cost_usd": f"{c.cost_usd:.6f}",
                    "attempt": c.attempt,
                    "error": c.error or "",
                }
                for c in final.llm_calls
            ]
            st.dataframe(rows, use_container_width=True)
        else:
            st.write("(no LLM calls were made)")

# Reset
st.markdown("---")
if st.button("Try another ticket"):
    st.session_state.selected_state = None
    st.rerun()
