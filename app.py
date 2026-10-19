"""Streamlit UI for the Travel Planner multi-agent system."""

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="assets/icon.png" if Path("assets/icon.png").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
CREAM       = "#FAF7F0"
CREAM_DARK  = "#EDE8DC"
CREAM_CARD  = "#F5F0E6"
ORANGE      = "#D97941"
ORANGE_LITE = "#EDA06A"
OLIVE       = "#5E7340"
OLIVE_LITE  = "#7D9858"
TEXT        = "#2C2218"
TEXT_MUTED  = "#7A6B5E"
BORDER      = "#CBBFA8"
SUCCESS     = "#4A7C3F"
WARN        = "#B85C20"

# ── SVG icon helpers ──────────────────────────────────────────────────────────
# Minimal inline SVGs (Heroicons-style, 16×16 viewBox)
def _svg(path_d: str, color: str = "currentColor", size: int = 16) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align:middle;margin-right:6px">{path_d}</svg>'
    )

ICO_PLANE   = _svg('<path d="M22 2L11 13"/><path d="M22 2L15 22L11 13L2 9L22 2z"/>')
ICO_BRAIN   = _svg('<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5V5a2 2 0 0 0 4 0v-.5A2.5 2.5 0 0 1 18.5 2h.5a2 2 0 0 1 2 2v1a4 4 0 0 1-4 4H7A4 4 0 0 1 3 5V4a2 2 0 0 1 2-2z"/><path d="M12 4.5V14"/><path d="M8 14v4a2 2 0 0 0 4 0v-4"/>')
ICO_SEARCH  = _svg('<circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>')
ICO_GLOBE   = _svg('<circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>')
ICO_SUN     = _svg('<circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>')
ICO_HOTEL   = _svg('<path d="M3 22V12a9 9 0 0 1 18 0v10"/><path d="M3 18h18"/><path d="M9 22v-4h6v4"/>')
ICO_CHECK   = _svg('<polyline points="20 6 9 17 4 12"/>', color=SUCCESS)
ICO_WARN    = _svg('<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>', color=WARN)
ICO_CLOCK   = _svg('<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>')
ICO_SETTINGS= _svg('<circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 1 0 0 14.14"/>')
ICO_DOWN    = _svg('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>')
ICO_NOTES   = _svg('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>')
ICO_LIST    = _svg('<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>')
ICO_MSG     = _svg('<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>')
ICO_DOT_ON  = f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{OLIVE};margin-right:6px;vertical-align:middle"></span>'
ICO_DOT_OFF = f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#C9B89A;margin-right:6px;vertical-align:middle"></span>'

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    /* ── Base ── */
    html, body, .stApp {{
        background: {CREAM} !important;
        color: {TEXT};
        font-family: "Inter", "Segoe UI", sans-serif;
    }}
    .main .block-container {{ padding-top: 1.5rem; max-width: 1100px; }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background: {CREAM_DARK} !important;
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stNumberInput input {{
        background: {CREAM} !important;
        border: 1px solid {BORDER} !important;
        color: {TEXT} !important;
        border-radius: 8px;
    }}

    /* ── Hero banner ── */
    .hero {{
        background: linear-gradient(135deg, {OLIVE} 0%, {OLIVE_LITE} 100%);
        border-radius: 14px;
        padding: 1.8rem 2.4rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(94,115,64,0.18);
    }}
    .hero h1 {{ color: {CREAM}; margin: 0; font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; }}
    .hero p  {{ color: rgba(250,247,240,0.82); margin: 0.4rem 0 0; font-size: 0.97rem; }}

    /* ── Inputs ── */
    .stTextArea textarea, .stTextInput input {{
        background: {CREAM_CARD} !important;
        border: 1.5px solid {BORDER} !important;
        border-radius: 10px !important;
        color: {TEXT} !important;
        font-size: 0.95rem !important;
    }}
    .stTextArea textarea:focus, .stTextInput input:focus {{
        border-color: {ORANGE} !important;
        box-shadow: 0 0 0 3px rgba(217,121,65,0.12) !important;
    }}

    /* ── Primary button ── */
    .stButton button[kind="primary"] {{
        background: {ORANGE} !important;
        color: {CREAM} !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.65rem 1.5rem !important;
        box-shadow: 0 3px 12px rgba(217,121,65,0.28) !important;
        transition: background 0.2s !important;
    }}
    .stButton button[kind="primary"]:hover {{
        background: {ORANGE_LITE} !important;
    }}

    /* ── Secondary (example) buttons ── */
    .stButton button:not([kind="primary"]) {{
        background: {CREAM_CARD} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        font-size: 0.83rem !important;
        text-align: left !important;
    }}
    .stButton button:not([kind="primary"]):hover {{
        border-color: {ORANGE} !important;
        color: {ORANGE} !important;
    }}

    /* ── Agent cards ── */
    .agent-card {{
        background: {CREAM_CARD};
        border: 1.5px solid {BORDER};
        border-radius: 11px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.2s, background 0.2s;
    }}
    .agent-card.running {{
        border-color: {ORANGE};
        background: #FFF4EC;
    }}
    .agent-card.done {{
        border-color: {OLIVE};
        background: #F0F5EB;
    }}
    .agent-card .card-label {{
        font-weight: 600;
        color: {TEXT};
        font-size: 0.88rem;
    }}
    .agent-card .card-status {{
        font-size: 0.78rem;
        margin-top: 2px;
    }}
    .status-running {{ color: {ORANGE}; }}
    .status-done    {{ color: {OLIVE}; }}
    .status-wait    {{ color: {TEXT_MUTED}; }}

    /* ── Metric row ── */
    .metric-row {{
        display: flex;
        gap: 0.8rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }}
    .metric-box {{
        background: {CREAM_CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 0.75rem 1.1rem;
        min-width: 120px;
        text-align: center;
        flex: 1;
    }}
    .metric-box .m-label {{
        color: {TEXT_MUTED};
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 500;
    }}
    .metric-box .m-value {{
        color: {TEXT};
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 2px;
    }}
    .metric-box .m-value.pass {{ color: {OLIVE}; }}
    .metric-box .m-value.warn {{ color: {ORANGE}; }}

    /* ── Plan box ── */
    .plan-box {{
        background: {CREAM_CARD};
        border: 1.5px solid {BORDER};
        border-left: 4px solid {OLIVE};
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        font-size: 0.91rem;
        line-height: 1.75;
        color: {TEXT};
        white-space: pre-wrap;
        max-height: 600px;
        overflow-y: auto;
    }}

    /* ── Section headers ── */
    .section-title {{
        color: {TEXT};
        font-size: 1.05rem;
        font-weight: 600;
        margin: 1.4rem 0 0.7rem;
        display: flex;
        align-items: center;
        gap: 6px;
        border-bottom: 2px solid {CREAM_DARK};
        padding-bottom: 0.4rem;
    }}

    /* ── Divider ── */
    hr {{ border-color: {BORDER} !important; }}

    /* ── Expander ── */
    .streamlit-expanderHeader {{
        background: {CREAM_CARD} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        color: {TEXT} !important;
    }}
    .streamlit-expanderContent {{
        background: {CREAM} !important;
        border: 1px solid {BORDER} !important;
        border-top: none !important;
    }}

    /* ── Success / warning alert ── */
    .custom-success {{
        background: #EDF5E8;
        border: 1px solid {OLIVE};
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        color: {SUCCESS};
        font-weight: 500;
        font-size: 0.93rem;
    }}
    .custom-warn {{
        background: #FFF0E6;
        border: 1px solid {ORANGE};
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        color: {WARN};
        font-weight: 500;
        font-size: 0.93rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result(timeout=180)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def run_planner(query: str, metadata: dict):
    from travel_planner.graph.workflow import run_travel_planner_async
    return _run_async(run_travel_planner_async(query, metadata, temperature=0.7))


def check_env():
    from travel_planner.config import (
        GROQ_API_KEY, OPENAI_API_KEY, TAVILY_API_KEY,
        has_chat_llm_credentials,
    )
    return {
        "Groq LLM":          bool(GROQ_API_KEY),
        "OpenAI (fallback)": bool(OPENAI_API_KEY),
        "Tavily Search":     bool(TAVILY_API_KEY),
        "LLM ready":         has_chat_llm_credentials(),
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f'<div class="section-title">{ICO_SETTINGS} API Status</div>',
        unsafe_allow_html=True,
    )
    env = check_env()
    for name, ok in env.items():
        dot = ICO_DOT_ON if ok else ICO_DOT_OFF
        st.markdown(f"{dot} <b>{name}</b>", unsafe_allow_html=True)

    st.divider()
    st.markdown(
        f'<div class="section-title">{ICO_PLANE} Trip Details</div>',
        unsafe_allow_html=True,
    )
    origin      = st.text_input("Origin city",      placeholder="e.g. Cairo")
    destination = st.text_input("Destination city", placeholder="e.g. Paris")
    days        = st.number_input("Duration (days)", 1, 30, 5)
    budget      = st.number_input("Budget (USD)", 0, 50_000, 1_500, 100)
    local_c     = st.text_input("Local constraints", placeholder="e.g. vegetarian, no hostels")

    st.divider()
    st.caption("Tavily · Open-Meteo · Frankfurter — all live APIs configured")


# ── Hero ──────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero">
        <h1>AI Travel Planner</h1>
        <p>Multi-agent &middot; Groq LLM &middot; Tavily search &middot; Open-Meteo weather &middot; Frankfurter FX</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Query input ───────────────────────────────────────────────────────────────

examples = [
    "Plan a 5-day trip from Cairo to Paris for 2 people with a $2000 budget.",
    "Plan a 7-day solo budget trip from New York to Tokyo under $3000.",
    "Plan a 3-day romantic trip from London to Barcelona with no budget limit.",
    "Plan a family trip from Dubai to Singapore for 4 days, budget $5000.",
]

col_q, col_eg = st.columns([3, 1])
with col_q:
    st.markdown(
        f'<div class="section-title">{ICO_NOTES} Travel Query</div>',
        unsafe_allow_html=True,
    )
    query = st.text_area("", height=100, label_visibility="collapsed",
                         placeholder="Describe the trip you want to plan…")
with col_eg:
    st.markdown(
        f'<div class="section-title">{ICO_LIST} Quick examples</div>',
        unsafe_allow_html=True,
    )
    for ex in examples:
        if st.button(ex[:52] + "…", key=ex, use_container_width=True):
            st.session_state["_example"] = ex
            st.rerun()

if "_example" in st.session_state:
    query = st.session_state.pop("_example")

st.markdown(
    f'<button disabled style="display:none"></button>',
    unsafe_allow_html=True,
)
run_btn = st.button(
    "Generate Travel Plan",
    type="primary",
    use_container_width=True,
    disabled=not query.strip(),
)

# ── Run ───────────────────────────────────────────────────────────────────────

if run_btn and query.strip():
    metadata = {}
    if origin:      metadata["org"]              = origin
    if destination: metadata["dest"]             = destination
    if days:        metadata["days"]             = days
    if budget:      metadata["budget"]           = budget
    if local_c:     metadata["local_constraint"] = local_c

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-title">{ICO_BRAIN} Agent Pipeline</div>',
        unsafe_allow_html=True,
    )

    agent_cols = st.columns(4)
    placeholders = {}
    agents_info = [
        ("constraint_extractor", "Constraint Extractor", ICO_SEARCH,  agent_cols[0]),
        ("researcher",           "Researcher",           ICO_GLOBE,   agent_cols[1]),
        ("flight_hotel",         "Flights & Hotels",     ICO_HOTEL,   agent_cols[2]),
        ("weather",              "Weather",              ICO_SUN,     agent_cols[3]),
    ]
    for key, label, icon, col in agents_info:
        with col:
            placeholders[key] = st.empty()
            placeholders[key].markdown(
                f'<div class="agent-card running">'
                f'<div class="card-label">{icon}{label}</div>'
                f'<div class="card-status status-running">Running...</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    solver_ph = st.empty()
    solver_ph.markdown(
        f'<div class="agent-card">'
        f'<div class="card-label">{ICO_BRAIN} Solver</div>'
        f'<div class="card-status status-wait">Waiting...</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    auditor_ph = st.empty()
    auditor_ph.markdown(
        f'<div class="agent-card">'
        f'<div class="card-label">{ICO_SEARCH} Auditor</div>'
        f'<div class="card-status status-wait">Waiting...</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    metrics_ph = st.empty()
    plan_ph    = st.empty()

    t_start = time.time()
    with st.spinner("Running multi-agent pipeline..."):
        try:
            result = run_planner(query, metadata)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()
    elapsed = time.time() - t_start

    # ── Mark agents done ──────────────────────────────────────────────────────
    for key, label, icon, _ in agents_info:
        placeholders[key].markdown(
            f'<div class="agent-card done">'
            f'<div class="card-label">{icon}{label}</div>'
            f'<div class="card-status status-done">Done</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    solver_calls = result.get("solver_calls", 1)
    solver_ph.markdown(
        f'<div class="agent-card done">'
        f'<div class="card-label">{ICO_BRAIN} Solver</div>'
        f'<div class="card-status status-done">Done — {solver_calls} call(s)</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    audit_passed = result.get("audit_passed", False)
    auditor_ph.markdown(
        f'<div class="agent-card {"done" if audit_passed else "running"}">'
        f'<div class="card-label">{ICO_CHECK if audit_passed else ICO_WARN} Auditor</div>'
        f'<div class="card-status {"status-done" if audit_passed else "status-running"}">'
        f'{"PASSED" if audit_passed else "REVISED"}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    constraints = result.get("constraints") or {}
    dest_name   = constraints.get("destination_city") or metadata.get("dest") or "—"
    trip_days   = constraints.get("days") or constraints.get("num_days") or days or "—"
    audit_cls   = "pass" if audit_passed else "warn"
    audit_lbl   = "PASS"  if audit_passed else "WARN"

    metrics_ph.markdown(
        f"""
        <div class="metric-row">
          <div class="metric-box">
            <div class="m-label">Destination</div>
            <div class="m-value" style="font-size:1.05rem">{dest_name}</div>
          </div>
          <div class="metric-box">
            <div class="m-label">Duration</div>
            <div class="m-value">{trip_days}d</div>
          </div>
          <div class="metric-box">
            <div class="m-label">Solver calls</div>
            <div class="m-value">{solver_calls}</div>
          </div>
          <div class="metric-box">
            <div class="m-label">Audit</div>
            <div class="m-value {audit_cls}">{audit_lbl}</div>
          </div>
          <div class="metric-box">
            <div class="m-label">Time</div>
            <div class="m-value" style="font-size:1.1rem">{elapsed:.1f}s</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Plan ──────────────────────────────────────────────────────────────────
    plan_text = result.get("draft_plan", "")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-title">{ICO_NOTES} Generated Travel Plan</div>',
        unsafe_allow_html=True,
    )

    if plan_text:
        plan_ph.markdown(
            f'<div class="plan-box">{plan_text}</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            f"Download plan (.txt)",
            data=plan_text,
            file_name=f"travel_plan_{dest_name.lower().replace(' ', '_')}.txt",
            mime="text/plain",
        )
    else:
        plan_ph.warning("No plan generated.")

    feedback = result.get("audit_feedback", "")
    if feedback and not audit_passed:
        with st.expander("Audit feedback"):
            st.markdown(feedback)

    research_notes = result.get("research_notes", "")
    if research_notes:
        with st.expander("Research notes — flights, hotels, weather, RAG"):
            st.markdown(f"```\n{research_notes[:4000]}\n```")

    if constraints:
        with st.expander("Extracted constraints"):
            st.json(constraints)

    msg_history = result.get("message_history", "")
    if msg_history:
        with st.expander("Agent message log"):
            st.code(msg_history, language=None)

    alert_cls = "custom-success" if audit_passed else "custom-warn"
    alert_txt = (
        f"Plan generated in {elapsed:.1f}s — audit PASSED."
        if audit_passed
        else f"Plan generated in {elapsed:.1f}s — audit flagged issues (plan still shown)."
    )
    st.markdown(f'<div class="{alert_cls}">{alert_txt}</div>', unsafe_allow_html=True)
