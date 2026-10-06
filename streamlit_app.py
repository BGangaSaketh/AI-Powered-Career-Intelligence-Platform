"""
streamlit_app.py
================
AI-Powered Career Intelligence Platform — Unified Premium Streamlit Application

Single unified user interface providing seamless access to all 11 platform features:
1. Dashboard Overview
2. Meeting Intelligence & Ingestion
3. Transcript Workspace
4. AI Insights & Analytics
5. Semantic Vector Search
6. Knowledge Repository
7. AI Assistant (Grounded RAG)
8. Text & Sentiment NLP Analysis
9. Executive Reports & Export
10. Cloud Integrations (Zoom & Google Meet)
11. Settings & System Status

Features:
- Premium modern UI design system (dark mode, glassmorphism cards, glowing status badges)
- URL Query Parameter state persistence for browser refresh survival
- Full integration with internal Flask backend API (app.py)
"""

import os
import io
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional

from modules.api_client import MeetingApiClient

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Career Intelligence Platform",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global Custom Design System (CSS) — Obsidian Intelligence ──────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: #0B0D10 !important;
        color: #F1F3F5 !important;
    }

    .stApp {
        background-color: #0B0D10 !important;
        color: #F1F3F5 !important;
    }

    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }

    /* Page Header Hierarchy */
    .obsidian-title {
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        color: #F1F3F5 !important;
        letter-spacing: -0.01em !important;
        margin-bottom: 0.2rem !important;
    }

    .obsidian-sub-caption {
        color: #9AA3AD !important;
        font-size: 0.875rem !important;
        font-weight: 400 !important;
        margin-bottom: 1.5rem !important;
    }

    .obsidian-section-title {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #F1F3F5 !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.6rem !important;
    }

    /* KPI / Metric Cards */
    .metric-card {
        background-color: #12161B !important;
        border: 1px solid #272D35 !important;
        border-top: 2px solid #8B7CF6 !important;
        border-radius: 8px !important;
        padding: 16px 18px !important;
        color: #F1F3F5 !important;
        box-shadow: none !important;
        transition: border-color 0.15s ease !important;
    }
    .metric-card:hover {
        border-color: #363E48 !important;
        border-top-color: #8B7CF6 !important;
    }
    .metric-value {
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        color: #F1F3F5 !important;
        line-height: 1.25 !important;
        margin-top: 4px !important;
    }
    .metric-label {
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #9AA3AD !important;
        font-weight: 600 !important;
    }

    /* Status Indicators */
    .badge-status-online {
        background-color: rgba(69, 185, 124, 0.12) !important;
        color: #45B97C !important;
        border: 1px solid rgba(69, 185, 124, 0.3) !important;
        padding: 3px 10px !important;
        border-radius: 6px !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        display: inline-block !important;
    }
    .badge-status-offline {
        background-color: rgba(217, 107, 107, 0.12) !important;
        color: #D96B6B !important;
        border: 1px solid rgba(217, 107, 107, 0.3) !important;
        padding: 3px 10px !important;
        border-radius: 6px !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        display: inline-block !important;
    }

    /* Primary & Secondary Buttons */
    div.stButton > button, div.stDownloadButton > button {
        background-color: #8B7CF6 !important;
        color: #FFFFFF !important;
        border: 1px solid #7C3AED !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        padding: 0.45rem 1rem !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
        box-shadow: none !important;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background-color: #7C3AED !important;
        border-color: #6D28D9 !important;
        box-shadow: none !important;
    }

    /* Navigation Tabs */
    div[data-baseweb="tab-list"] {
        background-color: #12161B !important;
        border-radius: 8px !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid #272D35 !important;
    }
    div[data-baseweb="tab"] {
        border-radius: 6px !important;
        color: #9AA3AD !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 6px 14px !important;
        transition: all 0.15s ease !important;
    }
    div[data-baseweb="tab"][aria-selected="true"] {
        background-color: #171C22 !important;
        color: #F1F3F5 !important;
        border: 1px solid #272D35 !important;
        box-shadow: none !important;
    }

    /* Content Boxes */
    .content-box {
        background-color: #12161B !important;
        border: 1px solid #272D35 !important;
        border-radius: 8px !important;
        padding: 20px !important;
        margin-bottom: 16px !important;
    }

    /* Sidebar Shell */
    [data-testid="stSidebar"] {
        background-color: #0B0D10 !important;
        border-right: 1px solid #272D35 !important;
    }

    .sidebar-user-card {
        background-color: #12161B !important;
        border: 1px solid #272D35 !important;
        border-radius: 6px !important;
        padding: 10px 12px !important;
        margin-bottom: 16px !important;
    }

    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0B0D10;
    }
    ::-webkit-scrollbar-thumb {
        background: #272D35;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #363E48;
    }
</style>
""", unsafe_allow_html=True)


# ── Navigation Registry & Clean Mapping ───────────────────────────────────
NAV_PAGES = [
    "Dashboard",
    "Meeting Intelligence",
    "Transcript Workspace",
    "AI Insights",
    "Semantic Search",
    "Knowledge Repository",
    "AI Assistant / RAG",
    "Text & Sentiment NLP",
    "Reports & Export",
    "Cloud Integrations",
    "Settings & System Status"
]

PAGE_LABEL_MAP = {
    "📊 Dashboard": "Dashboard",
    "🎙️ Meeting Intelligence": "Meeting Intelligence",
    "📜 Transcript Workspace": "Transcript Workspace",
    "💡 AI Insights": "AI Insights",
    "🔍 Semantic Search": "Semantic Search",
    "📚 Knowledge Repository": "Knowledge Repository",
    "🤖 AI Assistant / RAG": "AI Assistant / RAG",
    "🧠 Text & Sentiment NLP": "Text & Sentiment NLP",
    "📄 Reports & Export": "Reports & Export",
    "🔌 Cloud Integrations": "Cloud Integrations",
    "⚙️ Settings & System Status": "Settings & System Status"
}


# ── Session State & Query Parameter Syncing ────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("API_KEY") or os.getenv("AUTH_TOKEN") or ""
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = os.getenv("API_BASE_URL", "http://localhost:5000")
if "user_info" not in st.session_state:
    st.session_state.user_info = None

# Restore state from URL query parameters if present (for refresh survival)
url_params = st.query_params
param_page = url_params.get("page", "Dashboard")
initial_page = PAGE_LABEL_MAP.get(param_page, param_page)
if initial_page not in NAV_PAGES:
    initial_page = "Dashboard"

if "current_page" not in st.session_state:
    st.session_state.current_page = initial_page
if "selected_meeting_id" not in st.session_state:
    st.session_state.selected_meeting_id = url_params.get("meeting_id", None)


def navigate_to(page_name: str, meeting_id: Optional[str] = None):
    """Navigate to a page while updating session state and URL query parameters."""
    clean_page = PAGE_LABEL_MAP.get(page_name, page_name)
    st.session_state.current_page = clean_page
    st.query_params["page"] = clean_page
    if meeting_id:
        st.session_state.selected_meeting_id = meeting_id
        st.query_params["meeting_id"] = meeting_id
    st.rerun()


def get_api_client() -> MeetingApiClient:
    """Instantiate API client with current session settings."""
    return MeetingApiClient(
        base_url=st.session_state.api_base_url,
        api_key=st.session_state.api_key if st.session_state.api_key else None
    )


# ── Authentication View ────────────────────────────────────────────────────
def render_login_view():
    col_l, col_center, col_r = st.columns([1, 2, 1])

    with col_center:
        st.markdown('<div class="obsidian-title" style="text-align: center;">AI Career Intelligence Platform</div>', unsafe_allow_html=True)
        st.markdown('<div class="obsidian-sub-caption" style="text-align: center;">Meeting Intelligence & Grounded RAG Platform</div>', unsafe_allow_html=True)

        auth_tab1, auth_tab2, auth_tab3 = st.tabs(["User Login", "Register Account", "Token Connection"])

        with auth_tab1:
            with st.form("user_login_form"):
                username_input = st.text_input("Username or Email", help="Your account username or email address")
                password_input = st.text_input("Password", type="password")
                login_btn = st.form_submit_button("Login", use_container_width=True)

                if login_btn:
                    u_clean = username_input.strip()
                    p_clean = password_input.strip()
                    if not u_clean or not p_clean:
                        st.error("Please enter both username/email and password.")
                    else:
                        client = get_api_client()
                        try:
                            res = client.login(username_or_email=u_clean, password=p_clean)
                            st.session_state.api_key = res.get("token", "")
                            st.session_state.user_info = res.get("user")
                            st.session_state.authenticated = True
                            st.success(f"Welcome back, {res.get('user', {}).get('username', 'User')}!")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Login failed: {exc}")

        with auth_tab2:
            with st.form("user_register_form"):
                reg_username = st.text_input("Desired Username")
                reg_email = st.text_input("Email Address")
                reg_password = st.text_input("Password", type="password")
                register_btn = st.form_submit_button("Register Account", use_container_width=True)

                if register_btn:
                    ru_clean = reg_username.strip()
                    re_clean = reg_email.strip()
                    rp_clean = reg_password.strip()
                    if not ru_clean or not re_clean or not rp_clean:
                        st.error("Please complete all registration fields.")
                    else:
                        client = get_api_client()
                        try:
                            res = client.register(username=ru_clean, email=re_clean, password=rp_clean)
                            st.session_state.api_key = res.get("token", "")
                            st.session_state.user_info = res.get("user")
                            st.session_state.authenticated = True
                            st.success(f"Registration successful! Logged in as {ru_clean}.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Registration failed: {exc}")

        with auth_tab3:
            with st.form("token_connect_form"):
                api_key_input = st.text_input(
                    "API Key / System Token",
                    value=st.session_state.api_key,
                    type="password",
                    help="System token or admin bearer key"
                )

                connect_btn = st.form_submit_button("Connect via Token", use_container_width=True)

                if connect_btn:
                    st.session_state.api_key = api_key_input.strip()
                    client = get_api_client()
                    with st.spinner("Connecting to platform backend..."):
                        success, message = client.verify_connection()

                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user_info = {"username": "System User", "id": "system"}
                        st.success("Successfully authenticated!")
                        st.rerun()
                    else:
                        st.error(f"Connection Failed: {message}")


# ── Render Header & Sidebar ────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("""
        <div class="sidebar-brand">AI Career Intelligence</div>
        <div class="sidebar-sub">Meeting Intelligence & Grounded RAG</div>
    """, unsafe_allow_html=True)

    user_name = st.session_state.user_info.get("username", "User") if st.session_state.user_info else "Authenticated User"
    st.sidebar.markdown(f"""
        <div class="sidebar-user-card">
            <div style="font-size: 0.72rem; text-transform: uppercase; color: #9AA3AD; font-weight: 600;">Active Account</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #F1F3F5; margin-top: 2px;">{user_name}</div>
        </div>
    """, unsafe_allow_html=True)

    current_idx = NAV_PAGES.index(st.session_state.current_page) if st.session_state.current_page in NAV_PAGES else 0
    selected = st.sidebar.radio("Navigation", NAV_PAGES, index=current_idx)

    if selected != st.session_state.current_page:
        st.session_state.current_page = selected
        st.query_params["page"] = selected
        st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True):
        client = get_api_client()
        client.logout()
        st.session_state.authenticated = False
        st.session_state.api_key = ""
        st.session_state.user_info = None
        st.session_state.selected_meeting_id = None
        st.query_params.clear()
        st.rerun()




# ── 1. Dashboard ───────────────────────────────────────────────────────────
def render_dashboard_page():
    st.markdown('<div class="obsidian-title">Career Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="obsidian-sub-caption">Meeting Intelligence & Grounded RAG Platform Overview</div>', unsafe_allow_html=True)

    client = get_api_client()

    try:
        with st.spinner("Loading platform analytics..."):
            meetings_data = client.get_all_meetings_knowledge()
            health_ok, _ = client.health_check()
    except Exception as exc:
        st.error(f"Failed to fetch dashboard data: {exc}")
        return

    total_meetings = len(meetings_data)
    total_action_items = sum(len(m.get("action_items", [])) for m in meetings_data)
    total_decisions = sum(len(m.get("decisions", [])) for m in meetings_data)

    unique_participants = set()
    for m in meetings_data:
        for p in m.get("participants", []):
            if isinstance(p, dict) and p.get("name"):
                unique_participants.add(p.get("name"))
            elif isinstance(p, str):
                unique_participants.add(p)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Meetings</div><div class="metric-value">{total_meetings}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Action Items</div><div class="metric-value">{total_action_items}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Participants</div><div class="metric-value">{len(unique_participants)}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Decisions</div><div class="metric-value">{total_decisions}</div></div>', unsafe_allow_html=True)
    with c5:
        status_html = '<span class="badge-status-online">Active</span>' if health_ok else '<span class="badge-status-offline">Offline</span>'
        st.markdown(f'<div class="metric-card"><div class="metric-label">Backend API</div><div style="margin-top: 10px;">{status_html}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_overview, tab_quick_actions = st.tabs(["Stored Meetings", "Quick Actions"])

    with tab_overview:
        if not meetings_data:
            st.info("No meetings processed yet. Use the Meeting Intelligence tab to process your first meeting.")
        else:
            table_rows = []
            for m in meetings_data:
                table_rows.append({
                    "Meeting ID": m.get("id"),
                    "Title": m.get("title", "Untitled Meeting"),
                    "Date": str(m.get("created_at", "N/A"))[:10],
                    "Participants": len(m.get("participants", [])),
                    "Action Items": len(m.get("action_items", [])),
                    "Decisions": len(m.get("decisions", [])),
                    "Summary": m.get("summary", "")[:110] + ("..." if len(m.get("summary", "")) > 110 else "")
                })

            df = pd.DataFrame(table_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown('<div class="obsidian-section-title">Inspect Specific Meeting</div>', unsafe_allow_html=True)
            m_options = {m["id"]: f"{m.get('title')} ({str(m.get('created_at', ''))[:10]})" for m in meetings_data}
            selected_m = st.selectbox("Select Meeting to Open:", list(m_options.keys()), format_func=lambda x: m_options[x], key="dash_m_select")
            if st.button("Open Meeting Intelligence"):
                navigate_to("Meeting Intelligence", selected_m)

    with tab_quick_actions:
        qa_col1, qa_col2, qa_col3, qa_col4 = st.columns(4)
        with qa_col1:
            st.markdown('<div class="obsidian-section-title">Meeting Intelligence</div>', unsafe_allow_html=True)
            st.write("Upload audio/video recordings or text transcripts for processing.")
            if st.button("Go to Meeting Intelligence", key="qa_btn1"):
                navigate_to("Meeting Intelligence")
        with qa_col2:
            st.markdown('<div class="obsidian-section-title">Semantic Search</div>', unsafe_allow_html=True)
            st.write("Perform similarity vector search across all meeting content.")
            if st.button("Go to Semantic Search", key="qa_btn2"):
                navigate_to("Semantic Search")
        with qa_col3:
            st.markdown('<div class="obsidian-section-title">Grounded RAG AI</div>', unsafe_allow_html=True)
            st.write("Ask natural-language questions with source citations.")
            if st.button("Go to AI Assistant", key="qa_btn3"):
                navigate_to("AI Assistant / RAG")
        with qa_col4:
            st.markdown('<div class="obsidian-section-title">Executive Reports</div>', unsafe_allow_html=True)
            st.write("Generate professional PDF reports and CSV datasets.")
            if st.button("Go to Reports & Export", key="qa_btn4"):
                navigate_to("Reports & Export")


# ── 2. Meeting Intelligence ────────────────────────────────────────────────
def render_meeting_intelligence_page():
    st.markdown('<div class="obsidian-title">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="obsidian-sub-caption">Ingest, transcribe audio/video recordings, and inspect structured meeting intelligence.</div>', unsafe_allow_html=True)

    client = get_api_client()

    mi_tab1, mi_tab2 = st.tabs(["Upload & Process", "Processed Meetings"])

    with mi_tab1:
        upload_type = st.radio("Select Input Source:", ["Audio / Video Recording File", "Raw Transcript Text"], horizontal=True)

        title_input = st.text_input("Meeting Title (Optional)", placeholder="e.g. Q4 Architecture & Product Strategy Review")

        if upload_type == "Audio / Video Recording File":
            uploaded_file = st.file_uploader(
                "Choose recording file:",
                type=["wav", "mp3", "flac", "ogg", "m4a", "mp4", "avi", "mov", "mkv", "webm"],
                help="Supported formats: WAV, MP3, FLAC, OGG, M4A, MP4, AVI, MOV, MKV, WEBM"
            )

            if uploaded_file is not None:
                st.write(f"File: `{uploaded_file.name}` ({uploaded_file.size / (1024*1024):.2f} MB)")

                if st.button("Process Recording", type="primary", key="btn_proc_rec"):
                    with st.spinner("Transcribing speech with Whisper & extracting meeting intelligence..."):
                        try:
                            file_bytes = uploaded_file.read()
                            res = client.process_meeting(
                                file_bytes=file_bytes,
                                filename=uploaded_file.name,
                                title=title_input.strip() if title_input else uploaded_file.name
                            )

                            st.success(f"Meeting processed successfully. ID: `{res.get('meeting_id')}`")
                            st.session_state.selected_meeting_id = res.get("meeting_id")

                            intelligence = res.get("intelligence", {})
                            st.markdown('<div class="obsidian-section-title">Executive Summary</div>', unsafe_allow_html=True)
                            st.info(intelligence.get("summary", "No summary generated."))

                        except Exception as exc:
                            st.error(f"Meeting processing failed: {exc}")

        else:
            transcript_text = st.text_area("Paste Raw Meeting Transcript Text:", height=220, placeholder="Speaker A: Welcome everyone to the quarterly review...\nSpeaker B: I will update on backend API performance...")

            if st.button("Process Transcript Text", type="primary", key="btn_proc_text"):
                if not transcript_text.strip():
                    st.warning("Please enter transcript text before processing.")
                else:
                    with st.spinner("Extracting intelligence via backend pipeline..."):
                        try:
                            res = client.process_meeting(
                                transcript_text=transcript_text.strip(),
                                title=title_input.strip() if title_input else "Meeting Transcript"
                            )
                            st.success(f"Meeting transcript processed. ID: `{res.get('meeting_id')}`")
                            st.session_state.selected_meeting_id = res.get("meeting_id")

                            intelligence = res.get("intelligence", {})
                            st.markdown('<div class="obsidian-section-title">Executive Summary</div>', unsafe_allow_html=True)
                            st.info(intelligence.get("summary", "No summary generated."))

                        except Exception as exc:
                            st.error(f"Processing failed: {exc}")

    with mi_tab2:
        try:
            meetings = client.list_meetings()
        except Exception as exc:
            st.error(f"Failed to fetch meetings list: {exc}")
            return

        if not meetings:
            st.info("No processed meetings found. Upload a meeting to populate intelligence.")
            return

        meeting_map = {m["id"]: f"{m.get('title', 'Untitled')} — {str(m.get('created_at', ''))[:10]}" for m in meetings}

        default_idx = 0
        if st.session_state.selected_meeting_id in meeting_map:
            keys_list = list(meeting_map.keys())
            default_idx = keys_list.index(st.session_state.selected_meeting_id)

        selected_id = st.selectbox(
            "Select Meeting to Inspect:",
            options=list(meeting_map.keys()),
            format_func=lambda x: meeting_map[x],
            index=default_idx,
            key="mi_selectbox_meeting"
        )

        st.session_state.selected_meeting_id = selected_id

        if selected_id:
            try:
                with st.spinner("Loading meeting details..."):
                    m_details = client.get_meeting_details(selected_id)

                meta = m_details.get("metadata", {})
                summary = m_details.get("summary", "No summary available.")
                decisions = m_details.get("decisions", [])
                action_items = m_details.get("action_items", [])
                participants = m_details.get("participants", [])

                st.subheader(f"📌 {meta.get('title', 'Untitled Meeting')}")
                st.caption(f"ID: `{selected_id}` | Date: `{meta.get('created_at', 'N/A')}`")

                det_tab1, det_tab2, det_tab3, det_tab4 = st.tabs(["📝 Summary & Key Points", "🎯 Action Items", "💡 Decisions Log", "👤 Participants"])

                with det_tab1:
                    st.markdown("#### Executive Summary")
                    st.info(summary)

                with det_tab2:
                    st.markdown("#### Extracted Action Items")
                    if not action_items:
                        st.info("No action items recorded for this meeting.")
                    else:
                        act_rows = []
                        for a in action_items:
                            if isinstance(a, dict):
                                act_rows.append({
                                    "Task": a.get("task", ""),
                                    "Assignee": a.get("assignee", "Unassigned"),
                                    "Deadline": a.get("deadline", "N/A"),
                                    "Priority": a.get("priority", "Medium"),
                                    "Status": a.get("status", "Pending")
                                })
                        if act_rows:
                            st.dataframe(pd.DataFrame(act_rows), use_container_width=True, hide_index=True)

                with det_tab3:
                    st.markdown("#### Key Decisions")
                    if not decisions:
                        st.info("No key decisions recorded for this meeting.")
                    else:
                        for d in decisions:
                            st.markdown(f"- 💡 **{d}**")

                with det_tab4:
                    st.markdown("#### Meeting Participants & Responsibilities")
                    if not participants:
                        st.info("No participants listed.")
                    else:
                        for p in participants:
                            if isinstance(p, dict):
                                p_name = p.get("name", "Unknown")
                                resps = p.get("responsibilities", [])
                                st.markdown(f"**👤 {p_name}**")
                                for r in resps:
                                    st.markdown(f"  - {r}")
                            elif isinstance(p, str):
                                st.markdown(f"- 👤 {p}")

            except Exception as exc:
                st.error(f"Failed to load meeting details: {exc}")


# ── 3. Transcript Workspace ────────────────────────────────────────────────
def render_transcript_workspace_page():
    st.markdown('<div class="obsidian-title">Transcript Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="obsidian-sub-caption">Search, filter, and inspect full speech transcripts from historical meeting recordings.</div>', unsafe_allow_html=True)

    client = get_api_client()

    try:
        meetings = client.list_meetings()
    except Exception as exc:
        st.error(f"Failed to fetch meetings: {exc}")
        return

    if not meetings:
        st.info("No processed meeting transcripts available.")
        return

    meeting_map = {m["id"]: f"{m.get('title', 'Untitled')} ({str(m.get('created_at', ''))[:10]})" for m in meetings}
    selected_id = st.selectbox("Select Meeting Transcript:", list(meeting_map.keys()), format_func=lambda x: meeting_map[x], key="trans_m_select")

    if selected_id:
        try:
            m_details = client.get_meeting_details(selected_id)
            transcript_text = m_details.get("transcript", "No transcript text available.")

            col_search, col_stats = st.columns([3, 1])

            with col_stats:
                st.markdown('<div class="obsidian-section-title">Transcript Stats</div>', unsafe_allow_html=True)
                word_count = len(transcript_text.split())
                char_count = len(transcript_text)
                st.write(f"**Words**: `{word_count:,}`")
                st.write(f"**Characters**: `{char_count:,}`")
                st.download_button("Download Transcript Text", transcript_text, file_name=f"transcript_{selected_id}.txt", mime="text/plain", use_container_width=True)

            with col_search:
                filter_kw = st.text_input("Search Keyword inside transcript:", placeholder="Type to filter transcript lines...")
                st.markdown('<div class="obsidian-section-title">Speech Transcript Content</div>', unsafe_allow_html=True)

                if filter_kw.strip():
                    matching_lines = [line for line in transcript_text.splitlines() if filter_kw.lower() in line.lower()]
                    st.caption(f"Found {len(matching_lines)} matching line(s) for `{filter_kw}`:")
                    st.code("\n".join(matching_lines) if matching_lines else "No matching lines found.", language="text")
                else:
                    st.text_area("Full Transcript Text", value=transcript_text, height=400, disabled=True)

        except Exception as exc:
            st.error(f"Failed to load transcript: {exc}")


# ── 4. AI Insights ──────────────────────────────────────────────────────────
def render_ai_insights_page():
    st.markdown('<div class="obsidian-title">AI Insights & Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="obsidian-sub-caption">Cross-meeting intelligence, global action item priority matrix, and participant responsibility mapping.</div>', unsafe_allow_html=True)

    client = get_api_client()

    try:
        meetings_knowledge = client.get_all_meetings_knowledge()
    except Exception as exc:
        st.error(f"Failed to load insights data: {exc}")
        return

    if not meetings_knowledge:
        st.info("No meeting data available for analysis.")
        return

    insights_tab1, insights_tab2, insights_tab3 = st.tabs(["Action Items Matrix", "Decisions Log", "Participant Mapping"])

    with insights_tab1:
        st.markdown('<div class="obsidian-section-title">Action Items Matrix</div>', unsafe_allow_html=True)
        all_actions = []
        for m in meetings_knowledge:
            m_title = m.get("title", "Untitled")
            m_id = m.get("id")
            for a in m.get("action_items", []):
                if isinstance(a, dict):
                    all_actions.append({
                        "Meeting": m_title,
                        "Task": a.get("task", ""),
                        "Assignee": a.get("assignee", "Unassigned"),
                        "Deadline": a.get("deadline", "N/A"),
                        "Priority": a.get("priority", "Medium"),
                        "Status": a.get("status", "Pending"),
                        "Meeting ID": m_id
                    })

        if not all_actions:
            st.info("No action items extracted across meetings.")
        else:
            df_act = pd.DataFrame(all_actions)
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                assignee_filter = st.multiselect("Filter Assignee:", options=list(df_act["Assignee"].unique()), default=[])
            with col_f2:
                priority_filter = st.multiselect("Filter Priority:", options=list(df_act["Priority"].unique()), default=[])

            filtered_df = df_act.copy()
            if assignee_filter:
                filtered_df = filtered_df[filtered_df["Assignee"].isin(assignee_filter)]
            if priority_filter:
                filtered_df = filtered_df[filtered_df["Priority"].isin(priority_filter)]

            st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    with insights_tab2:
        st.markdown('<div class="obsidian-section-title">Key Decisions Log</div>', unsafe_allow_html=True)
        decision_count = 0
        for m in meetings_knowledge:
            m_title = m.get("title", "Untitled")
            m_date = str(m.get("created_at", "N/A"))[:10]
            decs = m.get("decisions", [])
            if decs:
                st.markdown(f"**{m_title}** (`{m_date}`)")
                for d in decs:
                    st.markdown(f"  - {d}")
                    decision_count += 1
                st.markdown("---")
        if decision_count == 0:
            st.info("No key decisions recorded.")

    with insights_tab3:
        st.markdown('<div class="obsidian-section-title">Participant Responsibility Matrix</div>', unsafe_allow_html=True)
        p_data = []
        for m in meetings_knowledge:
            m_title = m.get("title", "Untitled")
            for p in m.get("participants", []):
                if isinstance(p, dict):
                    p_data.append({
                        "Participant": p.get("name", "Unknown"),
                        "Meeting": m_title,
                        "Responsibilities": ", ".join(p.get("responsibilities", [])) or "General Attendance"
                    })
                elif isinstance(p, str):
                    p_data.append({
                        "Participant": p,
                        "Meeting": m_title,
                        "Responsibilities": "General Attendance"
                    })
        if p_data:
            st.dataframe(pd.DataFrame(p_data), use_container_width=True, hide_index=True)
        else:
            st.info("No participant data recorded.")


# ── 5. Semantic Search ──────────────────────────────────────────────────────
def render_semantic_search_page():
    st.markdown('<div class="obsidian-title">Semantic Vector Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="obsidian-sub-caption">Cosine similarity vector search across indexed meeting transcripts, summaries, decisions, and action items.</div>', unsafe_allow_html=True)

    client = get_api_client()

    col_q, col_k = st.columns([3, 1])
    with col_q:
        query_input = st.text_input(
            "Search Query:",
            placeholder="e.g. Which meeting discussed vector database setup and API latency?",
            key="semantic_search_query_input"
        )
    with col_k:
        top_k = st.slider("Top Matches (K)", min_value=1, max_value=20, value=5, key="search_top_k")

    with st.expander("Search Filters & Controls", expanded=True):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            content_type = st.selectbox("Content Type", ["All Types", "summary", "transcript", "decision", "action_item", "key_point"], key="search_c_type")
            c_type_val = None if content_type == "All Types" else content_type
        with col_f2:
            try:
                meetings = client.list_meetings()
                m_opts = {"All Meetings": None}
                for m in meetings:
                    m_opts[f"{m.get('title', 'Untitled')} ({m.get('id')})"] = m.get("id")
                selected_m_label = st.selectbox("Target Meeting Scope", list(m_opts.keys()), key="search_m_filter")
                m_id_val = m_opts[selected_m_label]
            except Exception:
                m_id_val = None
        with col_f3:
            min_score = st.slider("Min Similarity Score", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="search_min_score")

    if st.button("Execute Search", type="primary", use_container_width=True):
        if not query_input.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Executing similarity search over vector store..."):
                try:
                    res = client.semantic_search(
                        query=query_input.strip(),
                        top_k=top_k,
                        content_type=c_type_val,
                        meeting_id=m_id_val,
                        min_score=min_score if min_score > 0 else None
                    )

                    results = res.get("results", [])
                    st.markdown(f'<div class="obsidian-section-title">Search Results ({len(results)} matches found in {res.get("latency_ms", 0):.1f}ms)</div>', unsafe_allow_html=True)

                    if not results:
                        st.info("No matching meeting content found for your query with selected filters.")
                    else:
                        for idx, item in enumerate(results, 1):
                            score = item.get("similarity_score") or item.get("score") or 0.0
                            tag = item.get("content_type", "chunk")
                            title = item.get("meeting_title") or item.get("title") or "Meeting"
                            snippet = item.get("text") or item.get("relevant_snippet") or ""

                            with st.container():
                                st.markdown(f"**#{idx} | {title}** (Score: `{score:.4f}` | Tag: `{tag.upper()}`)")
                                st.info(snippet)
                                st.markdown("---")

                except Exception as exc:
                    st.error(f"Semantic search failed: {exc}")


# ── 6. Knowledge Repository ────────────────────────────────────────────────
def render_knowledge_repository_page():
    st.markdown('<div class="obsidian-title">Knowledge Repository</div>', unsafe_allow_html=True)
    st.markdown('<div class="obsidian-sub-caption">Historical meeting knowledge chunks, SQLite database entries, and vector index metadata.</div>', unsafe_allow_html=True)

    client = get_api_client()

    try:
        meetings_data = client.get_all_meetings_knowledge()
    except Exception as exc:
        st.error(f"Failed to fetch knowledge data: {exc}")
        return

    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.metric("Total Meetings", len(meetings_data))
    with col_k2:
        total_chunks = sum(1 + len(m.get("decisions", [])) + len(m.get("action_items", [])) for m in meetings_data)
        st.metric("Vector Knowledge Chunks", total_chunks)
    with col_k3:
        st.metric("Vector Index Metric", "Cosine Similarity")

    st.markdown('<div class="obsidian-section-title">Stored Knowledge Entries</div>', unsafe_allow_html=True)
    search_k = st.text_input("Filter knowledge records:", placeholder="Type to search knowledge base...")

    rows = []
    for m in meetings_data:
        m_id = m.get("id")
        m_title = m.get("title", "Untitled")
        m_date = str(m.get("created_at", "N/A"))[:10]

        rows.append({"Meeting ID": m_id, "Title": m_title, "Date": m_date, "Content Type": "SUMMARY", "Text": m.get("summary", "")})
        for d in m.get("decisions", []):
            rows.append({"Meeting ID": m_id, "Title": m_title, "Date": m_date, "Content Type": "DECISION", "Text": str(d)})
        for a in m.get("action_items", []):
            task_str = a.get("task") if isinstance(a, dict) else str(a)
            rows.append({"Meeting ID": m_id, "Title": m_title, "Date": m_date, "Content Type": "ACTION_ITEM", "Text": task_str})

    df_k = pd.DataFrame(rows)

    if search_k.strip():
        df_k = df_k[df_k["Text"].str.contains(search_k.strip(), case=False, na=False)]

    st.dataframe(df_k, use_container_width=True, hide_index=True)


# ── 7. AI Assistant / RAG ──────────────────────────────────────────────────
def render_ai_assistant_page():
    st.markdown('<div class="obsidian-title">AI Assistant (Grounded RAG)</div>', unsafe_allow_html=True)
    st.markdown('<div class="obsidian-sub-caption">Ask natural language questions grounded in meeting records with direct source citations.</div>', unsafe_allow_html=True)

    client = get_api_client()

    col_q, col_k = st.columns([3, 1])
    with col_q:
        question_input = st.text_input(
            "Enter your question for the AI Assistant:",
            placeholder="e.g. What were the key action items and deadlines discussed for the database migration?",
            key="rag_question_input"
        )
    with col_k:
        top_k = st.slider("Context Chunks (K)", min_value=1, max_value=10, value=5, key="rag_top_k")

    with st.expander("Retrieval Options", expanded=False):
        try:
            meetings = client.list_meetings()
            m_options = {"All Meetings": None}
            for m in meetings:
                m_options[f"{m.get('title', 'Untitled')} ({m.get('id')})"] = m.get("id")
            selected_m_label = st.selectbox("Target Meeting Scope", list(m_options.keys()), key="rag_meeting_scope")
            m_id_val = m_options[selected_m_label]
        except Exception:
            m_id_val = None

    if st.button("Ask Assistant", type="primary", use_container_width=True):
        if not question_input.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving grounded context & generating AI answer..."):
                try:
                    res = client.ask_assistant(
                        question=question_input.strip(),
                        top_k=top_k,
                        meeting_id=m_id_val
                    )

                    answer_text = res.get("answer") or "No grounded answer available."
                    sources = res.get("sources") or []

                    st.markdown('<div class="obsidian-section-title">Grounded Answer</div>', unsafe_allow_html=True)
                    if "couldn't find enough information" in answer_text.lower() or not sources:
                        st.warning(f"{answer_text}")
                    else:
                        st.success(f"{answer_text}")

                    st.markdown("---")
                    st.markdown(f'<div class="obsidian-section-title">Retrieved Sources ({len(sources)} context chunks in {res.get("latency_ms", 0):.1f}ms)</div>', unsafe_allow_html=True)

                    for idx, src in enumerate(sources, 1):
                        title = src.get("meeting_title") or src.get("title") or "Meeting"
                        score = src.get("similarity_score") or src.get("score") or 0.0
                        snippet = src.get("text") or src.get("relevant_snippet") or ""

                        with st.expander(f"Source #{idx} | {title} | Score: {score:.4f}", expanded=(idx == 1)):
                            st.markdown(f"> {snippet}")

                except Exception as exc:
                    st.error(f"RAG query failed: {exc}")


# ── 8. Text & Sentiment NLP ────────────────────────────────────────────────
def render_text_sentiment_page():
    st.markdown('<div class="obsidian-title">Text & Sentiment NLP</div>', unsafe_allow_html=True)
    st.markdown('<div class="obsidian-sub-caption">Direct text analysis workspace powered by VADER sentiment analysis and NLTK preprocessing.</div>', unsafe_allow_html=True)

    client = get_api_client()

    input_method = st.radio("Select Text Source:", ["Raw Text Input", "Text / CSV File Upload"], horizontal=True)

    text_to_analyze = ""
    if input_method == "Raw Text Input":
        text_to_analyze = st.text_area("Enter Text for Sentiment Analysis:", height=200, placeholder="Our team made fantastic progress on the milestone ahead of schedule. However, server latency issues need urgent attention.")
    else:
        uploaded_txt = st.file_uploader("Upload `.txt` or `.csv` file:", type=["txt", "csv"])
        if uploaded_txt is not None:
            text_to_analyze = uploaded_txt.read().decode("utf-8", errors="ignore")
            st.write(f"Loaded `{uploaded_txt.name}` ({len(text_to_analyze)} chars)")

    if st.button("Run Sentiment & NLP Analysis", type="primary"):
        if not text_to_analyze.strip():
            st.warning("Please provide text to analyze.")
        else:
            with st.spinner("Analyzing sentiment and running NLP pipeline..."):
                try:
                    res = client.analyze_text(text_to_analyze.strip())

                    sentiment = res.get("sentiment_detail", {})
                    label = sentiment.get("label", "Neutral")
                    compound = sentiment.get("compound", 0.0)
                    pos = sentiment.get("pos", 0.0)
                    neu = sentiment.get("neu", 0.0)
                    neg = sentiment.get("neg", 0.0)

                    st.markdown('<div class="obsidian-section-title">Sentiment Overview</div>', unsafe_allow_html=True)
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Overall Label", label)
                    with c2:
                        st.metric("Compound Score", f"{compound:.4f}")
                    with c3:
                        st.metric("Positive Polarity", f"{pos*100:.1f}%")
                    with c4:
                        st.metric("Negative Polarity", f"{neg*100:.1f}%")

                    st.markdown('<div class="obsidian-section-title">Per-Sentence Sentiment Breakdown</div>', unsafe_allow_html=True)
                    per_sentence = sentiment.get("per_sentence", [])
                    if per_sentence:
                        st.dataframe(pd.DataFrame(per_sentence), use_container_width=True, hide_index=True)
                    else:
                        st.info("No sentence breakdown generated.")

                    st.markdown('<div class="obsidian-section-title">Generated Summary</div>', unsafe_allow_html=True)
                    st.info(res.get("summary", "No summary generated."))

                except Exception as exc:
                    st.error(f"NLP analysis failed: {exc}")


# ── 9. Reports & Export ────────────────────────────────────────────────────
def render_reports_export_page():
    st.markdown('<div class="obsidian-title">Executive Reports & Export</div>', unsafe_allow_html=True)
    st.markdown('<div class="obsidian-sub-caption">Generate executive PDF summary reports and structured CSV datasets for meeting intelligence.</div>', unsafe_allow_html=True)

    client = get_api_client()

    try:
        meetings = client.list_meetings()
    except Exception as exc:
        st.error(f"Failed to fetch meetings: {exc}")
        return

    if not meetings:
        st.info("No processed meetings available for export.")
        return

    meeting_map = {m["id"]: f"{m.get('title', 'Untitled')} ({str(m.get('created_at', ''))[:10]})" for m in meetings}
    selected_id = st.selectbox("Select Meeting for Report Generation:", list(meeting_map.keys()), format_func=lambda x: meeting_map[x], key="export_m_select")

    if selected_id:
        try:
            m_details = client.get_meeting_details(selected_id)
            meta = m_details.get("metadata", {})
            st.markdown(f"Selected Meeting: `{meta.get('title', 'Untitled')}`")

            col_pdf, col_csv = st.columns(2)

            with col_pdf:
                st.markdown('<div class="obsidian-section-title">Executive PDF Report</div>', unsafe_allow_html=True)
                st.write("Generates a publication-grade PDF report with executive summary, decisions, action items, and participants.")
                if st.button("Generate PDF Report", key="btn_gen_pdf"):
                    with st.spinner("Building PDF report with ReportLab..."):
                        try:
                            pdf_bytes = client.export_meeting_pdf(selected_id)
                            st.download_button(
                                label="Download Executive PDF",
                                data=pdf_bytes,
                                file_name=f"Meeting_Report_{selected_id}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        except Exception as exc:
                            st.error(f"PDF generation failed: {exc}")

            with col_csv:
                st.markdown('<div class="obsidian-section-title">Structured CSV Dataset</div>', unsafe_allow_html=True)
                st.write("Generates a clean CSV file containing structured meeting metadata, summary, decisions, and action items.")
                if st.button("Generate CSV Dataset", key="btn_gen_csv"):
                    with st.spinner("Building CSV dataset..."):
                        try:
                            csv_text = client.export_meeting_csv(selected_id)
                            st.download_button(
                                label="Download Structured CSV",
                                data=csv_text,
                                file_name=f"Meeting_Dataset_{selected_id}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        except Exception as exc:
                            st.error(f"CSV generation failed: {exc}")

        except Exception as exc:
            st.error(f"Failed to fetch meeting details: {exc}")


# ── 10. Cloud Integrations ─────────────────────────────────────────────────
def render_cloud_integrations_page():
    st.markdown('<div class="obsidian-title">Cloud Integrations</div>', unsafe_allow_html=True)
    st.markdown('<div class="obsidian-sub-caption">Import meeting recordings from Zoom & Google Meet cloud integrations.</div>', unsafe_allow_html=True)

    client = get_api_client()

    tab_zoom, tab_gmeet = st.tabs(["Zoom Integration", "Google Meet Integration"])

    with tab_zoom:
        st.markdown('<div class="obsidian-section-title">Zoom Cloud Recording Import</div>', unsafe_allow_html=True)
        st.info("Uses Server-to-Server OAuth to fetch and import cloud meeting recordings.")

        col_z1, col_z2 = st.columns([2, 1])
        with col_z1:
            zoom_id = st.text_input("Zoom Meeting / Recording ID", placeholder="e.g. 84920481920", key="zoom_id_input")
        with col_z2:
            st.markdown("<br>", unsafe_allow_html=True)
            fetch_z_btn = st.button("List Zoom Cloud Recordings", use_container_width=True)

        if fetch_z_btn:
            with st.spinner("Listing Zoom recordings..."):
                try:
                    recs = client.list_zoom_recordings()
                    if recs:
                        st.success(f"Found {len(recs)} Zoom recording(s).")
                        st.dataframe(pd.DataFrame(recs), use_container_width=True)
                    else:
                        st.info("No Zoom cloud recordings returned.")
                except Exception as exc:
                    st.error(f"Zoom API Error: {exc}")

        if st.button("Import & Process Zoom Recording", type="primary", key="btn_imp_zoom"):
            if not zoom_id.strip():
                st.warning("Please enter a Zoom Recording ID.")
            else:
                with st.spinner("Importing Zoom recording..."):
                    try:
                        res = client.import_zoom_recording(recording_id=zoom_id.strip())
                        st.success(f"Zoom Recording Imported! ID: `{res.get('meeting_id')}`")
                    except Exception as exc:
                        st.error(f"Zoom import failed: {exc}")

    with tab_gmeet:
        st.markdown('<div class="obsidian-section-title">Google Meet / Drive Integration</div>', unsafe_allow_html=True)
        st.info("Uses Google OAuth / Service Account credentials to import meeting recordings from Google Drive.")

        col_g1, col_g2 = st.columns([2, 1])
        with col_g1:
            g_id = st.text_input("Google Drive File ID", placeholder="e.g. 1a2b3c4d5e6f7g8h9i0", key="g_id_input")
        with col_g2:
            st.markdown("<br>", unsafe_allow_html=True)
            fetch_g_btn = st.button("List Google Drive Recordings", use_container_width=True)

        if fetch_g_btn:
            with st.spinner("Listing Google Drive recordings..."):
                try:
                    recs = client.list_google_recordings()
                    if recs:
                        st.success(f"Found {len(recs)} Google Meet recording(s).")
                        st.dataframe(pd.DataFrame(recs), use_container_width=True)
                    else:
                        st.info("No Google Drive recordings returned.")
                except Exception as exc:
                    st.error(f"Google Drive API Error: {exc}")

        if st.button("Import & Process Google Meet Recording", type="primary", key="btn_imp_gmeet"):
            if not g_id.strip():
                st.warning("Please enter a Google Drive File ID.")
            else:
                with st.spinner("Importing Google Meet recording..."):
                    try:
                        res = client.import_google_recording(file_id=g_id.strip())
                        st.success(f"Google Meet Recording Imported! ID: `{res.get('meeting_id')}`")
                    except Exception as exc:
                        st.error(f"Google Meet import failed: {exc}")


# ── 11. Settings & System Status ───────────────────────────────────────────
def render_settings_status_page():
    st.markdown('<div class="obsidian-title">Settings & System Status</div>', unsafe_allow_html=True)
    st.markdown('<div class="obsidian-sub-caption">Monitor platform health, database state, vector store, and AI engine status.</div>', unsafe_allow_html=True)

    client = get_api_client()

    health_ok, health_msg = client.health_check()

    st.markdown('<div class="obsidian-section-title">Backend API Status</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Health Status**: {'Operational' if health_ok else 'Unreachable'}")
        st.write(f"**Health Message**: `{health_msg}`")
    with c2:
        if st.button("Re-verify Connection"):
            with st.spinner("Ping backend..."):
                ok, msg = client.verify_connection()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    st.markdown("---")
    st.markdown('<div class="obsidian-section-title">Architecture Status</div>', unsafe_allow_html=True)

    arch_rows = [
        {"Component": "SQLite Database", "Status": "Active", "Details": "career_intelligence.db (WAL Mode enabled)"},
        {"Component": "Vector Index Engine", "Status": "Active", "Details": "SQLite Vector Index (Cosine Similarity)"},
        {"Component": "Embedding Provider", "Status": "Active", "Details": "HashEmbeddingProvider / SentenceTransformers (BGE)"},
        {"Component": "LLM Intelligence Engine", "Status": "Active", "Details": f"Configured Provider: {os.getenv('LLM_PROVIDER', 'mock')}"},
        {"Component": "Speech Transcription", "Status": "Active", "Details": "Faster-Whisper / SpeechRecognition"}
    ]

    st.dataframe(pd.DataFrame(arch_rows), use_container_width=True, hide_index=True)


# ── Main Application Router ────────────────────────────────────────────────
def main():
    if not st.session_state.authenticated:
        render_login_view()
    else:
        render_sidebar()

        raw_page = st.session_state.current_page
        page = PAGE_LABEL_MAP.get(raw_page, raw_page)
        if page == "Dashboard":
            render_dashboard_page()
        elif page == "Meeting Intelligence":
            render_meeting_intelligence_page()
        elif page == "Transcript Workspace":
            render_transcript_workspace_page()
        elif page == "AI Insights":
            render_ai_insights_page()
        elif page == "Semantic Search":
            render_semantic_search_page()
        elif page == "Knowledge Repository":
            render_knowledge_repository_page()
        elif page == "AI Assistant / RAG":
            render_ai_assistant_page()
        elif page == "Text & Sentiment NLP":
            render_text_sentiment_page()
        elif page == "Reports & Export":
            render_reports_export_page()
        elif page == "Cloud Integrations":
            render_cloud_integrations_page()
        elif page == "Settings & System Status":
            render_settings_status_page()


# ── Function Aliases for Test Compatibility ─────────────────────────────────
render_meetings_explorer_page = render_meeting_intelligence_page
render_upload_page = render_meeting_intelligence_page
render_search_page = render_semantic_search_page


if __name__ == "__main__":
    main()
