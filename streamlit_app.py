"""
streamlit_app.py
================
AI-Powered Career Intelligence Platform — Streamlit Dashboard Frontend

Milestone 4 Task 1 Implementation.
Integrates directly with existing Flask backend API (app.py).
Does not duplicate backend business logic.
"""

import os
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional

from modules.api_client import MeetingApiClient

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Career Intelligence Platform — Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS Styling ──────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    
    /* Priority badges */
    .badge-high {
        background-color: #ef4444;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-medium {
        background-color: #f59e0b;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-low {
        background-color: #10b981;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-unknown {
        background-color: #6b7280;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ───────────────────────────────────────────
# ── Session State Initialization ───────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("API_KEY") or os.getenv("AUTH_TOKEN") or ""
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = os.getenv("API_BASE_URL", "http://localhost:5000")
if "current_page" not in st.session_state:
    st.session_state.current_page = "📊 Main Dashboard"
if "selected_meeting_id" not in st.session_state:
    st.session_state.selected_meeting_id = None
if "user_info" not in st.session_state:
    st.session_state.user_info = None


def get_api_client() -> MeetingApiClient:
    """Instantiate API client with current session settings."""
    return MeetingApiClient(
        base_url=st.session_state.api_base_url,
        api_key=st.session_state.api_key if st.session_state.api_key else None
    )


# ── Authentication View ────────────────────────────────────────────────────
def render_login_view():
    st.title("💼 AI Career Intelligence Platform")
    st.subheader("User Authentication & Connection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("Authenticate to access your private meeting intelligence, transcriptions, semantic search, and AI assistant.")
        
        auth_tab1, auth_tab2, auth_tab3 = st.tabs(["🔑 User Login", "📝 User Registration", "🔑 API Token Connection"])
        
        with auth_tab1:
            with st.form("user_login_form"):
                username_input = st.text_input("Username or Email", help="Your account username or email address")
                password_input = st.text_input("Password", type="password")
                login_btn = st.form_submit_button("🔑 Login", use_container_width=True)
                
                if login_btn:
                    if not username_input or not password_input:
                        st.error("Please enter both username/email and password.")
                    else:
                        client = get_api_client()
                        try:
                            res = client.login(username_or_email=username_input, password=password_input)
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
                register_btn = st.form_submit_button("📝 Register Account", use_container_width=True)
                
                if register_btn:
                    if not reg_username or not reg_email or not reg_password:
                        st.error("Please complete all registration fields.")
                    else:
                        client = get_api_client()
                        try:
                            res = client.register(username=reg_username, email=reg_email, password=reg_password)
                            st.session_state.api_key = res.get("token", "")
                            st.session_state.user_info = res.get("user")
                            st.session_state.authenticated = True
                            st.success(f"Registration successful! Logged in as {reg_username}.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Registration failed: {exc}")

        with auth_tab3:
            with st.form("token_connect_form"):
                api_url_input = st.text_input(
                    "Backend API Base URL",
                    value=st.session_state.api_base_url,
                    help="URL of the running app.py Flask server"
                )
                
                api_key_input = st.text_input(
                    "API Key / System Token",
                    value=st.session_state.api_key,
                    type="password",
                    help="System token or admin bearer key"
                )
                
                connect_btn = st.form_submit_button("🔗 Connect via Token", use_container_width=True)
                
                if connect_btn:
                    st.session_state.api_base_url = api_url_input.strip()
                    st.session_state.api_key = api_key_input.strip()
                    
                    client = get_api_client()
                    with st.spinner("Connecting to backend API..."):
                        success, message = client.verify_connection()
                    
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user_info = {"username": "System User", "id": "system"}
                        st.success("Successfully authenticated and connected!")
                        st.rerun()
                    else:
                        st.error(f"Authentication / Connection Failed: {message}")

    with col2:
        st.markdown("### Security & Architecture")
        st.markdown("- **User Isolation**: Protected meetings per account")
        st.markdown("- **RAG Context**: Isolated vector search metadata")
        st.markdown("- **Report Exports**: Access-controlled PDF/CSV generation")
        st.markdown("- **Backend Framework**: Flask API")


# ── Render Header & Sidebar ────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.title("💼 Career Intelligence")
    
    st.sidebar.markdown("---")
    user_name = st.session_state.user_info.get("username", "User") if st.session_state.user_info else "Authenticated User"
    st.sidebar.markdown(f"**👤 User**: `{user_name}`")
    st.sidebar.markdown(f"**API Host**: `{st.session_state.api_base_url}`")
    st.sidebar.markdown(f"**Auth Status**: 🟢 Authenticated")
    st.sidebar.markdown("---")
    
    pages = [
        "📊 Main Dashboard",
        "📅 Meetings Explorer",
        "📤 Upload Recording / Transcript",
        "🔍 Semantic Search",
        "🤖 AI Assistant (RAG)"
    ]
    
    selected = st.sidebar.radio("Navigation", pages, index=pages.index(st.session_state.current_page))
    st.session_state.current_page = selected
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        client = get_api_client()
        client.logout()
        st.session_state.authenticated = False
        st.session_state.api_key = ""
        st.session_state.user_info = None
        st.session_state.selected_meeting_id = None
        st.rerun()


# ── Page 1: Main Dashboard ──────────────────────────────────────────────────
def render_dashboard_page():
    st.title("📊 Main Dashboard")
    st.caption("AI-Powered Meeting Analytics & Intelligence Overview")
    
    client = get_api_client()
    
    try:
        with st.spinner("Loading meeting knowledge base..."):
            meetings_data = client.get_all_meetings_knowledge()
    except Exception as exc:
        st.error(f"Failed to fetch dashboard data from backend API: {exc}")
        return

    total_meetings = len(meetings_data)
    total_action_items = sum(len(m.get("action_items", [])) for m in meetings_data)
    total_decisions = sum(len(m.get("decisions", [])) for m in meetings_data)
    
    # Calculate unique participants
    unique_participants = set()
    for m in meetings_data:
        for p in m.get("participants", []):
            if isinstance(p, dict) and p.get("name"):
                unique_participants.add(p.get("name"))
            elif isinstance(p, str):
                unique_participants.add(p)

    # Metric Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Meetings</div>
            <div class="metric-value">{total_meetings}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Action Items</div>
            <div class="metric-value">{total_action_items}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Participants</div>
            <div class="metric-value">{len(unique_participants)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Decisions</div>
            <div class="metric-value">{total_decisions}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dashboard Content Tabs
    dash_tab1, dash_tab2 = st.tabs(["📅 Recent Meetings Overview", "🎯 Action Items Distribution"])
    
    with dash_tab1:
        if not meetings_data:
            st.info("No meetings processed yet. Upload a recording or transcript to populate the dashboard!")
        else:
            table_rows = []
            for m in meetings_data:
                table_rows.append({
                    "Meeting ID": m.get("id"),
                    "Title": m.get("title", "Untitled Meeting"),
                    "Date": m.get("created_at", "N/A"),
                    "Participants": len(m.get("participants", [])),
                    "Action Items": len(m.get("action_items", [])),
                    "Decisions": len(m.get("decisions", [])),
                    "Summary Preview": m.get("summary", "")[:120] + ("..." if len(m.get("summary", "")) > 120 else "")
                })
            
            df = pd.DataFrame(table_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.markdown("### Quick View Meeting Details")
            meeting_options = {m["id"]: f"{m.get('title')} ({m.get('created_at', 'N/A')})" for m in meetings_data}
            selected_id = st.selectbox("Select Meeting to Open in Explorer:", list(meeting_options.keys()), format_func=lambda x: meeting_options[x])
            
            if st.button("🔍 Open Meeting Details"):
                st.session_state.selected_meeting_id = selected_id
                st.session_state.current_page = "📅 Meetings Explorer"
                st.rerun()

    with dash_tab2:
        if not meetings_data:
            st.info("No action item data available.")
        else:
            all_actions = []
            for m in meetings_data:
                for a in m.get("action_items", []):
                    if isinstance(a, dict):
                        a_copy = dict(a)
                        a_copy["meeting_title"] = m.get("title")
                        all_actions.append(a_copy)
            
            if all_actions:
                df_actions = pd.DataFrame(all_actions)
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("#### Priority Distribution")
                    if "priority" in df_actions.columns:
                        st.bar_chart(df_actions["priority"].value_counts())
                
                with col_b:
                    st.markdown("#### Status Breakdown")
                    if "status" in df_actions.columns:
                        st.bar_chart(df_actions["status"].value_counts())
                
                st.markdown("#### All Extracted Action Items")
                st.dataframe(df_actions, use_container_width=True, hide_index=True)
            else:
                st.info("No action items found in historical meetings.")


# ── Page 2: Meetings Explorer & Analytics ──────────────────────────────────
def render_meetings_explorer_page():
    st.title("📅 Meetings Explorer & Analytics")
    st.caption("Deep-dive into specific meeting intelligence, transcripts, decisions, action items, and real-time analytics.")
    
    client = get_api_client()
    try:
        meetings = client.list_meetings()
    except Exception as exc:
        st.error(f"Failed to fetch meetings list: {exc}")
        return

    if not meetings:
        st.info("No processed meetings found in the system. Use the Upload tab to process a new meeting.")
        return

    meeting_map = {m["id"]: f"{m.get('title', 'Untitled')} — {str(m.get('created_at', ''))[:10]}" for m in meetings}
    
    # Pre-select if navigated from dashboard
    default_idx = 0
    if st.session_state.selected_meeting_id in meeting_map:
        keys_list = list(meeting_map.keys())
        default_idx = keys_list.index(st.session_state.selected_meeting_id)

    selected_id = st.selectbox(
        "Select Meeting to Inspect:",
        options=list(meeting_map.keys()),
        format_func=lambda x: meeting_map[x],
        index=default_idx
    )
    
    st.session_state.selected_meeting_id = selected_id

    if not selected_id:
        return

    try:
        with st.spinner("Fetching meeting details & analytics from backend API..."):
            m_details = client.get_meeting_details(selected_id)
    except Exception as exc:
        st.error(f"Error retrieving meeting details: {exc}")
        return

    # Extract meeting attributes safely
    meeting_title = m_details.get("title") or "Untitled Meeting"
    created_at = m_details.get("created_at") or "N/A"
    meeting_status = m_details.get("status") or "completed"
    raw_transcript = m_details.get("transcript") or m_details.get("raw_transcript") or ""
    summary = m_details.get("summary") or ""
    key_points = m_details.get("key_points") or []
    decisions = m_details.get("decisions") or []
    action_items = m_details.get("action_items") or []
    participants = m_details.get("participants") or []
    deadlines = m_details.get("deadlines") or []

    # Dynamic Analytics Calculations
    num_participants = len(participants)
    num_action_items = len(action_items)
    num_decisions = len(decisions)
    
    completed_actions = sum(1 for a in action_items if isinstance(a, dict) and a.get("status", "").lower() == "completed")
    pending_actions = sum(1 for a in action_items if isinstance(a, dict) and a.get("status", "").lower() in ("pending", "in progress", "unknown"))
    
    completion_rate = (completed_actions / num_action_items * 100) if num_action_items > 0 else 0.0
    word_count = m_details.get("word_count") or (len(raw_transcript.split()) if raw_transcript else 0)
    est_reading_time = max(1, word_count // 180) if word_count > 0 else 0

    # Header Card & Export Controls
    head_col1, head_col2 = st.columns([3, 2])
    with head_col1:
        st.markdown(f"## 📌 {meeting_title}")
        st.caption(f"🆔 Meeting ID: `{selected_id}` | 📅 Date: `{created_at}` | Status: 🟢 `{meeting_status.upper()}` | 📝 Word Count: `{word_count}` words")
    with head_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        exp_c1, exp_c2 = st.columns(2)
        safe_t = "".join(c for c in meeting_title if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
        with exp_c1:
            try:
                pdf_bytes = client.export_meeting_pdf(selected_id)
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"report_{selected_id}_{safe_t}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="dl_pdf_btn"
                )
            except Exception as pdf_err:
                st.error(f"PDF error: {pdf_err}")
        with exp_c2:
            try:
                csv_data = client.export_meeting_csv(selected_id)
                st.download_button(
                    label="📊 Download CSV Report",
                    data=csv_data,
                    file_name=f"report_{selected_id}_{safe_t}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_csv_btn"
                )
            except Exception as csv_err:
                st.error(f"CSV error: {csv_err}")
    st.markdown("---")


    # Tabs for Meeting Details & Analytics
    tab_analytics, tab_summary, tab_transcript, tab_actions, tab_decisions, tab_participants = st.tabs([
        "📊 Meeting Analytics",
        "📋 Summary & Key Points",
        "📜 Transcript",
        "✅ Action Items",
        "🎯 Decisions & Deadlines",
        "👥 Participants"
    ])

    # 1. ANALYTICS TAB
    with tab_analytics:
        st.markdown("### 📈 Meeting Analytics & Productivity Metrics")
        
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.metric("Participants", num_participants)
        with m_col2:
            st.metric("Action Items", num_action_items)
        with m_col3:
            st.metric("Decisions", num_decisions)
        with m_col4:
            st.metric("Completed Actions", completed_actions)
        with m_col5:
            st.metric("Completion Rate", f"{completion_rate:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts Row
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("#### Action Items by Priority")
            if action_items:
                priorities = [a.get("priority", "Unknown") for a in action_items if isinstance(a, dict)]
                p_df = pd.DataFrame(priorities, columns=["Priority"]).value_counts().reset_index(name="Count")
                st.bar_chart(p_df.set_index("Priority"))
            else:
                st.info("No action items available for priority breakdown.")

        with chart_col2:
            st.markdown("#### Action Items by Assignee")
            if action_items:
                assignees = [a.get("assigned_to") or "Unassigned" for a in action_items if isinstance(a, dict)]
                a_df = pd.DataFrame(assignees, columns=["Assignee"]).value_counts().reset_index(name="Count")
                st.bar_chart(a_df.set_index("Assignee"))
            else:
                st.info("No action items available for assignee breakdown.")

        st.markdown("---")
        st.markdown("#### ⏱️ Meeting Metadata & Reading Stats")
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        with meta_col1:
            st.write(f"**Transcript Word Count**: {word_count} words")
        with meta_col2:
            st.write(f"**Estimated Reading Time**: ~{est_reading_time} minutes")
        with meta_col3:
            st.write(f"**Action Completion Status**: {completed_actions} Done / {pending_actions} Pending")

    # 2. SUMMARY TAB
    with tab_summary:
        st.markdown("### Executive Summary")
        if summary:
            st.info(summary)
        else:
            st.info("No executive summary recorded for this meeting.")
        
        st.markdown("### Key Discussion Points")
        if key_points:
            for pt in key_points:
                st.markdown(f"- 🔹 {pt}")
        else:
            st.write("No key discussion points recorded.")

    # 3. TRANSCRIPT TAB
    with tab_transcript:
        st.markdown("### Transcript Viewer")
        if raw_transcript and raw_transcript.strip():
            st.caption(f"Total Words: {word_count} | Estimated Reading Time: {est_reading_time} mins")
            
            search_query = st.text_input("🔍 Search/Filter text inside transcript:", key="trans_filter_q")
            
            with st.expander("📖 Expandable Transcript View", expanded=True):
                if search_query.strip():
                    lines = raw_transcript.split("\n")
                    matched = [l for l in lines if search_query.lower() in l.lower()]
                    st.markdown(f"**Found {len(matched)} matching line(s):**")
                    for m in matched:
                        st.markdown(f"> ... {m} ...")
                else:
                    st.text_area("Full Raw Transcript", value=raw_transcript, height=400)
        else:
            st.info("No transcript recorded or transcript is empty for this meeting.")

    # 4. ACTION ITEMS TAB
    with tab_actions:
        st.markdown("### Action Items & Task Assignments")
        if action_items:
            df_act = pd.DataFrame(action_items)
            # Reorder columns cleanly if present
            expected_cols = ["task", "assigned_to", "deadline", "priority", "status"]
            cols = [c for c in expected_cols if c in df_act.columns] + [c for c in df_act.columns if c not in expected_cols]
            df_act = df_act[cols]
            st.dataframe(df_act, use_container_width=True, hide_index=True)
        else:
            st.info("No action items recorded for this meeting.")

    # 5. DECISIONS & DEADLINES TAB
    with tab_decisions:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("### Decisions Reached")
            if decisions:
                for d in decisions:
                    st.markdown(f"- 🎯 **{d}**")
            else:
                st.info("No decisions recorded for this meeting.")

        with col_d2:
            st.markdown("### Deadlines & Timelines")
            if deadlines:
                for dl in deadlines:
                    if isinstance(dl, dict):
                        st.markdown(f"- ⏰ **{dl.get('task', 'Task')}**: `{dl.get('deadline', 'Unspecified')}` (Assigned: {dl.get('assigned_to', 'Unassigned')})")
                    else:
                        st.markdown(f"- ⏰ **{dl}**")
            else:
                st.info("No specific deadlines recorded for this meeting.")

    # 6. PARTICIPANTS TAB
    with tab_participants:
        st.markdown("### Participant Information & Responsibilities")
        if participants:
            for p in participants:
                if isinstance(p, dict):
                    name = p.get("name", "Unknown Participant")
                    resps = p.get("responsibilities", [])
                    with st.expander(f"👤 {name}", expanded=True):
                        if resps:
                            st.markdown("**Assigned Responsibilities:**")
                            for r in resps:
                                st.markdown(f"  - {r}")
                        else:
                            st.markdown("*No specific responsibilities listed.*")
                elif isinstance(p, str):
                    st.markdown(f"- 👤 {p}")
        else:
            st.info("No participants recorded for this meeting.")



# ── Page 3: Upload Recording / Transcript ──────────────────────────────────
def render_upload_page():
    st.title("📤 Meeting Intelligence Upload & Processing")
    st.caption("Upload meeting audio/video recordings or raw transcripts for Whisper transcription and AI extraction.")
    
    client = get_api_client()
    
    upload_type = st.radio("Select Input Method:", ["🎙️ Audio / Video Recording File", "📝 Raw Transcript Text", "📹 Import from Zoom Cloud", "🎥 Import from Google Meet"], horizontal=True)
    
    title_input = st.text_input("Meeting Title (Optional)", placeholder="e.g. Q4 Career Roadmap & Strategy Session")
    
    if upload_type == "🎙️ Audio / Video Recording File":
        uploaded_file = st.file_uploader(
            "Choose meeting recording file:",
            type=["wav", "mp3", "flac", "ogg", "m4a", "mp4", "avi", "mov", "mkv", "webm"],
            help="Supported formats: WAV, MP3, FLAC, OGG, M4A, MP4, AVI, MOV, MKV, WEBM"
        )
        
        if uploaded_file is not None:
            st.write(f"📁 **Selected File**: `{uploaded_file.name}` ({uploaded_file.size / (1024*1024):.2f} MB)")
            
            if st.button("🚀 Process Recording with Backend Pipeline", type="primary"):
                with st.spinner("Transcribing speech with Whisper & extracting meeting intelligence..."):
                    try:
                        file_bytes = uploaded_file.read()
                        res = client.process_meeting(
                            file_bytes=file_bytes,
                            filename=uploaded_file.name,
                            title=title_input.strip() if title_input else uploaded_file.name
                        )
                        
                        st.success(f"✅ Meeting processed successfully! ID: `{res.get('meeting_id')}`")
                        
                        intelligence = res.get("intelligence", {})
                        st.markdown("### Extracted Summary Preview")
                        st.info(intelligence.get("summary", "No summary available."))
                        
                        if st.button("🔍 Open Processed Meeting Details"):
                            st.session_state.selected_meeting_id = res.get("meeting_id")
                            st.session_state.current_page = "📅 Meetings Explorer"
                            st.rerun()

                    except Exception as exc:
                        st.error(f"Meeting processing failed: {exc}")

    elif upload_type == "📝 Raw Transcript Text":
        transcript_text = st.text_area("Paste Raw Meeting Transcript Text:", height=250, placeholder="Speaker A: Let's discuss our product roadmap for the next quarter...\nSpeaker B: Sure, I will take the lead on backend API integration...")
        
        if st.button("🚀 Process Transcript Text", type="primary"):
            if not transcript_text.strip():
                st.warning("Please enter or paste transcript text before submitting.")
            else:
                with st.spinner("Processing transcript text through LLM intelligence pipeline..."):
                    try:
                        res = client.process_meeting(
                            transcript_text=transcript_text.strip(),
                            title=title_input.strip() if title_input else "Meeting Transcript"
                        )
                        st.success(f"✅ Transcript intelligence extracted! Meeting ID: `{res.get('meeting_id')}`")
                        
                        intelligence = res.get("intelligence", {})
                        st.markdown("### Extracted Summary Preview")
                        st.info(intelligence.get("summary", "No summary available."))

                        if st.button("🔍 View Complete Details"):
                            st.session_state.selected_meeting_id = res.get("meeting_id")
                            st.session_state.current_page = "📅 Meetings Explorer"
                            st.rerun()

                    except Exception as exc:
                        st.error(f"Processing failed: {exc}")

    elif upload_type == "📹 Import from Zoom Cloud":
        st.markdown("### 📹 Zoom Cloud Recording Import")
        st.info("Import authorized Zoom Cloud recordings directly into the pipeline using official Server-to-Server OAuth integration.")

        col_z1, col_z2 = st.columns([2, 1])
        with col_z1:
            zoom_id_input = st.text_input("Zoom Meeting / Recording ID", placeholder="e.g. 84920481920 or uuid", key="zoom_input_id")
        with col_z2:
            st.markdown("<br>", unsafe_allow_html=True)
            fetch_recordings_btn = st.button("🔄 Fetch Available Cloud Recordings", use_container_width=True)

        if fetch_recordings_btn:
            with st.spinner("Fetching cloud recordings from Zoom API..."):
                try:
                    recs = client.list_zoom_recordings()
                    if recs:
                        st.session_state["cached_zoom_recs"] = recs
                        st.success(f"Found {len(recs)} available cloud recording(s).")
                    else:
                        st.info("No cloud recordings found for this Zoom account.")
                except Exception as exc:
                    st.error(f"Zoom API Error: {exc}")

        cached_recs = st.session_state.get("cached_zoom_recs", [])
        if cached_recs:
            rec_map = {str(r.get("id")): f"{r.get('topic', 'Zoom Meeting')} ({str(r.get('start_time', ''))[:10]})" for r in cached_recs}
            selected_rec_id = st.selectbox("Select Recording from Zoom Account:", list(rec_map.keys()), format_func=lambda x: rec_map[x], key="zoom_select_dropdown")
            if selected_rec_id:
                zoom_id_input = selected_rec_id

        if st.button("🚀 Import & Process Zoom Recording", type="primary", key="btn_import_zoom"):
            if not zoom_id_input.strip():
                st.warning("Please enter or select a Zoom recording ID.")
            else:
                with st.spinner("Downloading Zoom recording → Transcribing with Whisper → Processing LLM → Persisting to SQLite DB..."):
                    try:
                        res = client.import_zoom_recording(
                            recording_id=zoom_id_input.strip(),
                            title=title_input.strip() if title_input else None
                        )

                        if res.get("status") == "duplicate":
                            st.warning(f"⚠️ {res.get('message')}")
                        else:
                            st.success(f"✅ Zoom recording imported successfully! ID: `{res.get('meeting_id')}`")
                            st.markdown("### Summary Preview")
                            st.info(res.get("summary", "No summary available."))

                        target_id = res.get("meeting_id")
                        if target_id and st.button("🔍 Open Processed Meeting Details", key="zoom_open_details_btn"):
                            st.session_state.selected_meeting_id = target_id
                            st.session_state.current_page = "📅 Meetings Explorer"
                            st.rerun()

                    except Exception as exc:
                        st.error(f"Zoom import failed: {exc}")

    else: # Google Meet Import
        st.markdown("### 🎥 Google Meet Recording Import")
        st.info("Import authorized Google Meet recordings from Google Drive into the pipeline using Google OAuth integration.")

        col_g1, col_g2 = st.columns([2, 1])
        with col_g1:
            gmeet_id_input = st.text_input("Google Drive File / Recording ID", placeholder="e.g. 1a2b3c4d5e6f7g8h9i0", key="gmeet_input_id")
        with col_g2:
            st.markdown("<br>", unsafe_allow_html=True)
            fetch_gmeet_btn = st.button("🔄 Fetch Available Google Meet Recordings", use_container_width=True, key="btn_fetch_gmeet")

        if fetch_gmeet_btn:
            with st.spinner("Fetching recordings from Google Drive API..."):
                try:
                    recs = client.list_google_recordings()
                    if recs:
                        st.session_state["cached_gmeet_recs"] = recs
                        st.success(f"Found {len(recs)} available Google Meet recording(s).")
                    else:
                        st.info("No recordings found in Google Drive.")
                except Exception as exc:
                    st.error(f"Google Drive API Error: {exc}")

        cached_gmeet = st.session_state.get("cached_gmeet_recs", [])
        if cached_gmeet:
            g_map = {str(r.get("id")): f"{r.get('name', 'Google Meet Recording')} ({str(r.get('createdTime', ''))[:10]})" for r in cached_gmeet}
            selected_g_id = st.selectbox("Select Recording from Google Drive:", list(g_map.keys()), format_func=lambda x: g_map[x], key="gmeet_select_dropdown")
            if selected_g_id:
                gmeet_id_input = selected_g_id

        if st.button("🚀 Import & Process Google Meet Recording", type="primary", key="btn_import_gmeet"):
            if not gmeet_id_input.strip():
                st.warning("Please enter or select a Google Drive file ID.")
            else:
                with st.spinner("Downloading Google Meet recording → Transcribing with Whisper → Processing LLM → Persisting to SQLite DB..."):
                    try:
                        res = client.import_google_recording(
                            file_id=gmeet_id_input.strip(),
                            title=title_input.strip() if title_input else None
                        )

                        if res.get("status") == "duplicate":
                            st.warning(f"⚠️ {res.get('message')}")
                        else:
                            st.success(f"✅ Google Meet recording imported successfully! ID: `{res.get('meeting_id')}`")
                            st.markdown("### Summary Preview")
                            st.info(res.get("summary", "No summary available."))

                        target_id = res.get("meeting_id")
                        if target_id and st.button("🔍 Open Processed Meeting Details", key="gmeet_open_details_btn"):
                            st.session_state.selected_meeting_id = target_id
                            st.session_state.current_page = "📅 Meetings Explorer"
                            st.rerun()

                    except Exception as exc:
                        st.error(f"Google Meet import failed: {exc}")




# ── Page 4: Semantic Search ────────────────────────────────────────────────
def render_search_page():
    st.title("🔍 Natural Language Semantic Search")
    st.caption("Perform semantic vector search across meeting transcripts, summaries, decisions, and action items via existing /search endpoint.")
    
    client = get_api_client()

    col_q, col_k = st.columns([3, 1])
    with col_q:
        query_input = st.text_input(
            "Enter Search Query:",
            placeholder="e.g. Which meeting discussed the database migration?",
            key="semantic_search_query_input"
        )
    with col_k:
        top_k = st.slider("Top Results (K)", min_value=1, max_value=20, value=5, key="search_top_k")

    with st.expander("⚙️ Advanced Filters & Metadata Controls", expanded=True):
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            content_type = st.selectbox(
                "Content Type Filter",
                ["All Types", "summary", "transcript", "decision", "action_item", "key_point"],
                key="search_content_type"
            )
            content_type_val = None if content_type == "All Types" else content_type
            
        with col_f2:
            # Optional meeting filter
            try:
                meetings = client.list_meetings()
                m_options = {"All Meetings": None}
                for m in meetings:
                    m_options[f"{m.get('title', 'Untitled')} ({m.get('id')})"] = m.get('id')
                selected_m_label = st.selectbox("Target Meeting Filter", list(m_options.keys()), key="search_meeting_filter")
                meeting_id_val = m_options[selected_m_label]
            except Exception:
                meeting_id_val = None

        with col_f3:
            enable_date_filter = st.checkbox("Filter by Date Range", value=False, key="search_enable_dates")
            if enable_date_filter:
                date_range = st.date_input("Date Range (Start - End)", value=[], key="search_date_range")
                start_date_str = date_range[0].strftime("%Y-%m-%d") if len(date_range) > 0 else None
                end_date_str = date_range[1].strftime("%Y-%m-%d") if len(date_range) > 1 else None
            else:
                start_date_str = None
                end_date_str = None

        with col_f4:
            min_score = st.slider("Min Similarity Score", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="search_min_score")
            deduplicate = st.checkbox("Deduplicate Results", value=False, key="search_dedup")

    if st.button("🔎 Execute Semantic Search", type="primary", use_container_width=True):
        if not query_input.strip():
            st.warning("Please enter a natural language search query.")
        else:
            with st.spinner("Executing vector search via backend /search endpoint..."):
                try:
                    res = client.semantic_search(
                        query=query_input.strip(),
                        top_k=top_k,
                        content_type=content_type_val,
                        meeting_id=meeting_id_val,
                        start_date=start_date_str,
                        end_date=end_date_str,
                        min_score=min_score if min_score > 0 else None,
                        deduplicate=deduplicate
                    )
                    
                    results = res.get("results", [])
                    st.markdown(f"### 🎯 Search Results ({len(results)} matches found in `{res.get('latency_ms', 0):.1f}ms`)")
                    
                    if not results:
                        st.info("No matching meeting content found for your query with the selected filters.")
                    else:
                        for idx, item in enumerate(results, 1):
                            score = item.get("similarity_score") or item.get("similarity") or item.get("score") or 0.0
                            c_type = item.get("content_type", "chunk")
                            meeting_title = item.get("meeting_title") or item.get("title") or "Meeting"
                            meeting_id = item.get("meeting_id") or "N/A"
                            meeting_date = str(item.get("created_at", "N/A"))[:10]
                            snippet = item.get("text") or item.get("relevant_snippet") or ""

                            with st.container():
                                st.markdown(f"#### #{idx} | 📌 **{meeting_title}**")
                                st.caption(f"📅 Date: `{meeting_date}` | Tag: `{c_type.upper()}` | Similarity Score: `{score:.4f}` | ID: `{meeting_id}`")
                                st.info(f"**Relevant Snippet:**\n\n{snippet}")
                                
                                if meeting_id and meeting_id != "N/A":
                                    if st.button(f"🔍 Open Meeting Details ({meeting_id})", key=f"open_mtg_{idx}_{meeting_id}"):
                                        st.session_state.selected_meeting_id = meeting_id
                                        st.session_state.current_page = "📅 Meetings Explorer"
                                        st.rerun()
                                st.markdown("---")

                except Exception as exc:
                    st.error(f"Semantic search failed: {exc}")


# ── Page 5: AI Assistant (RAG) ──────────────────────────────────────────────
def render_ai_assistant_page():
    st.title("🤖 AI Grounded Assistant (RAG)")
    st.caption("Ask questions about historical career meetings. Answers are strictly grounded in retrieved meeting records via existing /ask RAG endpoint.")
    
    client = get_api_client()

    col_q, col_k = st.columns([3, 1])
    with col_q:
        question_input = st.text_input(
            "Enter your question for the AI Assistant:",
            placeholder="e.g. What deadline was decided for the mobile application?",
            key="rag_question_input"
        )
    with col_k:
        top_k = st.slider("Context Chunks (K)", min_value=1, max_value=10, value=5, key="rag_top_k")

    with st.expander("⚙️ Advanced Retrieval & Grounding Controls", expanded=False):
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            content_type = st.selectbox(
                "Filter Context Content Type",
                ["All Types", "summary", "transcript", "decision", "action_item", "key_point"],
                key="rag_content_type"
            )
            content_type_val = None if content_type == "All Types" else content_type

        with col_r2:
            try:
                meetings = client.list_meetings()
                m_options = {"All Meetings": None}
                for m in meetings:
                    m_options[f"{m.get('title', 'Untitled')} ({m.get('id')})"] = m.get('id')
                selected_m_label = st.selectbox("Filter Target Meeting", list(m_options.keys()), key="rag_meeting_filter")
                meeting_id_val = m_options[selected_m_label]
            except Exception:
                meeting_id_val = None

        with col_r3:
            enable_date_filter = st.checkbox("Filter by Date Range", value=False, key="rag_enable_dates")
            if enable_date_filter:
                date_range = st.date_input("Date Range (Start - End)", value=[], key="rag_date_range")
                start_date_str = date_range[0].strftime("%Y-%m-%d") if len(date_range) > 0 else None
                end_date_str = date_range[1].strftime("%Y-%m-%d") if len(date_range) > 1 else None
            else:
                start_date_str = None
                end_date_str = None

    if st.button("💬 Ask Grounded AI Assistant", type="primary", use_container_width=True):
        if not question_input.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving grounded context & generating AI answer..."):
                try:
                    res = client.ask_assistant(
                        question=question_input.strip(),
                        top_k=top_k,
                        content_type=content_type_val,
                        meeting_id=meeting_id_val,
                        start_date=start_date_str,
                        end_date=end_date_str
                    )
                    
                    answer_text = res.get("answer") or "I couldn't find enough information in the available meeting records to answer this question."
                    sources = res.get("sources") or []

                    st.markdown("### 🤖 Answer")
                    if "couldn't find enough information" in answer_text.lower() or not sources:
                        st.warning(f"⚠️ **Grounded Response**: {answer_text}")
                    else:
                        st.success(f"**Grounded Answer:**\n\n{answer_text}")
                    
                    st.markdown("---")
                    st.markdown(f"### 📚 Grounded Source Knowledge ({len(sources)} context chunks retrieved in `{res.get('latency_ms', 0):.1f}ms`)")
                    
                    if not sources:
                        st.info("No matching source context chunks were retrieved for this query.")
                    else:
                        for idx, src in enumerate(sources, 1):
                            m_title = src.get("meeting_title") or src.get("title") or "Meeting"
                            m_id = src.get("meeting_id") or "N/A"
                            m_date = str(src.get("created_at", "N/A"))[:10]
                            score = src.get("similarity_score") or src.get("similarity") or src.get("score") or 0.0
                            c_type = src.get("content_type", "chunk")
                            snippet = src.get("text") or src.get("relevant_snippet") or ""

                            with st.expander(f"Source #{idx} | 📌 {m_title} | Score: {score:.4f}", expanded=(idx==1)):
                                st.markdown(f"**Source Meeting**: `{m_title}`")
                                st.markdown(f"**Meeting Date**: `{m_date}` | **Content Type**: `{c_type.upper()}` | **ID**: `{m_id}`")
                                st.markdown(f"**Retrieved Context Snippet:**\n\n> {snippet}")
                                
                                if m_id and m_id != "N/A":
                                    if st.button(f"🔍 Open Source Meeting Details ({m_id})", key=f"rag_open_mtg_{idx}_{m_id}"):
                                        st.session_state.selected_meeting_id = m_id
                                        st.session_state.current_page = "📅 Meetings Explorer"
                                        st.rerun()

                except Exception as exc:
                    st.error(f"RAG Assistant query failed: {exc}")


# ── Main Entry Point ────────────────────────────────────────────────────────
def main():
    if not st.session_state.authenticated:
        render_login_view()
    else:
        render_sidebar()
        
        page = st.session_state.current_page
        if page == "📊 Main Dashboard":
            render_dashboard_page()
        elif page == "📅 Meetings Explorer":
            render_meetings_explorer_page()
        elif page == "📤 Upload Recording / Transcript":
            render_upload_page()
        elif page == "🔍 Semantic Search":
            render_search_page()
        elif page == "🤖 AI Assistant (RAG)":
            render_ai_assistant_page()


if __name__ == "__main__":
    main()

