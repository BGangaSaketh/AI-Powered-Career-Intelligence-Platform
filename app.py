"""
app.py
======
AI-Powered Career Intelligence Platform — Flask Application Entry Point

Milestone 1 Routes:
  GET  /                  → Serve the web UI
  POST /api/analyze       → Run the full text pipeline
  POST /api/transcribe    → Transcribe audio/video and optionally analyze

Milestone 2 Meeting Intelligence Routes:
  POST /meetings/process      (alias /api/meetings/process)   → Complete meeting processing pipeline
  GET  /meetings/<meeting_id> (alias /api/meetings/<meeting_id>) → Retrieve meeting intelligence by ID
  GET  /meetings              (alias /api/meetings)            → List all processed meetings
"""

import os
import uuid
import logging
from functools import wraps

from flask import Flask, request, jsonify, render_template, Response, g
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── Internal modules ──────────────────────────────────────────────────────
from modules.ingestion      import ingest_text, ingest_txt_file, ingest_csv_file
from modules.preprocessing  import preprocess
from modules.sentiment      import analyze_sentiment
from modules.reporting      import generate_report
from modules.transcription  import transcribe_file
from modules.report_generator import generate_meeting_pdf_report, generate_meeting_csv_report


# Milestone 2 & Milestone 3 Modules
from modules.meeting_service import process_meeting_input
from modules.database import (
    get_meeting,
    list_meetings,
    init_db,
    get_meeting_metadata,
    get_meeting_transcript,
    get_meeting_summary,
    get_meeting_decisions,
    get_meeting_action_items,
    get_meeting_participants,
    get_meeting_deadlines,
    get_all_meetings_knowledge,
    create_user,
    authenticate_user,
    get_user_by_token,
    get_user_by_id,
    invalidate_user_token
)
from modules.semantic_search import SemanticSearchService
from modules.rag_service import RAGService
from modules.zoom_service import ZoomService, ZoomAuthError, ZoomApiError
from modules.google_meet_service import GoogleMeetService, GoogleAuthError, GoogleApiError



# ── App setup ─────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Upload configuration
UPLOAD_FOLDER          = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_TEXT_EXTS      = {"txt", "csv"}
ALLOWED_MEDIA_EXTS     = {"wav", "mp3", "flac", "ogg", "m4a", "mp4", "avi", "mov", "mkv", "webm"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
init_db()


# ── Helpers & Authentication ───────────────────────────────────────────────

def _allowed_file(filename: str, allowed_set: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


def check_auth(req) -> bool:
    """
    Verify authentication credentials and populate Flask g.user and g.user_id.
    Checks environment variable REQUIRE_AUTH ("true"/"1") or API_KEY / AUTH_TOKEN.
    Accepts credentials via:
      - Header: Authorization: Bearer <token>
      - Header: X-API-Key: <key>
      - Query param / form field / JSON body: api_key or token
    """
    require_auth = os.getenv("REQUIRE_AUTH", "false").lower() in ("true", "1", "yes")
    expected_key = os.getenv("API_KEY") or os.getenv("AUTH_TOKEN")

    auth_header = req.headers.get("Authorization", "").strip()
    provided_token = None
    if auth_header.startswith("Bearer "):
        provided_token = auth_header[7:].strip()

    if not provided_token:
        provided_token = req.headers.get("X-API-Key", "").strip()

    if not provided_token:
        json_data = req.get_json(silent=True) or {}
        provided_token = req.args.get("api_key") or req.args.get("token") or req.form.get("api_key") or json_data.get("api_key") or json_data.get("token")

    g.user = None
    g.user_id = None

    if provided_token:
        user_record = get_user_by_token(provided_token)
        if user_record:
            g.user = user_record
            g.user_id = user_record["id"]
            return True
        elif expected_key and provided_token == expected_key:
            g.user = {"id": "system", "username": "system_admin"}
            g.user_id = "system"
            return True
        elif not require_auth and not expected_key:
            g.user = {"id": "system", "username": "system_admin"}
            g.user_id = "system"
            return True
        else:
            return False

    if not require_auth and not expected_key:
        g.user = {"id": "system", "username": "default_user"}
        g.user_id = "system"
        return True

    return False


def require_authentication(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not check_auth(request):
            logger.warning(f"Unauthorized access attempt to '{request.path}'")
            return jsonify({"status": "error", "message": "Unauthorized access. Invalid or missing authentication credentials."}), 401
        return f(*args, **kwargs)
    return decorated


def _run_pipeline(raw_text: str, ingestion_result: dict) -> dict:
    """
    Run preprocessing → sentiment → report on validated text.
    Returns the full report dict.
    """
    preprocessing_result = preprocess(raw_text)
    sentiment_result     = analyze_sentiment(raw_text)
    report               = generate_report(ingestion_result, preprocessing_result, sentiment_result)

    return {
        "report":               report,
        "preprocessing_detail": {
            "sentences":       preprocessing_result.get("sentences",       []),
            "tokens":          preprocessing_result.get("tokens_no_punct", [])[:50],  # cap for UI
            "filtered_tokens": preprocessing_result.get("filtered_tokens", [])[:50],
            "lemmas":          preprocessing_result.get("lemmas",          [])[:50],
        },
        "sentiment_detail": {
            "compound":     sentiment_result.get("compound",     0.0),
            "pos":          sentiment_result.get("pos",          0.0),
            "neg":          sentiment_result.get("neg",          0.0),
            "neu":          sentiment_result.get("neu",          0.0),
            "label":        sentiment_result.get("label",        "Neutral"),
            "per_sentence": sentiment_result.get("per_sentence", []),
        },
    }


# ── System Health Routes ───────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health_check():
    """System health check endpoint for orchestrator/launchers."""
    return jsonify({
        "status": "ok",
        "service": "AI-Powered Career Intelligence Platform API",
        "message": "Backend API operational"
    }), 200


# ── User Authentication Routes ──────────────────────────────────────────────

@app.route("/auth/register", methods=["POST"])
@app.route("/api/auth/register", methods=["POST"])
def register_endpoint():
    """Register a new user account."""
    try:
        data = request.get_json(silent=True) or request.form.to_dict() or {}
        username = str(data.get("username") or "").strip()
        email = str(data.get("email") or "").strip()
        password = str(data.get("password") or "")

        if not username or not email or not password:
            return jsonify({"status": "error", "message": "Username, email, and password parameters are required."}), 400

        user_info = create_user(username=username, email=email, password=password)
        return jsonify({
            "status": "ok",
            "message": "User registered successfully.",
            "user": {
                "id": user_info["id"],
                "username": user_info["username"],
                "email": user_info["email"],
                "created_at": user_info["created_at"]
            },
            "token": user_info["token"]
        }), 201

    except ValueError as val_err:
        return jsonify({"status": "error", "message": str(val_err)}), 400
    except Exception as exc:
        logger.exception("User registration error")
        return jsonify({"status": "error", "message": f"Registration failed: {exc}"}), 500


@app.route("/auth/login", methods=["POST"])
@app.route("/api/auth/login", methods=["POST"])
def login_endpoint():
    """Authenticate user credentials and return access token."""
    try:
        data = request.get_json(silent=True) or request.form.to_dict() or {}
        username_or_email = str(data.get("username") or data.get("email") or data.get("username_or_email") or "").strip()
        password = str(data.get("password") or "")

        if not username_or_email or not password:
            return jsonify({"status": "error", "message": "Username/email and password parameters are required."}), 400

        user_info = authenticate_user(username_or_email=username_or_email, password=password)
        if not user_info:
            return jsonify({"status": "error", "message": "Invalid username/email or password."}), 401

        return jsonify({
            "status": "ok",
            "message": "Login successful.",
            "user": {
                "id": user_info["id"],
                "username": user_info["username"],
                "email": user_info["email"],
                "created_at": user_info["created_at"]
            },
            "token": user_info["token"]
        }), 200

    except Exception as exc:
        logger.exception("User login error")
        return jsonify({"status": "error", "message": f"Login failed: {exc}"}), 500


@app.route("/auth/logout", methods=["POST"])
@app.route("/api/auth/logout", methods=["POST"])
def logout_endpoint():
    """Invalidate session token."""
    try:
        auth_header = request.headers.get("Authorization", "").strip()
        token = None
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        if not token:
            token = request.headers.get("X-API-Key", "").strip()
        if not token:
            data = request.get_json(silent=True) or {}
            token = request.args.get("token") or data.get("token")

        if token:
            invalidate_user_token(token)

        return jsonify({"status": "ok", "message": "Successfully logged out."}), 200
    except Exception as exc:
        logger.exception("User logout error")
        return jsonify({"status": "error", "message": f"Logout failed: {exc}"}), 500


@app.route("/auth/me", methods=["GET"])
@app.route("/api/auth/me", methods=["GET"])
@require_authentication
def get_current_user_endpoint():
    """Retrieve profile of currently authenticated user."""
    return jsonify({"status": "ok", "user": g.user}), 200


# ── Milestone 1 Routes ─────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main web UI."""
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Analyze text from any of the supported input methods:
      - form field 'text'         → raw text
      - file upload 'txt_file'    → .txt file
      - file upload 'csv_file'    → .csv file
    """
    ingestion_result = None

    if "csv_file" in request.files and request.files["csv_file"].filename:
        f = request.files["csv_file"]
        if not _allowed_file(f.filename, {"csv"}):
            return jsonify({"status": "error", "message": "Only .csv files are allowed."}), 400
        ingestion_result = ingest_csv_file(f)

    elif "txt_file" in request.files and request.files["txt_file"].filename:
        f = request.files["txt_file"]
        if not _allowed_file(f.filename, {"txt"}):
            return jsonify({"status": "error", "message": "Only .txt files are allowed."}), 400
        ingestion_result = ingest_txt_file(f)

    else:
        json_data = request.get_json(silent=True) or {}
        raw_text = (json_data.get("text") or request.form.get("text") or "").strip()
        ingestion_result = ingest_text(raw_text)


    if ingestion_result["status"] == "error":
        return jsonify({
            "status":  "error",
            "message": " | ".join(ingestion_result["errors"]),
        }), 400

    try:
        result = _run_pipeline(ingestion_result["raw_text"], ingestion_result)
        result["status"] = "ok"
        result["source"] = ingestion_result["source"]
        return jsonify(result), 200
    except Exception as exc:
        logger.exception("Pipeline error")
        return jsonify({"status": "error", "message": f"Pipeline error: {exc}"}), 500


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    """
    Transcribe speech from an uploaded audio or video file.
    Optionally runs the full NLP pipeline on the transcript.
    """
    if "media_file" not in request.files or not request.files["media_file"].filename:
        return jsonify({"status": "error", "message": "No media file provided."}), 400

    f = request.files["media_file"]
    if not _allowed_file(f.filename, ALLOWED_MEDIA_EXTS):
        return jsonify({
            "status":  "error",
            "message": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_MEDIA_EXTS))}",
        }), 400

    safe_name   = secure_filename(f.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    save_path   = os.path.join(UPLOAD_FOLDER, unique_name)
    f.save(save_path)

    try:
        transcription_result = transcribe_file(save_path)

        if transcription_result["status"] == "error":
            return jsonify({
                "status":  "error",
                "message": " | ".join(transcription_result["errors"]),
            }), 400

        transcript = transcription_result["transcript"]

        analyze_flag = request.form.get("analyze", "true").lower() == "true"
        pipeline_result = {}
        if analyze_flag and transcript:
            ingestion_result = ingest_text(transcript)
            if ingestion_result["status"] == "ok":
                pipeline_result = _run_pipeline(transcript, ingestion_result)

        return jsonify({
            "status":       "ok",
            "transcript":   transcript,
            "file_type":    transcription_result["file_type"],
            "pipeline":     pipeline_result,
        }), 200

    except Exception as exc:
        logger.exception("Transcription error")
        return jsonify({"status": "error", "message": f"Transcription error: {exc}"}), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


# ── Milestone 2 Meeting Intelligence Routes ────────────────────────────────

@app.route("/meetings/process", methods=["POST"])
@app.route("/api/meetings/process", methods=["POST"])
@require_authentication
def process_meeting_endpoint():
    """
    Process meeting recording or transcript:
      - Media File ('media_file') OR Raw Transcript ('transcript')
      - Title ('title', optional)
    Runs Whisper transcription -> LLM extraction -> Schema Validation -> DB Persistence
    """
    title = request.form.get("title", "Meeting Recording").strip()
    raw_transcript = request.form.get("transcript", "").strip()

    save_path = None
    if "media_file" in request.files and request.files["media_file"].filename:
        f = request.files["media_file"]
        if not _allowed_file(f.filename, ALLOWED_MEDIA_EXTS):
            return jsonify({
                "status": "error",
                "message": f"Unsupported media format. Allowed: {', '.join(sorted(ALLOWED_MEDIA_EXTS))}"
            }), 400
        safe_name = secure_filename(f.filename)
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        save_path = os.path.join(UPLOAD_FOLDER, unique_name)
        f.save(save_path)
        if not title or title == "Meeting Recording":
            title = f.filename

    if not save_path and not raw_transcript:
        return jsonify({"status": "error", "message": "Provide an audio/video file or transcript text."}), 400

    try:
        user_id = getattr(g, "user_id", None)
        result = process_meeting_input(
            file_path=save_path,
            raw_transcript_input=raw_transcript,
            title=title,
            user_id=user_id
        )
        return jsonify(result), 200
    except ValueError as val_err:
        return jsonify({"status": "error", "message": str(val_err)}), 400
    except Exception as exc:
        logger.exception("Meeting processing error")
        return jsonify({"status": "error", "message": f"Meeting processing failed: {exc}"}), 500
    finally:
        if save_path and os.path.exists(save_path):
            os.remove(save_path)


@app.route("/meetings/<meeting_id>", methods=["GET"])
@app.route("/api/meetings/<meeting_id>", methods=["GET"])
@require_authentication
def get_meeting_endpoint(meeting_id: str):
    """Retrieve processed meeting intelligence by meeting ID."""
    try:
        user_id = getattr(g, "user_id", None)
        meeting_data = get_meeting(meeting_id, user_id=user_id)
        if not meeting_data:
            if get_meeting(meeting_id):
                return jsonify({"status": "error", "message": f"Access denied. Meeting '{meeting_id}' belongs to another user."}), 403
            return jsonify({"status": "error", "message": f"Meeting '{meeting_id}' not found."}), 404
        meeting_data["status"] = "ok"
        return jsonify(meeting_data), 200
    except Exception as exc:
        logger.exception(f"Error fetching meeting '{meeting_id}'")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/meetings", methods=["GET"])
@app.route("/api/meetings", methods=["GET"])
@require_authentication
def list_meetings_endpoint():
    """List all processed meetings for authenticated user."""
    try:
        user_id = getattr(g, "user_id", None)
        meetings = list_meetings(user_id=user_id)
        return jsonify({"status": "ok", "meetings": meetings}), 200
    except Exception as exc:
        logger.exception("Error listing meetings")
        return jsonify({"status": "error", "message": str(exc)}), 500


# ── Milestone 4 Zoom Integration Routes ────────────────────────────────────

zoom_service = ZoomService()


@app.route("/zoom/recordings", methods=["GET"])
@app.route("/api/zoom/recordings", methods=["GET"])
@require_authentication
def list_zoom_recordings_endpoint():
    """List available Zoom cloud recordings."""
    try:
        from_date = request.args.get("from_date") or request.args.get("from")
        to_date = request.args.get("to_date") or request.args.get("to")
        user_id = request.args.get("user_id", "me")
        
        recordings = zoom_service.list_cloud_recordings(user_id=user_id, from_date=from_date, to_date=to_date)
        return jsonify({"status": "ok", "recordings": recordings}), 200
    except ZoomAuthError as auth_err:
        return jsonify({"status": "error", "message": f"Zoom Auth Error: {auth_err}"}), 401
    except Exception as exc:
        logger.exception("Error listing Zoom recordings")
        return jsonify({"status": "error", "message": f"Zoom retrieval failed: {exc}"}), 500


@app.route("/zoom/import", methods=["POST"])
@app.route("/api/zoom/import", methods=["POST"])
@require_authentication
def import_zoom_recording_endpoint():
    """Import a specific Zoom cloud recording into the processing pipeline."""
    try:
        data = request.get_json(silent=True) or request.form.to_dict() or {}
        recording_id = data.get("recording_id") or data.get("meeting_id")
        download_url = data.get("download_url")
        title = data.get("title")

        if not recording_id:
            return jsonify({"status": "error", "message": "recording_id parameter is required."}), 400

        result = zoom_service.import_zoom_recording(
            recording_id=recording_id,
            download_url=download_url,
            title=title
        )
        return jsonify(result), 200
    except ZoomAuthError as auth_err:
        return jsonify({"status": "error", "message": f"Zoom Auth Error: {auth_err}"}), 401
    except ValueError as val_err:
        return jsonify({"status": "error", "message": str(val_err)}), 400
    except Exception as exc:
        logger.exception("Error importing Zoom recording")
        return jsonify({"status": "error", "message": f"Zoom import failed: {exc}"}), 500


@app.route("/zoom/webhook", methods=["POST"])
@app.route("/api/zoom/webhook", methods=["POST"])
def zoom_webhook_endpoint():
    """Handle Zoom Cloud Recording Completed webhooks and URL validation challenges."""
    try:
        payload = request.get_json(silent=True) or {}
        event = payload.get("event")

        # 1. URL Validation Challenge
        if event == "endpoint.url_validation" or "plainToken" in payload.get("payload", {}):
            plain_token = payload.get("payload", {}).get("plainToken") or request.args.get("plainToken")
            res_payload = zoom_service.validate_webhook_url(plain_token)
            return jsonify(res_payload), 200

        # 2. Recording Completed Event
        if event == "recording.completed":
            object_data = payload.get("payload", {}).get("object", {})
            recording_id = str(object_data.get("id") or object_data.get("uuid"))
            topic = object_data.get("topic")

            if recording_id:
                logger.info(f"Received Zoom recording.completed webhook for meeting '{recording_id}'. Processing...")
                res = zoom_service.import_zoom_recording(recording_id=recording_id, title=topic)
                return jsonify({"status": "ok", "message": "Webhook processed.", "result": res}), 200

        return jsonify({"status": "ok", "message": "Event ignored."}), 200

    except Exception as exc:
        logger.exception("Error handling Zoom webhook")
        return jsonify({"status": "error", "message": f"Webhook error: {exc}"}), 500


# ── Milestone 4 Google Meet Integration Routes ──────────────────────────────

google_meet_service = GoogleMeetService()


@app.route("/google/recordings", methods=["GET"])
@app.route("/api/google/recordings", methods=["GET"])
@require_authentication
def list_google_recordings_endpoint():
    """List available Google Meet cloud recordings from Google Drive."""
    try:
        folder_id = request.args.get("folder_id")
        recordings = google_meet_service.list_meet_recordings(folder_id=folder_id)
        return jsonify({"status": "ok", "recordings": recordings}), 200
    except GoogleAuthError as auth_err:
        return jsonify({"status": "error", "message": f"Google Auth Error: {auth_err}"}), 401
    except Exception as exc:
        logger.exception("Error listing Google Meet recordings")
        return jsonify({"status": "error", "message": f"Google Meet retrieval failed: {exc}"}), 500


@app.route("/google/import", methods=["POST"])
@app.route("/api/google/import", methods=["POST"])
@require_authentication
def import_google_recording_endpoint():
    """Import a specific Google Meet cloud recording into the processing pipeline."""
    try:
        data = request.get_json(silent=True) or request.form.to_dict() or {}
        file_id = data.get("file_id") or data.get("recording_id")
        title = data.get("title")

        if not file_id:
            return jsonify({"status": "error", "message": "file_id parameter is required."}), 400

        result = google_meet_service.import_google_meet_recording(file_id=file_id, title=title)
        return jsonify(result), 200
    except GoogleAuthError as auth_err:
        return jsonify({"status": "error", "message": f"Google Auth Error: {auth_err}"}), 401
    except ValueError as val_err:
        return jsonify({"status": "error", "message": str(val_err)}), 400
    except Exception as exc:
        logger.exception("Error importing Google Meet recording")
        return jsonify({"status": "error", "message": f"Google Meet import failed: {exc}"}), 500

@app.route("/meetings/<meeting_id>/export/pdf", methods=["GET"])
@app.route("/api/meetings/<meeting_id>/export/pdf", methods=["GET"])
@require_authentication
def export_meeting_pdf_endpoint(meeting_id: str):
    """Generate and download PDF executive report for a specific meeting."""
    try:
        user_id = getattr(g, "user_id", None)
        meeting_data = get_meeting(meeting_id, user_id=user_id)
        if not meeting_data:
            if get_meeting(meeting_id):
                return jsonify({"status": "error", "message": f"Access denied. Meeting '{meeting_id}' belongs to another user."}), 403
            return jsonify({"status": "error", "message": f"Meeting '{meeting_id}' not found."}), 404
        
        pdf_bytes = generate_meeting_pdf_report(meeting_data)
        safe_title = "".join(c for c in meeting_data.get("title", "meeting") if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
        filename = f"report_{meeting_id}_{safe_title}.pdf"

        return Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as exc:
        logger.exception(f"Error generating PDF report for meeting '{meeting_id}'")
        return jsonify({"status": "error", "message": f"PDF report generation failed: {exc}"}), 500


@app.route("/meetings/<meeting_id>/export/csv", methods=["GET"])
@app.route("/api/meetings/<meeting_id>/export/csv", methods=["GET"])
@require_authentication
def export_meeting_csv_endpoint(meeting_id: str):
    """Generate and download CSV report for a specific meeting."""
    try:
        user_id = getattr(g, "user_id", None)
        meeting_data = get_meeting(meeting_id, user_id=user_id)
        if not meeting_data:
            if get_meeting(meeting_id):
                return jsonify({"status": "error", "message": f"Access denied. Meeting '{meeting_id}' belongs to another user."}), 403
            return jsonify({"status": "error", "message": f"Meeting '{meeting_id}' not found."}), 404
        
        csv_str = generate_meeting_csv_report(meeting_data)
        safe_title = "".join(c for c in meeting_data.get("title", "meeting") if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
        filename = f"report_{meeting_id}_{safe_title}.csv"

        return Response(
            csv_str,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as exc:
        logger.exception(f"Error generating CSV report for meeting '{meeting_id}'")
        return jsonify({"status": "error", "message": f"CSV report generation failed: {exc}"}), 500


# ── Milestone 3 Meeting Knowledge Repository Routes ────────────────────────

@app.route("/meetings/knowledge", methods=["GET"])
@app.route("/api/meetings/knowledge", methods=["GET"])
@require_authentication
def get_all_meetings_knowledge_endpoint():
    """Retrieve full knowledge objects for all historical meetings."""
    try:
        user_id = getattr(g, "user_id", None)
        data = get_all_meetings_knowledge(user_id=user_id)
        return jsonify({"status": "ok", "meetings": data}), 200
    except Exception as exc:
        logger.exception("Error listing historical meeting knowledge")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/meetings/<meeting_id>/transcript", methods=["GET"])
@app.route("/api/meetings/<meeting_id>/transcript", methods=["GET"])
@require_authentication
def get_transcript_endpoint(meeting_id: str):
    """Retrieve transcript for a specific meeting."""
    try:
        user_id = getattr(g, "user_id", None)
        mtg = get_meeting(meeting_id, user_id=user_id)
        if not mtg:
            if get_meeting(meeting_id):
                return jsonify({"status": "error", "message": f"Access denied. Meeting '{meeting_id}' belongs to another user."}), 403
            return jsonify({"status": "error", "message": f"Meeting '{meeting_id}' not found."}), 404
        data = get_meeting_transcript(meeting_id)
        data["status"] = "ok"
        return jsonify(data), 200
    except Exception as exc:
        logger.exception(f"Error fetching transcript for meeting '{meeting_id}'")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/meetings/<meeting_id>/summary", methods=["GET"])
@app.route("/api/meetings/<meeting_id>/summary", methods=["GET"])
@require_authentication
def get_summary_endpoint(meeting_id: str):
    """Retrieve summary for a specific meeting."""
    try:
        user_id = getattr(g, "user_id", None)
        mtg = get_meeting(meeting_id, user_id=user_id)
        if not mtg:
            if get_meeting(meeting_id):
                return jsonify({"status": "error", "message": f"Access denied. Meeting '{meeting_id}' belongs to another user."}), 403
            return jsonify({"status": "error", "message": f"Meeting '{meeting_id}' not found."}), 404
        data = get_meeting_summary(meeting_id)
        data["status"] = "ok"
        return jsonify(data), 200
    except Exception as exc:
        logger.exception(f"Error fetching summary for meeting '{meeting_id}'")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/meetings/<meeting_id>/decisions", methods=["GET"])
@app.route("/api/meetings/<meeting_id>/decisions", methods=["GET"])
@require_authentication
def get_decisions_endpoint(meeting_id: str):
    """Retrieve decisions for a specific meeting."""
    try:
        user_id = getattr(g, "user_id", None)
        mtg = get_meeting(meeting_id, user_id=user_id)
        if not mtg:
            if get_meeting(meeting_id):
                return jsonify({"status": "error", "message": f"Access denied. Meeting '{meeting_id}' belongs to another user."}), 403
            return jsonify({"status": "error", "message": f"Meeting '{meeting_id}' not found."}), 404
        data = get_meeting_decisions(meeting_id)
        return jsonify({"status": "ok", "meeting_id": meeting_id, "decisions": data}), 200
    except Exception as exc:
        logger.exception(f"Error fetching decisions for meeting '{meeting_id}'")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/meetings/<meeting_id>/action-items", methods=["GET"])
@app.route("/api/meetings/<meeting_id>/action-items", methods=["GET"])
@require_authentication
def get_action_items_endpoint(meeting_id: str):
    """Retrieve action items for a specific meeting."""
    try:
        user_id = getattr(g, "user_id", None)
        mtg = get_meeting(meeting_id, user_id=user_id)
        if not mtg:
            if get_meeting(meeting_id):
                return jsonify({"status": "error", "message": f"Access denied. Meeting '{meeting_id}' belongs to another user."}), 403
            return jsonify({"status": "error", "message": f"Meeting '{meeting_id}' not found."}), 404
        data = get_meeting_action_items(meeting_id)
        return jsonify({"status": "ok", "meeting_id": meeting_id, "action_items": data}), 200
    except Exception as exc:
        logger.exception(f"Error fetching action items for meeting '{meeting_id}'")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/meetings/<meeting_id>/participants", methods=["GET"])
@app.route("/api/meetings/<meeting_id>/participants", methods=["GET"])
@require_authentication
def get_participants_endpoint(meeting_id: str):
    """Retrieve participants for a specific meeting."""
    try:
        user_id = getattr(g, "user_id", None)
        mtg = get_meeting(meeting_id, user_id=user_id)
        if not mtg:
            if get_meeting(meeting_id):
                return jsonify({"status": "error", "message": f"Access denied. Meeting '{meeting_id}' belongs to another user."}), 403
            return jsonify({"status": "error", "message": f"Meeting '{meeting_id}' not found."}), 404
        data = get_meeting_participants(meeting_id)
        return jsonify({"status": "ok", "meeting_id": meeting_id, "participants": data}), 200
    except Exception as exc:
        logger.exception(f"Error fetching participants for meeting '{meeting_id}'")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/meetings/<meeting_id>/deadlines", methods=["GET"])
@app.route("/api/meetings/<meeting_id>/deadlines", methods=["GET"])
@require_authentication
def get_deadlines_endpoint(meeting_id: str):
    """Retrieve deadlines for a specific meeting."""
    try:
        user_id = getattr(g, "user_id", None)
        mtg = get_meeting(meeting_id, user_id=user_id)
        if not mtg:
            if get_meeting(meeting_id):
                return jsonify({"status": "error", "message": f"Access denied. Meeting '{meeting_id}' belongs to another user."}), 403
            return jsonify({"status": "error", "message": f"Meeting '{meeting_id}' not found."}), 404
        data = get_meeting_deadlines(meeting_id)
        return jsonify({"status": "ok", "meeting_id": meeting_id, "deadlines": data}), 200
    except Exception as exc:
        logger.exception(f"Error fetching deadlines for meeting '{meeting_id}'")
        return jsonify({"status": "error", "message": str(exc)}), 500


# ── Milestone 3 Task 4 Semantic Search Routes ──────────────────────────────

semantic_search_service = SemanticSearchService()


@app.route("/search", methods=["GET", "POST"])
@app.route("/api/search", methods=["GET", "POST"])
@app.route("/search/semantic", methods=["GET", "POST"])
@app.route("/api/search/semantic", methods=["GET", "POST"])
@app.route("/meetings/search/semantic", methods=["GET", "POST"])
@require_authentication
def semantic_search_endpoint():
    """
    Natural Language Semantic Search over Historical Meetings:
    Accepts JSON body or query params:
      - query / q: str
      - top_k: int (default 5)
      - content_type: str (optional)
      - meeting_id: str (optional)
      - deduplicate: bool (default false)
    """
    try:
        user_id = getattr(g, "user_id", None)
        if request.method == "POST":
            data = request.get_json(silent=True) or request.form.to_dict() or {}
            query = data.get("query") or data.get("q") or ""
            try:
                top_k = int(data.get("top_k", 5))
            except (ValueError, TypeError):
                return jsonify({"status": "error", "message": "Invalid top_k parameter. Must be an integer."}), 400
            content_type = data.get("content_type")
            meeting_id = data.get("meeting_id")
            start_date = data.get("start_date") or data.get("from_date")
            end_date = data.get("end_date") or data.get("to_date")
            min_score = float(data["min_score"]) if data.get("min_score") is not None else None
            deduplicate = str(data.get("deduplicate", "false")).lower() in ("true", "1")
        else:
            query = request.args.get("query") or request.args.get("q") or ""
            try:
                top_k = int(request.args.get("top_k", 5))
            except (ValueError, TypeError):
                return jsonify({"status": "error", "message": "Invalid top_k parameter. Must be an integer."}), 400
            content_type = request.args.get("content_type")
            meeting_id = request.args.get("meeting_id")
            start_date = request.args.get("start_date") or request.args.get("from_date")
            end_date = request.args.get("end_date") or request.args.get("to_date")
            min_score = float(request.args["min_score"]) if request.args.get("min_score") is not None else None
            deduplicate = request.args.get("deduplicate", "false").lower() in ("true", "1")

        query_clean = str(query).strip() if query else ""

        # Log Search Request (without credentials)
        logger.info(
            f"[SEARCH REQUEST] Method: {request.method} | Path: {request.path} | "
            f"Query: '{query_clean[:100]}' | top_k: {top_k} | content_type: {content_type} | meeting_id: {meeting_id}"
        )

        if not query_clean:
            logger.info("[SEARCH COMPLETED] Empty search query provided. Returning empty result set.")
            return jsonify({
                "status": "ok",
                "query": "",
                "latency_ms": 0.0,
                "total_results": 0,
                "results": []
            }), 200

        res = semantic_search_service.search(
            query=query_clean,
            top_k=top_k,
            meeting_id=meeting_id,
            content_type=content_type,
            start_date=start_date,
            end_date=end_date,
            min_score=min_score,
            deduplicate=deduplicate,
            db_path=os.getenv("DATABASE_PATH"),
            user_id=user_id
        )

        # Log Search Completion and Latency
        logger.info(
            f"[SEARCH COMPLETED] Query: '{query_clean[:100]}' | "
            f"Results Count: {res.get('total_results', 0)} | Latency: {res.get('latency_ms', 0.0)}ms"
        )
        return jsonify(res), 200

    except Exception as exc:
        logger.exception("[SEARCH FAILURE] Error executing semantic search")
        return jsonify({"status": "error", "message": f"Search failed: {exc}"}), 500


# ── Milestone 3 Task 5 RAG Question Answering Routes ───────────────────────

rag_service = RAGService(semantic_search_service=semantic_search_service)


@app.route("/ask", methods=["GET", "POST"])
@app.route("/api/ask", methods=["GET", "POST"])
@app.route("/search/rag", methods=["GET", "POST"])
@app.route("/api/search/rag", methods=["GET", "POST"])
@app.route("/meetings/search/rag", methods=["GET", "POST"])
@app.route("/api/meetings/qa", methods=["GET", "POST"])
@require_authentication
def rag_qa_endpoint():
    """
    Retrieval-Augmented Generation (RAG) Grounded Question Answering:
    Accepts JSON body or query params:
      - question / q / query: str
      - top_k: int (default 5)
      - content_type: str (optional)
      - meeting_id: str (optional)
    """
    try:
        user_id = getattr(g, "user_id", None)
        if request.method == "POST":
            data = request.get_json(silent=True) or request.form.to_dict() or {}
            question = data.get("question") or data.get("q") or data.get("query") or ""
            try:
                top_k = int(data.get("top_k", 5))
            except (ValueError, TypeError):
                return jsonify({"status": "error", "message": "Invalid top_k parameter. Must be an integer."}), 400
            content_type = data.get("content_type")
            meeting_id = data.get("meeting_id")
            start_date = data.get("start_date") or data.get("from_date")
            end_date = data.get("end_date") or data.get("to_date")
        else:
            question = request.args.get("question") or request.args.get("q") or request.args.get("query") or ""
            try:
                top_k = int(request.args.get("top_k", 5))
            except (ValueError, TypeError):
                return jsonify({"status": "error", "message": "Invalid top_k parameter. Must be an integer."}), 400
            content_type = request.args.get("content_type")
            meeting_id = request.args.get("meeting_id")
            start_date = request.args.get("start_date") or request.args.get("from_date")
            end_date = request.args.get("end_date") or request.args.get("to_date")

        q_clean = str(question).strip() if question else ""

        # Log RAG Request (without credentials)
        logger.info(
            f"[RAG REQUEST] Method: {request.method} | Path: {request.path} | "
            f"Question: '{q_clean[:100]}' | top_k: {top_k} | content_type: {content_type} | meeting_id: {meeting_id}"
        )

        if not q_clean:
            logger.info("[RAG COMPLETED] Empty question prompt provided. Returning default fallback answer.")
            return jsonify({
                "status": "ok",
                "question": "",
                "answer": "I couldn't find enough information in the available meeting records to answer this question.",
                "sources": [],
                "latency_ms": 0.0,
                "context_chunks_used": 0
            }), 200

        res = rag_service.answer_question(
            question=q_clean,
            top_k=top_k,
            meeting_id=meeting_id,
            content_type=content_type,
            start_date=start_date,
            end_date=end_date,
            db_path=os.getenv("DATABASE_PATH"),
            user_id=user_id
        )

        # Log RAG Completion and Latency
        logger.info(
            f"[RAG COMPLETED] Question: '{q_clean[:100]}' | "
            f"Context Chunks Used: {res.get('context_chunks_used', 0)} | Latency: {res.get('latency_ms', 0.0)}ms"
        )
        return jsonify(res), 200

    except Exception as exc:
        logger.exception("[RAG FAILURE] Error executing RAG question answering")
        return jsonify({"status": "error", "message": f"RAG Q&A failed: {exc}"}), 500


# ── Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", os.getenv("BACKEND_PORT", 5000)))
    print("\n" + "="*60)
    print("  AI-Powered Career Intelligence Platform Backend API")
    print(f"  Internal API Server running on port {port}")
    print("="*60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False)

