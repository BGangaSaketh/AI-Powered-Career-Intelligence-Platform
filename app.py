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

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── Internal modules ──────────────────────────────────────────────────────
from modules.ingestion      import ingest_text, ingest_txt_file, ingest_csv_file
from modules.preprocessing  import preprocess
from modules.sentiment      import analyze_sentiment
from modules.reporting      import generate_report
from modules.transcription  import transcribe_file

# Milestone 2 Modules
from modules.meeting_service import process_meeting_input
from modules.database import get_meeting, list_meetings, init_db

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


# ── Helpers ───────────────────────────────────────────────────────────────

def _allowed_file(filename: str, allowed_set: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


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
        raw_text = request.form.get("text", "").strip()
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
        result = process_meeting_input(
            file_path=save_path,
            raw_transcript_input=raw_transcript,
            title=title
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
def get_meeting_endpoint(meeting_id: str):
    """Retrieve processed meeting intelligence by meeting ID."""
    try:
        meeting_data = get_meeting(meeting_id)
        if not meeting_data:
            return jsonify({"status": "error", "message": f"Meeting '{meeting_id}' not found."}), 404
        meeting_data["status"] = "ok"
        return jsonify(meeting_data), 200
    except Exception as exc:
        logger.exception(f"Error fetching meeting '{meeting_id}'")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/meetings", methods=["GET"])
@app.route("/api/meetings", methods=["GET"])
def list_meetings_endpoint():
    """List all processed meetings."""
    try:
        meetings = list_meetings()
        return jsonify({"status": "ok", "meetings": meetings}), 200
    except Exception as exc:
        logger.exception("Error listing meetings")
        return jsonify({"status": "error", "message": str(exc)}), 500


# ── Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AI-Powered Career Intelligence Platform")
    print("  Milestone 1 & Milestone 2 — Meeting Intelligence System")
    print("  Server running at http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
