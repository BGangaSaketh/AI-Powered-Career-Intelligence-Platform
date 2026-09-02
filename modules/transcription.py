"""
modules/transcription.py
========================
Audio / Video Transcription Module

Supports:
  - WAV, MP3, FLAC, OGG audio files   → transcribed via SpeechRecognition
  - MP4, AVI, MOV, MKV video files    → audio extracted first via moviepy,
                                         then transcribed

The Google Web Speech API is used (free, requires internet).
Audio is split into chunks for better recognition of longer recordings.
"""

import os
import uuid
import logging

logger = logging.getLogger(__name__)

# We import lazily inside functions so that missing optional dependencies
# produce a clean error message rather than crashing the whole app.

# Supported file extensions
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Temporary directory for extracted audio files
_TEMP_DIR = os.path.join(os.path.dirname(__file__), "..", "temp_audio")


def _ensure_temp_dir():
    """Create the temporary audio directory if it doesn't exist."""
    os.makedirs(_TEMP_DIR, exist_ok=True)


def _get_extension(filename: str) -> str:
    """Return the lowercased file extension including the dot."""
    _, ext = os.path.splitext(filename)
    return ext.lower()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe_file(file_path: str) -> dict:
    """
    Transcribe speech from an audio or video file.

    Parameters
    ----------
    file_path : str
        Absolute path to the uploaded audio/video file.

    Returns
    -------
    dict
        {
            "status":     "ok" | "error",
            "transcript": str,
            "file_type":  "audio" | "video",
            "errors":     list[str]
        }
    """
    ext = _get_extension(file_path)

    if ext in VIDEO_EXTENSIONS:
        # Extract audio from video first
        audio_path, extract_errors = extract_audio_from_video(file_path)
        if extract_errors:
            return {
                "status":     "error",
                "transcript": "",
                "file_type":  "video",
                "errors":     extract_errors,
            }
        transcript, transcribe_errors = transcribe_audio(audio_path)
        # Clean up extracted audio
        _safe_remove(audio_path)
        return {
            "status":     "ok" if not transcribe_errors else "error",
            "transcript": transcript,
            "file_type":  "video",
            "errors":     transcribe_errors,
        }

    elif ext in AUDIO_EXTENSIONS:
        transcript, transcribe_errors = transcribe_audio(file_path)
        return {
            "status":     "ok" if not transcribe_errors else "error",
            "transcript": transcript,
            "file_type":  "audio",
            "errors":     transcribe_errors,
        }

    else:
        return {
            "status":     "error",
            "transcript": "",
            "file_type":  "unknown",
            "errors":     [
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(AUDIO_EXTENSIONS | VIDEO_EXTENSIONS))}"
            ],
        }


def extract_audio_from_video(video_path: str) -> tuple:
    """
    Extract audio track from a video file and save it as a 16 kHz mono WAV.

    Google Speech API requires 16 kHz mono PCM WAV.  We set these parameters
    directly during moviepy extraction to avoid a second conversion step.

    Parameters
    ----------
    video_path : str
        Path to the video file.

    Returns
    -------
    tuple
        (audio_path: str, errors: list[str])
        audio_path is empty string if extraction failed.
    """
    try:
        from moviepy import VideoFileClip
    except ImportError:
        return "", ["moviepy is not installed. Run: pip install moviepy"]

    _ensure_temp_dir()
    raw_path   = os.path.join(_TEMP_DIR, f"{uuid.uuid4().hex}_raw.wav")
    audio_path = os.path.join(_TEMP_DIR, f"{uuid.uuid4().hex}.wav")

    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            clip.close()
            return "", ["Video file has no audio track."]

        # Write raw audio from the video
        clip.audio.write_audiofile(raw_path, logger=None)
        clip.close()

        # Normalise to 16 kHz mono 16-bit PCM using pydub
        normalised, norm_errors = _normalise_audio(raw_path, audio_path)
        _safe_remove(raw_path)

        if norm_errors:
            return "", norm_errors
        return audio_path, []

    except Exception as exc:
        _safe_remove(raw_path)
        return "", [f"Failed to extract audio from video: {exc}"]


def transcribe_audio(audio_path: str) -> tuple:
    """
    Transcribe speech from an audio file using the Google Web Speech API.

    Steps:
      1. Convert / normalise audio to 16 kHz mono 16-bit WAV (required by
         Google Speech API) using pydub.
      2. Split into 30-second chunks so the free API does not reject large
         payloads.
      3. Concatenate chunk transcripts into a single string.

    Parameters
    ----------
    audio_path : str
        Path to the audio file (any format supported by pydub/ffmpeg).

    Returns
    -------
    tuple
        (transcript: str, errors: list[str])
    """
    try:
        import speech_recognition as sr
    except ImportError:
        return "", ["SpeechRecognition is not installed. Run: pip install SpeechRecognition"]

    try:
        from pydub import AudioSegment
    except ImportError:
        return "", ["pydub is not installed. Run: pip install pydub"]

    # ── Step 1: Normalise to 16 kHz mono 16-bit WAV ───────────────────────
    _ensure_temp_dir()
    normalised_path = os.path.join(_TEMP_DIR, f"{uuid.uuid4().hex}_norm.wav")

    _, norm_errors = _normalise_audio(audio_path, normalised_path)
    if norm_errors:
        return "", norm_errors

    # ── Step 2: Split into 30-second chunks ───────────────────────────────
    try:
        sound        = AudioSegment.from_wav(normalised_path)
        chunk_ms     = 30_000          # 30 seconds in milliseconds
        chunks       = [sound[i : i + chunk_ms]
                        for i in range(0, len(sound), chunk_ms)]
    except Exception as exc:
        _safe_remove(normalised_path)
        return "", [f"Failed to segment audio: {exc}"]

    # ── Step 3: Transcribe each chunk ─────────────────────────────────────
    recognizer    = sr.Recognizer()
    transcripts   = []
    errors        = []

    for idx, chunk in enumerate(chunks):
        chunk_path = os.path.join(_TEMP_DIR, f"{uuid.uuid4().hex}_chunk{idx}.wav")
        try:
            chunk.export(chunk_path, format="wav")

            with sr.AudioFile(chunk_path) as source:
                audio_data = recognizer.record(source)

            text = recognizer.recognize_google(audio_data)
            if text:
                transcripts.append(text)

        except sr.UnknownValueError:
            # Chunk was silent or unclear — skip it, do not fail the whole file
            pass
        except sr.RequestError as exc:
            errors.append(
                f"Google Speech API error on chunk {idx + 1}: {exc}. "
                "Check your internet connection."
            )
            break   # No point retrying remaining chunks
        except Exception as exc:
            errors.append(f"Error processing chunk {idx + 1}: {exc}")
        finally:
            _safe_remove(chunk_path)

    _safe_remove(normalised_path)

    if errors:
        return " ".join(transcripts), errors

    if not transcripts:
        return "", [
            "Speech Recognition could not understand any audio in the file. "
            "Please ensure the recording is clear, in English, and has minimal background noise."
        ]

    return " ".join(transcripts), []


def _normalise_audio(input_path: str, output_path: str) -> tuple:
    """
    Convert any audio file to 16 kHz, mono, 16-bit PCM WAV.

    Google Web Speech API only accepts this specific format.  Using any
    other sample rate or channel count causes a "Bad Request" error.

    Parameters
    ----------
    input_path  : str  Path to the source audio file.
    output_path : str  Path where the normalised WAV will be written.

    Returns
    -------
    tuple  (output_path: str, errors: list[str])
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        return "", ["pydub is not installed. Run: pip install pydub"]

    try:
        ext   = _get_extension(input_path).lstrip(".")
        fmt   = ext if ext else "wav"
        sound = AudioSegment.from_file(input_path, format=fmt)

        # Convert: mono, 16 kHz, 16-bit
        sound = sound.set_channels(1)        # mono
        sound = sound.set_frame_rate(16000)  # 16 kHz
        sound = sound.set_sample_width(2)    # 16-bit

        sound.export(output_path, format="wav")
        return output_path, []

    except Exception as exc:
        return "", [f"Audio normalisation failed: {exc}"]


def _safe_remove(path: str):
    """Delete a file without raising an exception if it doesn't exist."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
