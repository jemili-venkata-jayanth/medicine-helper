"""
speech_to_text.py
------------------
Converts a spoken audio query into text using OpenAI's Whisper model,
running locally (no API key/internet needed once the model is downloaded).

Whisper handles accented English reasonably well and also supports
transcription in Hindi/Telugu directly if you pass the right language hint.
"""

import whisper

# "base" is a good speed/accuracy tradeoff for a project demo.
# Options (smallest to largest, accuracy improves, speed drops):
# tiny, base, small, medium, large
MODEL_SIZE = "base"

_model = None


def get_model():
    global _model
    if _model is None:
        print("Loading Whisper speech-to-text model (first run downloads it)...")
        _model = whisper.load_model(MODEL_SIZE)
    return _model


def transcribe_audio(audio_path: str, language_hint: str = None):
    """
    Transcribes the given audio file to text.
    language_hint: e.g. "en", "hi", "te" - if known, improves accuracy.
                   If None, Whisper auto-detects the language.
    """
    model = get_model()
    options = {}
    if language_hint:
        options["language"] = language_hint

    result = model.transcribe(audio_path, **options)
    return {
        "text": result.get("text", "").strip(),
        "detected_language": result.get("language"),
    }
