"""
text_to_speech.py
-------------------
Converts a text response into spoken audio using gTTS (Google Text-to-Speech).

gTTS needs internet access (it calls Google's TTS service) but supports
many Indian languages out of the box with natural-sounding voices, which
makes it the simplest option for an MVP. If you need fully offline TTS
later, Coqui TTS is the drop-in replacement to explore.
"""

import os
import uuid

from gtts import gTTS

# Maps our internal language codes to gTTS language codes.
# (For most Indian languages these already match.)
LANG_CODE_MAP = {
    "en": "en",
    "hi": "hi",
    "te": "te",
}

AUDIO_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated_audio")
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)


def synthesize_speech(text: str, lang: str = "en") -> str:
    """
    Converts text to speech and saves it as an mp3 file.
    Returns the file path of the generated audio.
    """
    gtts_lang = LANG_CODE_MAP.get(lang, "en")

    tts = gTTS(text=text, lang=gtts_lang)

    filename = f"{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(AUDIO_OUTPUT_DIR, filename)
    tts.save(filepath)

    return filepath


def build_speakable_text(medicine_response: dict) -> str:
    """
    Turns a structured medicine response (uses, warnings, etc.) into a
    natural spoken sentence, in the same language as the response data,
    rather than reading out a raw list.
    """
    name = medicine_response["name"]
    uses = ", ".join(medicine_response["uses"])
    warnings = ". ".join(medicine_response["warnings"])
    lang = medicine_response.get("language", "en")

    templates = {
        "en": (
            "{name} is commonly used for {uses}. "
            "Please note: {warnings}. "
            "This is general information only - please consult a doctor or "
            "pharmacist before use."
        ),
        "hi": (
            "{name} आमतौर पर {uses} के लिए उपयोग की जाती है। "
            "कृपया ध्यान दें: {warnings}। "
            "यह केवल सामान्य जानकारी है - कृपया उपयोग करने से पहले डॉक्टर या "
            "फार्मासिस्ट से सलाह लें।"
        ),
        "te": (
            "{name} సాధారణంగా {uses} కోసం ఉపయోగించబడుతుంది। "
            "దయచేసి గమనించండి: {warnings}। "
            "ఇది సాధారణ సమాచారం మాత్రమే - దయచేసి వాడకముందు వైద్యుడిని లేదా "
            "ఫార్మసిస్ట్‌ను సంప్రదించండి।"
        ),
    }

    template = templates.get(lang, templates["en"])
    return template.format(name=name, uses=uses, warnings=warnings)
