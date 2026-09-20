"""
main.py
--------
Medicine Helper - Backend API (ML/AI pipeline)

Endpoints:
  GET  /                    - health check
  GET  /medicines           - list all medicines (for debugging/autocomplete)
  GET  /lookup-medicine     - semantic search by text query
  POST /ocr-medicine        - upload a photo of a medicine strip/box -> lookup
  POST /voice-query         - upload spoken audio -> transcribe -> lookup
  POST /speak               - given text + language, returns spoken audio (mp3)

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import json
import os
import shutil
import tempfile

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from semantic_search import MedicineSearchEngine
from ocr_reader import extract_text_lines
from speech_to_text import transcribe_audio
from text_to_speech import synthesize_speech, build_speakable_text

app = FastAPI(title="Medicine Helper API (ML/AI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup - the embedding model + FAISS index stay in memory
# for the life of the server, so requests are fast.
print("Starting up: building semantic search index...")
search_engine = MedicineSearchEngine()
print("Ready.")


def build_response(med: dict, lang: str, score: float) -> dict:
    base = {
        "id": med["id"],
        "name": med["name"],
        "composition": med["composition"],
        "uses": med["uses"],
        "dosage_info": med["dosage_info"],
        "side_effects": med["side_effects"],
        "warnings": med["warnings"],
        "language": "en",
        "match_confidence": round(score, 2),
    }

    if lang != "en" and lang in med.get("translations", {}):
        translated = med["translations"][lang]
        base["uses"] = translated.get("uses", base["uses"])
        base["warnings"] = translated.get("warnings", base["warnings"])
        base["language"] = lang

    return base


@app.get("/")
def root():
    return {"status": "ok", "message": "Medicine Helper ML API is running"}


@app.get("/medicines")
def list_medicines():
    return [{"id": m["id"], "name": m["name"]} for m in search_engine.medicines]


@app.get("/lookup-medicine")
def lookup_medicine(
    name: str = Query(..., description="Medicine name or description to search for"),
    lang: str = Query("en", description="Language code: en, hi, te"),
):
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Please provide a medicine name")

    med, score = search_engine.best_match(name)

    if med is None:
        raise HTTPException(
            status_code=404,
            detail=f"No confident match found for '{name}'. Try rephrasing or check spelling.",
        )

    return build_response(med, lang, score)


@app.post("/ocr-medicine")
async def ocr_medicine(
    image: UploadFile = File(...),
    lang: str = Form("en"),
):
    """
    Accepts a photo of a medicine strip/box, runs OCR to extract text,
    then runs each detected text line through semantic search and
    returns the best overall match.
    """
    suffix = os.path.splitext(image.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    try:
        detected_lines = extract_text_lines(tmp_path)
        if not detected_lines:
            raise HTTPException(
                status_code=404,
                detail="Could not read any text from the image. Try a clearer, well-lit photo.",
            )

        # Try each detected text line against semantic search, keep the best match overall.
        best_med, best_score = None, 0.0
        for text, ocr_confidence in detected_lines:
            med, score = search_engine.best_match(text)
            if med is not None and score > best_score:
                best_med, best_score = med, score

        if best_med is None:
            raise HTTPException(
                status_code=404,
                detail=f"Read text from image ({[t for t, _ in detected_lines[:3]]}) "
                       f"but could not match it to a known medicine.",
            )

        response = build_response(best_med, lang, best_score)
        response["ocr_detected_text"] = [t for t, _ in detected_lines[:5]]
        return response

    finally:
        os.remove(tmp_path)


@app.post("/voice-query")
async def voice_query(
    audio: UploadFile = File(...),
    lang: str = Form("en"),
    speech_language_hint: str = Form(None),
):
    """
    Accepts a spoken audio query, transcribes it with Whisper, then
    runs the transcribed text through semantic search.
    """
    suffix = os.path.splitext(audio.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name

    try:
        transcription = transcribe_audio(tmp_path, language_hint=speech_language_hint)
        query_text = transcription["text"]

        if not query_text:
            raise HTTPException(
                status_code=404,
                detail="Could not understand the audio. Please try again clearly.",
            )

        med, score = search_engine.best_match(query_text)
        if med is None:
            raise HTTPException(
                status_code=404,
                detail=f"Heard '{query_text}' but could not match it to a known medicine.",
            )

        response = build_response(med, lang, score)
        response["transcribed_text"] = query_text
        response["detected_speech_language"] = transcription["detected_language"]
        return response

    finally:
        os.remove(tmp_path)


@app.post("/speak")
async def speak(
    text: str = Form(None),
    medicine_json: str = Form(None),
    lang: str = Form("en"),
):
    """
    Converts text to speech and returns an mp3 file.
    Either pass raw `text`, OR pass `medicine_json` (a JSON string of a
    medicine response from /lookup-medicine) and it will be turned into
    a natural spoken sentence automatically.
    """
    if medicine_json:
        try:
            medicine_response = json.loads(medicine_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid medicine_json")
        speakable_text = build_speakable_text(medicine_response)
    elif text:
        speakable_text = text
    else:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'medicine_json'")

    audio_path = synthesize_speech(speakable_text, lang=lang)
    return FileResponse(audio_path, media_type="audio/mpeg", filename="response.mp3")
