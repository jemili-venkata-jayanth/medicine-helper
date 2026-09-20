"""
ocr_reader.py
-------------
Extracts text from a photo of a medicine strip or box using EasyOCR.

Medicine strip photos are messy (foil glare, small font, multiple
languages printed together), so this returns ALL detected text lines
with confidence scores rather than guessing a single "answer" - the
calling code then runs each line through semantic search and keeps
the best-scoring match.
"""

import easyocr

# Initialize once at import time - loading the model is slow, so we don't
# want to do it on every request. 'en' is enough for most Indian medicine
# packaging since brand names are printed in English/Latin script.
_reader = None


def get_reader():
    global _reader
    if _reader is None:
        print("Loading OCR model (first run downloads the model)...")
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def extract_text_lines(image_path: str, min_confidence: float = 0.4):
    """
    Returns a list of (text, confidence) tuples detected in the image,
    sorted by confidence descending. Filters out low-confidence noise.
    """
    reader = get_reader()
    results = reader.readtext(image_path)  # [(bbox, text, confidence), ...]

    lines = [(text.strip(), float(conf)) for (_, text, conf) in results
             if conf >= min_confidence and text.strip()]

    lines.sort(key=lambda x: x[1], reverse=True)
    return lines
