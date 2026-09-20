"""
desktop_app.py
---------------
Medicine Helper - Native Windows Desktop App (PySide6)

A single-window desktop application that directly uses the same ML
modules as the API version (semantic search, OCR, speech-to-text,
text-to-speech) - no separate server needed, everything runs in one
process.

Run with:
    python desktop_app.py
"""

import sys
import os
import threading

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QTextEdit, QComboBox, QFileDialog,
    QMessageBox, QProgressBar, QFrame
)
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QFont

from semantic_search import MedicineSearchEngine
from ocr_reader import extract_text_lines
from speech_to_text import transcribe_audio
from text_to_speech import synthesize_speech, build_speakable_text

from playsound import playsound  # for playing back generated speech audio


LANGUAGES = {
    "English": "en",
    "Hindi (हिंदी)": "hi",
    "Telugu (తెలుగు)": "te",
}


class WorkerSignals(QObject):
    """Used to safely send results from background threads back to the GUI."""
    finished = Signal(dict)
    error = Signal(str)
    status = Signal(str)


class MedicineHelperWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medicine Helper")
        self.setMinimumSize(560, 640)

        self.signals = WorkerSignals()
        self.signals.finished.connect(self._on_result_ready)
        self.signals.error.connect(self._on_error)
        self.signals.status.connect(self._on_status)

        self.current_result = None

        self._build_ui()
        self._load_engine_in_background()

    # ---------------- UI construction ----------------

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f4f6f8;
            }
            #titleLabel {
                color: #1c2b33;
            }
            #subtitleLabel {
                color: #6b7c85;
                font-size: 12px;
            }
            #langLabel {
                color: #4a5a63;
                font-size: 12px;
            }
            #statusLabel {
                color: #0d8f6f;
                font-style: italic;
                font-size: 12px;
            }
            QLineEdit {
                background-color: #ffffff;
                border: 1px solid #d5dde0;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 13px;
                color: #1c2b33;
            }
            QLineEdit:focus {
                border: 1px solid #14a085;
            }
            QComboBox {
                background-color: #ffffff;
                border: 1px solid #d5dde0;
                border-radius: 8px;
                padding: 6px 10px;
                color: #1c2b33;
            }
            QComboBox::drop-down {
                border: none;
                width: 22px;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #1c2b33;
                border: 1px solid #d5dde0;
                selection-background-color: #14a085;
                selection-color: #ffffff;
                outline: none;
                padding: 4px;
            }
            #primaryButton {
                background-color: #14a085;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 6px 22px;
                font-weight: 600;
                font-size: 13px;
            }
            #primaryButton:hover {
                background-color: #118a72;
            }
            #primaryButton:pressed {
                background-color: #0d715d;
            }
            #secondaryButton {
                background-color: #ffffff;
                color: #1c2b33;
                border: 1px solid #d5dde0;
                border-radius: 8px;
                padding: 6px 14px;
                font-size: 13px;
            }
            #secondaryButton:hover {
                background-color: #eef2f3;
                border: 1px solid #14a085;
            }
            #speakButton {
                background-color: #1c2b33;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                font-size: 13px;
            }
            #speakButton:hover:enabled {
                background-color: #101c22;
            }
            #speakButton:disabled {
                background-color: #cdd4d7;
                color: #8a9499;
            }
            #resultCard {
                background-color: #ffffff;
                border: 1px solid #e1e7e9;
                border-radius: 10px;
                padding: 14px;
                font-size: 13px;
                color: #1c2b33;
            }
            #disclaimerLabel {
                color: #9a6b00;
                background-color: #fff7e6;
                border: 1px solid #f0dfa8;
                border-radius: 6px;
                padding: 8px 10px;
                font-size: 11px;
            }
            #progressBar {
                border: none;
                border-radius: 2px;
                background-color: #e1e7e9;
            }
            #progressBar::chunk {
                background-color: #14a085;
                border-radius: 2px;
            }
            QMessageBox {
                background-color: #ffffff;
            }
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 22, 24, 20)

        self._apply_theme()

        # Title + language selector
        header_row = QHBoxLayout()
        title = QLabel("Medicine Helper")
        title.setObjectName("titleLabel")
        title.setFont(QFont("Segoe UI", 20, QFont.DemiBold))
        header_row.addWidget(title)
        header_row.addStretch()

        lang_label = QLabel("Language")
        lang_label.setObjectName("langLabel")
        header_row.addWidget(lang_label)

        self.language_dropdown = QComboBox()
        self.language_dropdown.addItems(LANGUAGES.keys())
        self.language_dropdown.setMinimumWidth(150)
        header_row.addWidget(self.language_dropdown)
        layout.addLayout(header_row)

        subtitle = QLabel("Know what your medicine is for, in your own language.")
        subtitle.setObjectName("subtitleLabel")
        layout.addWidget(subtitle)

        # Status label (shows model loading progress etc.)
        self.status_label = QLabel("Starting up...")
        self.status_label.setObjectName("statusLabel")
        layout.addWidget(self.status_label)

        # Search bar row
        search_row = QHBoxLayout()
        search_row.setSpacing(8)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Type a medicine name (e.g. Dolo 650)")
        self.search_input.setMinimumHeight(40)
        self.search_input.returnPressed.connect(self._on_text_search)
        search_row.addWidget(self.search_input)

        search_btn = QPushButton("Search")
        search_btn.setObjectName("primaryButton")
        search_btn.setMinimumHeight(40)
        search_btn.setCursor(Qt.PointingHandCursor)
        search_btn.clicked.connect(self._on_text_search)
        search_row.addWidget(search_btn)
        layout.addLayout(search_row)

        # Action buttons row
        actions_row = QHBoxLayout()
        actions_row.setSpacing(8)

        scan_btn = QPushButton("Scan Strip Photo")
        scan_btn.setObjectName("secondaryButton")
        scan_btn.setMinimumHeight(38)
        scan_btn.setCursor(Qt.PointingHandCursor)
        scan_btn.clicked.connect(self._on_scan_image)
        actions_row.addWidget(scan_btn)

        voice_btn = QPushButton("Upload Voice Query")
        voice_btn.setObjectName("secondaryButton")
        voice_btn.setMinimumHeight(38)
        voice_btn.setCursor(Qt.PointingHandCursor)
        voice_btn.clicked.connect(self._on_voice_query)
        actions_row.addWidget(voice_btn)

        layout.addLayout(actions_row)

        # Result display
        self.result_display = QTextEdit()
        self.result_display.setObjectName("resultCard")
        self.result_display.setReadOnly(True)
        self.result_display.setPlaceholderText(
            "Search, scan, or upload a voice query to see medicine information here."
        )
        layout.addWidget(self.result_display, stretch=1)

        # Speak button
        self.speak_btn = QPushButton("Speak Result Aloud")
        self.speak_btn.setObjectName("speakButton")
        self.speak_btn.setMinimumHeight(42)
        self.speak_btn.setCursor(Qt.PointingHandCursor)
        self.speak_btn.clicked.connect(self._on_speak)
        self.speak_btn.setEnabled(False)
        layout.addWidget(self.speak_btn)

        # Progress bar (indeterminate, shown during background work)
        self.progress = QProgressBar()
        self.progress.setObjectName("progressBar")
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.hide()
        layout.addWidget(self.progress)

        # Disclaimer
        disclaimer = QLabel(
            "This provides general information only, not a prescription. "
            "Always consult a doctor or pharmacist before taking any medicine."
        )
        disclaimer.setObjectName("disclaimerLabel")
        disclaimer.setWordWrap(True)
        layout.addWidget(disclaimer)

    # ---------------- Engine loading ----------------

    def _load_engine_in_background(self):
        self._set_busy(True, "Loading semantic search model (first run may take a minute)...")

        def load():
            try:
                self.search_engine = MedicineSearchEngine()
                self.signals.status.emit("Ready. Search, scan, or speak a medicine name.")
            except Exception as e:
                self.signals.error.emit(f"Failed to load search engine: {e}")
            finally:
                self.signals.finished.emit({"_engine_loaded": True})

        threading.Thread(target=load, daemon=True).start()

    # ---------------- Helpers ----------------

    def _set_busy(self, busy: bool, status_text: str = ""):
        self.progress.setVisible(busy)
        if status_text:
            self.status_label.setText(status_text)

    def _current_lang_code(self):
        return LANGUAGES[self.language_dropdown.currentText()]

    def _build_response_dict(self, med: dict, score: float, lang: str) -> dict:
        base = {
            "id": med["id"],
            "name": med["name"],
            "composition": med["composition"],
            "uses": med["uses"],
            "dosage_info": med["dosage_info"],
            "side_effects": med["side_effects"],
            "warnings": med["warnings"],
            "match_confidence": round(score, 2),
            "language": "en",
        }
        if lang != "en" and lang in med.get("translations", {}):
            translated = med["translations"][lang]
            base["uses"] = translated.get("uses", base["uses"])
            base["warnings"] = translated.get("warnings", base["warnings"])
            base["language"] = lang
        return base

    def _display_result(self, result: dict):
        self.current_result = result
        text = (
            f"<h2 style='color:#14a085; margin-bottom:2px;'>{result['name']}</h2>"
            f"<p style='color:#8a9499; font-size:11px; margin-top:0;'>Match confidence: {result['match_confidence'] * 100:.0f}%</p>"
            f"<p><b style='color:#1c2b33;'>Used for</b><br>{'<br>'.join('• ' + u for u in result['uses'])}</p>"
            f"<p><b style='color:#1c2b33;'>Composition</b><br>{result['composition']}</p>"
            f"<p><b style='color:#1c2b33;'>General dosage info</b><br>{result['dosage_info']}</p>"
            f"<p><b style='color:#1c2b33;'>Possible side effects</b><br>{'<br>'.join('• ' + s for s in result['side_effects'])}</p>"
            f"<p><b style='color:#c0392b;'>Warnings</b><br>{'<br>'.join('• ' + w for w in result['warnings'])}</p>"
        )
        self.result_display.setHtml(text)
        self.speak_btn.setEnabled(True)

    # ---------------- Actions ----------------

    def _on_text_search(self):
        query = self.search_input.text().strip()
        if not query:
            return

        self._set_busy(True, f"Searching for '{query}'...")

        def work():
            try:
                med, score = self.search_engine.best_match(query)
                if med is None:
                    self.signals.error.emit(f"No confident match found for '{query}'.")
                    return
                result = self._build_response_dict(med, score, self._current_lang_code())
                self.signals.finished.emit(result)
            except Exception as e:
                self.signals.error.emit(str(e))

        threading.Thread(target=work, daemon=True).start()

    def _on_scan_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Medicine Strip/Box Photo", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not file_path:
            return

        self._set_busy(True, "Reading text from image (OCR)...")

        def work():
            try:
                detected_lines = extract_text_lines(file_path)
                if not detected_lines:
                    self.signals.error.emit("Could not read any text from the image.")
                    return

                best_med, best_score = None, 0.0
                for text, _conf in detected_lines:
                    med, score = self.search_engine.best_match(text)
                    if med is not None and score > best_score:
                        best_med, best_score = med, score

                if best_med is None:
                    detected_preview = ", ".join(t for t, _ in detected_lines[:3])
                    self.signals.error.emit(
                        f"Read text ({detected_preview}) but could not match a known medicine."
                    )
                    return

                result = self._build_response_dict(best_med, best_score, self._current_lang_code())
                self.signals.finished.emit(result)
            except Exception as e:
                self.signals.error.emit(str(e))

        threading.Thread(target=work, daemon=True).start()

    def _on_voice_query(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Voice Recording", "", "Audio (*.wav *.mp3 *.m4a)"
        )
        if not file_path:
            return

        self._set_busy(True, "Transcribing audio (this may take a moment)...")

        def work():
            try:
                transcription = transcribe_audio(file_path)
                query_text = transcription["text"]
                if not query_text:
                    self.signals.error.emit("Could not understand the audio.")
                    return

                med, score = self.search_engine.best_match(query_text)
                if med is None:
                    self.signals.error.emit(
                        f"Heard '{query_text}' but could not match a known medicine."
                    )
                    return

                result = self._build_response_dict(med, score, self._current_lang_code())
                result["_transcribed_text"] = query_text
                self.signals.finished.emit(result)
            except Exception as e:
                self.signals.error.emit(str(e))

        threading.Thread(target=work, daemon=True).start()

    def _on_speak(self):
        if not self.current_result:
            return

        self._set_busy(True, "Generating speech...")

        def work():
            try:
                spoken_text = build_speakable_text(self.current_result)
                lang = self._current_lang_code()
                audio_path = synthesize_speech(spoken_text, lang=lang)

                self.signals.status.emit("Speaking...")
                playsound(audio_path)

                self.signals.finished.emit({"_speech_done": True})
            except Exception as e:
                self.signals.error.emit(f"Could not generate speech: {e}")

        threading.Thread(target=work, daemon=True).start()

    # ---------------- Signal handlers (run on the main/GUI thread) ----------------

    def _on_result_ready(self, result: dict):
        self._set_busy(False)

        if result.get("_engine_loaded"):
            return
        if result.get("_speech_done"):
            return

        transcribed = result.pop("_transcribed_text", None)
        self._display_result(result)

        if transcribed:
            self.status_label.setText(f"Heard: \"{transcribed}\"")

    def _on_error(self, message: str):
        self._set_busy(False)
        self.status_label.setText("Ready.")
        QMessageBox.warning(self, "Not found", message)

    def _on_status(self, message: str):
        self.status_label.setText(message)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MedicineHelperWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
