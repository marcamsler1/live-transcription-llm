import os
import json
import time
import queue
import threading
import zipfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Callable, List, Optional, Dict, Any, Tuple
from dotenv import load_dotenv

import sounddevice as sd
import numpy as np
import torch
import torchaudio
import requests
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import soundfile as sf
import subprocess
from openai import OpenAI

load_dotenv()

# === KONFIGURATION ===
DEFAULT_CONFIG = {
    "openai_model": "gpt-4.1",
    "temperature": 0.5,
    "sample_rate": 16000,
    "input_sample_rate": 44100,
    "chunk_duration": 0.5,
    "silence_threshold": 1.0,
    "vad_threshold": 0.5,
    "min_audio_seconds": 0.35,
    "request_timeout": 120,
    "min_rms_threshold": 0.008,
    "normalize_peak_target": 0.92,
    "recent_entry_limit": 8,
    "summary_trigger_entries": 14,
    "summary_keep_recent": 6,
    "party_options": ["SP", "FDP", "SVP", "MITTE", "GLP", "GRUENE"],
    "piper_voice_path": "piper_voices/de_DE-thorsten-high.onnx",
    "piper_command": "piper",
    "audio_chunk_dir": "audio_chunks",
    "piper_output_dir": "piper_output",
    "session_dir": "sessions",
    "project_descriptions_dir": "project_descriptions",
    "config_path": "config.json"
}


def load_app_config() -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    config_path = config["config_path"]

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)
            if isinstance(file_config, dict):
                config.update(file_config)
        except Exception:
            pass
    else:
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    return config


CONFIG = load_app_config()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = CONFIG["openai_model"]
TEMPERATURE = CONFIG["temperature"]
SAMPLE_RATE = CONFIG["sample_rate"]
INPUT_SAMPLE_RATE = CONFIG["input_sample_rate"]
CHUNK_DURATION = CONFIG["chunk_duration"]
SILENCE_THRESHOLD = CONFIG["silence_threshold"]
VAD_THRESHOLD = CONFIG["vad_threshold"]
MIN_AUDIO_SECONDS = CONFIG["min_audio_seconds"]
REQUEST_TIMEOUT = CONFIG["request_timeout"]
MIN_RMS_THRESHOLD = CONFIG["min_rms_threshold"]
NORMALIZE_PEAK_TARGET = CONFIG["normalize_peak_target"]
RECENT_ENTRY_LIMIT = CONFIG["recent_entry_limit"]
SUMMARY_TRIGGER_ENTRIES = CONFIG["summary_trigger_entries"]
SUMMARY_KEEP_RECENT = CONFIG["summary_keep_recent"]
PIPER_VOICE_PATH = CONFIG["piper_voice_path"]
PIPER_COMMAND = CONFIG["piper_command"]
AUDIO_CHUNK_DIR = CONFIG["audio_chunk_dir"]
PIPER_OUTPUT_DIR = CONFIG["piper_output_dir"]
SESSION_DIR = CONFIG["session_dir"]
PROJECT_DESCRIPTIONS_DIR = CONFIG["project_descriptions_dir"]
WHISPER_TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions"

os.makedirs(AUDIO_CHUNK_DIR, exist_ok=True)
os.makedirs(PIPER_OUTPUT_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)
os.makedirs(PROJECT_DESCRIPTIONS_DIR, exist_ok=True)

vs_dict_json_string = os.getenv("VS_DICT_JSON", "{}")
VS_DICT = json.loads(vs_dict_json_string)


@dataclass
class ConversationEntry:
    timestamp: str
    text: str


class AppLogger:
    def __init__(self, log_callback: Optional[Callable[[str], None]] = None):
        self.log_callback = log_callback

    def info(self, message: str):
        print(f"[INFO] {message}")
        if self.log_callback:
            self.log_callback(message)

    def error(self, message: str):
        print(f"[FEHLER] {message}")
        if self.log_callback:
            self.log_callback(message)


class ConversationManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.entries: List[ConversationEntry] = []
        self.summary_memory: str = ""
        self.last_response: str = ""

    def add_entry(self, text: str):
        cleaned = text.strip()
        if not cleaned:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self.lock:
            self.entries.append(ConversationEntry(timestamp=timestamp, text=cleaned))

    def clear(self):
        with self.lock:
            self.entries = []
            self.summary_memory = ""
            self.last_response = ""

    def set_last_response(self, text: str):
        with self.lock:
            self.last_response = text

    def get_last_response(self) -> str:
        with self.lock:
            return self.last_response

    def get_entries(self) -> List[ConversationEntry]:
        with self.lock:
            return list(self.entries)

    def get_formatted_transcript(self) -> str:
        with self.lock:
            summary_part = ""
            if self.summary_memory.strip():
                summary_part = f"[ZUSAMMENFASSUNG FRÜHERER DISKUSSION]\n{self.summary_memory.strip()}\n\n"
            recent_part = "\n\n".join(f"[{entry.timestamp}] {entry.text}" for entry in self.entries)
            return f"{summary_part}{recent_part}".strip()

    def get_prompt_context(self) -> str:
        with self.lock:
            sections = []
            if self.summary_memory.strip():
                sections.append("Frühere Diskussion (Zusammenfassung):\n" + self.summary_memory.strip())
            recent_entries = self.entries[-RECENT_ENTRY_LIMIT:]
            if recent_entries:
                recent_text = "\n".join(f"- {entry.text}" for entry in recent_entries)
                sections.append("Letzte Diskussionsbeiträge:\n" + recent_text)
            return "\n\n".join(sections).strip()

    def needs_summarization(self) -> bool:
        with self.lock:
            return len(self.entries) >= SUMMARY_TRIGGER_ENTRIES

    def get_entries_for_summarization(self) -> Tuple[str, List[ConversationEntry], str]:
        with self.lock:
            if len(self.entries) < SUMMARY_TRIGGER_ENTRIES:
                return "", [], ""
            old_entries = self.entries[:-SUMMARY_KEEP_RECENT]
            summary_input = "\n".join(f"- {entry.text}" for entry in old_entries)
            return self.summary_memory, old_entries, summary_input

    def apply_summary(self, new_summary: str):
        with self.lock:
            self.summary_memory = new_summary.strip()
            self.entries = self.entries[-SUMMARY_KEEP_RECENT:]

    def serialize(self, selected_party: str) -> Dict[str, Any]:
        with self.lock:
            return {
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "selected_party": selected_party,
                "summary_memory": self.summary_memory,
                "last_response": self.last_response,
                "entries": [asdict(entry) for entry in self.entries],
            }


class SessionService:
    def __init__(self, session_dir: str, logger: AppLogger):
        self.session_dir = session_dir
        self.logger = logger

    def auto_save(self, session_data: Dict[str, Any]) -> str:
        filename = os.path.join(
            self.session_dir,
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        return filename

    def export_json(self, filepath: str, session_data: Dict[str, Any]):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

    def export_txt(self, filepath: str, session_data: Dict[str, Any]):
        lines = []
        lines.append("Diskussions-Assistent Export")
        lines.append(f"Gespeichert am: {session_data.get('saved_at', '')}")
        lines.append(f"Gewählte Partei: {session_data.get('selected_party', '')}")
        lines.append("")
        summary_memory = session_data.get("summary_memory", "").strip()
        if summary_memory:
            lines.append("Zusammenfassung früherer Diskussion:")
            lines.append(summary_memory)
            lines.append("")
        lines.append("Letzte Diskussionsbeiträge:")
        for entry in session_data.get("entries", []):
            lines.append(f"[{entry.get('timestamp', '')}] {entry.get('text', '')}")
        lines.append("")
        lines.append("Letzte generierte Antwort:")
        lines.append(session_data.get("last_response", ""))

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def export_pdf(self, filepath: str, session_data: Dict[str, Any]):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
        except ImportError as e:
            raise RuntimeError("Für PDF-Export wird reportlab benötigt.") from e

        c = canvas.Canvas(filepath, pagesize=A4)
        width, height = A4
        x = 50
        y = height - 50
        line_height = 14

        def write_line(text: str):
            nonlocal y
            chunks = [text[i:i+100] for i in range(0, len(text), 100)] or [""]
            for chunk in chunks:
                if y < 60:
                    c.showPage()
                    y = height - 50
                c.drawString(x, y, chunk)
                y -= line_height

        write_line("Diskussions-Assistent Export")
        write_line(f"Gespeichert am: {session_data.get('saved_at', '')}")
        write_line(f"Gewählte Partei: {session_data.get('selected_party', '')}")
        write_line("")

        summary_memory = session_data.get("summary_memory", "").strip()
        if summary_memory:
            write_line("Zusammenfassung früherer Diskussion:")
            for line in summary_memory.splitlines():
                write_line(line)
            write_line("")

        write_line("Letzte Diskussionsbeiträge:")
        for entry in session_data.get("entries", []):
            write_line(f"[{entry.get('timestamp', '')}] {entry.get('text', '')}")
        write_line("")
        write_line("Letzte generierte Antwort:")
        for line in session_data.get("last_response", "").splitlines():
            write_line(line)

        c.save()


class TranscriptionService:
    def __init__(self, api_key: str, logger: AppLogger):
        self.api_key = api_key
        self.logger = logger

    def transcribe_file(self, filename: str, language: str = "de") -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY ist nicht gesetzt.")

        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            with open(filename, "rb") as f:
                files = {
                    "file": (os.path.basename(filename), f),
                    "model": (None, "whisper-1"),
                    "language": (None, language),
                }
                response = requests.post(
                    WHISPER_TRANSCRIPTION_URL,
                    headers=headers,
                    files=files,
                    timeout=REQUEST_TIMEOUT,
                )
        except requests.Timeout as e:
            raise RuntimeError("Die Transkriptionsanfrage hat das Zeitlimit überschritten.") from e
        except requests.RequestException as e:
            raise RuntimeError(f"Netzwerkfehler bei der Transkription: {e}") from e
        except OSError as e:
            raise RuntimeError(f"Audiodatei konnte nicht gelesen werden: {e}") from e

        if response.status_code != 200:
            raise RuntimeError(f"Whisper API Fehler {response.status_code}: {response.text}")

        try:
            payload = response.json()
            return payload.get("text", "").strip()
        except ValueError as e:
            raise RuntimeError("Ungültige JSON-Antwort von der Transkriptions-API.") from e


class LLMService:
    def __init__(self, api_key: str, logger: AppLogger):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY ist nicht gesetzt.")
        self.client = OpenAI(api_key=api_key)
        self.logger = logger

    def build_prompt(self, context_text: str, partei: str, project_context: str = "") -> str:
        project_section = ""
        if project_context.strip():
            project_section = f"\n\nProjektinformationen:\n{project_context.strip()}"
        return (
            f"Sie sprechen als Fraktionschef der {partei} im Stadtparlament St.Gallen.\n\n"
            f"Ihre Aufgabe:\n"
            f"Reagieren Sie auf die laufende politische Diskussion mit einer kurzen, realistisch klingenden Stellungnahme aus Sicht Ihrer Partei.\n\n"
            f"Vorgaben:\n"
            f"- Schreiben Sie in klarem, sachlichem Hochdeutsch.\n"
            f"- Bleiben Sie glaubwürdig, kommunalpolitisch und lösungsorientiert.\n"
            f"- Greifen Sie 1 bis 2 konkrete Punkte aus der Diskussion auf.\n"
            f"- Formulieren Sie eine klare Position, eine kurze Begründung und wenn passend eine Forderung oder einen nächsten politischen Schritt.\n"
            f"- Verwenden Sie kurze, gut sprechbare Sätze.\n"
            f"- Bleiben Sie bei maximal 4 Sätzen.\n"
            f"- Verwenden Sie keine Floskeln und keine übertrieben aggressive Sprache.\n"
            f"- Erfinden Sie keine Fakten, Zahlen oder Beschlüsse.\n"
            f"- Wenn Informationen fehlen, formulieren Sie vorsichtig und allgemein.\n"
            f"- Orientieren Sie sich an typischen politischen Grundhaltungen der {partei}.\n\n"
            f"Kontext:\n{context_text.strip()}{project_section}"
        )

    def build_summary_prompt(self, existing_summary: str, text_to_summarize: str) -> str:
        existing = existing_summary.strip() or "Bisher keine Zusammenfassung vorhanden."
        return (
            "Sie verdichten eine laufende politische Diskussion für den späteren Gebrauch in einem Debattenassistenten.\n\n"
            "Aufgabe:\n"
            "Erstellen Sie eine knappe, sachliche Arbeitszusammenfassung der älteren Diskussion.\n\n"
            "Vorgaben:\n"
            "- Maximal 8 kurze Stichpunkte oder kurze Sätze.\n"
            "- Behalten Sie nur zentrale Themen, Konfliktlinien, Forderungen und offene Punkte.\n"
            "- Entfernen Sie Wiederholungen, Füllwörter und Nebensächlichkeiten.\n"
            "- Erfinden Sie keine Fakten.\n"
            "- Schreiben Sie auf Deutsch.\n\n"
            f"Bestehende Zusammenfassung:\n{existing}\n\n"
            f"Neu zu verdichtende ältere Diskussionsbeiträge:\n{text_to_summarize.strip()}"
        )

    def _build_tools(self, partei: str, vs_dict: Dict[str, str]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        if partei in vs_dict:
            return ([{
                "type": "file_search",
                "vector_store_ids": [vs_dict[partei]],
                "max_num_results": 20,
            }], "required")
        return ([], None)

    def query(self, context_text: str, partei: str, vs_dict: Dict[str, str], project_context: str = "") -> str:
        prompt = self.build_prompt(context_text, partei, project_context)
        tools, tool_choice = self._build_tools(partei, vs_dict)

        try:
            response = self.client.responses.create(
                model=OPENAI_MODEL,
                temperature=TEMPERATURE,
                tool_choice=tool_choice,
                tools=tools,
                input=prompt,
            )
        except Exception as e:
            raise RuntimeError(f"Fehler bei der LLM-Abfrage: {e}") from e

        try:
            reply = response.output_text.strip()
        except Exception as e:
            raise RuntimeError(f"LLM-Antwort konnte nicht gelesen werden: {e}") from e

        if not reply:
            raise RuntimeError("Das Modell hat keine Antwort zurückgegeben.")
        return reply

    def summarize(self, existing_summary: str, text_to_summarize: str) -> str:
        prompt = self.build_summary_prompt(existing_summary, text_to_summarize)
        try:
            response = self.client.responses.create(
                model=OPENAI_MODEL,
                temperature=0.2,
                input=prompt,
            )
            summary = response.output_text.strip()
        except Exception as e:
            raise RuntimeError(f"Fehler bei der Zusammenfassung: {e}") from e

        if not summary:
            raise RuntimeError("Die Zusammenfassung ist leer ausgefallen.")
        return summary


class TTSService:
    def __init__(self, piper_command: str, voice_path: str, output_dir: str, logger: AppLogger):
        self.piper_command = piper_command
        self.voice_path = voice_path
        self.output_dir = output_dir
        self.logger = logger

    def speak(self, text: str):
        if not text.strip():
            raise RuntimeError("Leerer Text kann nicht vertont werden.")

        input_txt_path = os.path.join(self.output_dir, "piper_input.txt")
        output_wav_path = os.path.join(self.output_dir, "response.wav")

        try:
            with open(input_txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError as e:
            raise RuntimeError(f"TTS-Eingabedatei konnte nicht geschrieben werden: {e}") from e

        try:
            subprocess.run(
                [
                    self.piper_command,
                    "--model", self.voice_path,
                    "--output-file", output_wav_path,
                    "--input-file", input_txt_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as e:
            raise RuntimeError("Piper wurde nicht gefunden. Prüfe den PATH oder PIPER_COMMAND.") from e
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").strip()
            raise RuntimeError(f"Piper konnte die Audiodatei nicht erzeugen. {stderr}") from e

        try:
            import playsound
            playsound.playsound(output_wav_path)
        except Exception as e:
            raise RuntimeError(f"Die generierte Audiodatei konnte nicht abgespielt werden: {e}") from e


class ProjectDescriptionService:
    def __init__(self, project_dir: str):
        self.project_dir = project_dir

    def list_projects(self) -> List[str]:
        if not os.path.isdir(self.project_dir):
            return []
        projects = []
        for filename in os.listdir(self.project_dir):
            lower = filename.lower()
            if lower.endswith(".docx") and not filename.startswith("~$"):
                projects.append(filename)
        return sorted(projects, key=str.casefold)

    def _extract_docx_text(self, filepath: str) -> str:
        try:
            with zipfile.ZipFile(filepath, "r") as archive:
                xml_bytes = archive.read("word/document.xml")
        except Exception as e:
            raise RuntimeError(f"Projektdatei konnte nicht gelesen werden: {e}") from e

        try:
            root = ET.fromstring(xml_bytes)
            namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            paragraphs = []
            for paragraph in root.findall(".//w:p", namespace):
                runs = []
                for text_node in paragraph.findall(".//w:t", namespace):
                    if text_node.text:
                        runs.append(text_node.text)
                line = "".join(runs).strip()
                if line:
                    paragraphs.append(line)
            return "\n".join(paragraphs).strip()
        except Exception as e:
            raise RuntimeError(f"Projektdatei konnte nicht verarbeitet werden: {e}") from e

    def get_project_text(self, project_filename: str) -> str:
        if not project_filename:
            return ""
        filepath = os.path.join(self.project_dir, project_filename)
        if not os.path.exists(filepath):
            raise RuntimeError("Die ausgewählte Projektdatei wurde nicht gefunden.")
        return self._extract_docx_text(filepath)


class AudioRecorder:
    def __init__(self, audio_queue: queue.Queue, logger: AppLogger, status_callback: Callable[[str], None]):
        self.audio_queue = audio_queue
        self.logger = logger
        self.status_callback = status_callback
        self.audio_stream: Optional[sd.InputStream] = None
        self.is_recording = False

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            self.status_callback(f"Audio-Status: {status}")
        if self.is_recording:
            self.audio_queue.put(indata.copy())

    def start(self):
        if self.is_recording:
            return
        try:
            self.audio_stream = sd.InputStream(
                samplerate=INPUT_SAMPLE_RATE,
                channels=1,
                callback=self.audio_callback,
                blocksize=int(INPUT_SAMPLE_RATE * CHUNK_DURATION),
            )
            self.audio_stream.start()
            self.is_recording = True
        except Exception as e:
            raise RuntimeError(f"Audioaufnahme konnte nicht gestartet werden: {e}") from e

    def stop(self):
        if not self.is_recording:
            return
        self.is_recording = False
        try:
            if self.audio_stream is not None:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
        except Exception as e:
            raise RuntimeError(f"Audioaufnahme konnte nicht gestoppt werden: {e}") from e


class DiscussionAssistantApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Diskussions-Assistent")
        self.root.geometry("1100x760")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.status_var = tk.StringVar(value="Initialisierung läuft ...")
        self.logger = AppLogger(log_callback=lambda msg: self.root.after(0, lambda: self.set_status(msg)))
        self.audio_queue: queue.Queue = queue.Queue()
        self.conversation_manager = ConversationManager()
        self.project_display_to_filename: Dict[str, str] = {}

        self.vad_triggered = False
        self.speech_buffer: List[np.ndarray] = []
        self.silence_start: Optional[float] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.summary_lock = threading.Lock()
        self.is_summarizing = False

        self._build_gui()

        self.set_status("Lade VAD-Modell ...")
        try:
            self.model, self.utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
        except Exception as e:
            messagebox.showerror("Fehler", f"VAD-Modell konnte nicht geladen werden:\n{e}")
            raise

        self.audio_recorder = AudioRecorder(
            audio_queue=self.audio_queue,
            logger=self.logger,
            status_callback=lambda text: self.root.after(0, lambda: self.set_status(text)),
        )
        self.transcription_service = TranscriptionService(api_key=OPENAI_API_KEY, logger=self.logger)
        self.llm_service = LLMService(api_key=OPENAI_API_KEY, logger=self.logger)
        self.tts_service = TTSService(
            piper_command=PIPER_COMMAND,
            voice_path=PIPER_VOICE_PATH,
            output_dir=PIPER_OUTPUT_DIR,
            logger=self.logger,
        )
        self.session_service = SessionService(session_dir=SESSION_DIR, logger=self.logger)
        self.project_description_service = ProjectDescriptionService(project_dir=PROJECT_DESCRIPTIONS_DIR)

        self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.processing_thread.start()
        self.refresh_project_dropdown()
        self.set_status("Bereit")

    # === GUI ===
    def _build_gui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill="x", padx=10, pady=10)

        tk.Label(top_frame, text="Partei wählen:").pack(side="left")
        self.partei_var = tk.StringVar()
        self.dropdown = ttk.Combobox(top_frame, textvariable=self.partei_var, state="readonly", width=15)
        self.dropdown["values"] = tuple(CONFIG["party_options"])
        self.dropdown.pack(side="left", padx=(8, 20))

        tk.Label(top_frame, text="Projekt:").pack(side="left")
        self.project_var = tk.StringVar(value="Kein Projekt")
        self.project_dropdown = ttk.Combobox(top_frame, textvariable=self.project_var, state="readonly", width=30)
        self.project_dropdown["values"] = ("Kein Projekt",)
        self.project_dropdown.current(0)
        self.project_dropdown.pack(side="left", padx=(8, 8))

        self.start_button = tk.Button(top_frame, text="Aufnahme starten", command=self.start_recording)
        self.start_button.pack(side="left", padx=4)

        self.stop_button = tk.Button(top_frame, text="Aufnahme stoppen", command=self.stop_recording, state="disabled")
        self.stop_button.pack(side="left", padx=4)

        self.clear_button = tk.Button(top_frame, text="Transkript löschen", command=self.clear_transcript)
        self.clear_button.pack(side="left", padx=4)

        self.generate_button = tk.Button(top_frame, text="Antwort generieren", command=self.on_generate_response)
        self.generate_button.pack(side="left", padx=(20, 4))

        self.speak_button = tk.Button(top_frame, text="Antwort vorlesen", command=self.on_speak_response, state="disabled")
        self.speak_button.pack(side="left", padx=4)

        self.save_button = tk.Button(top_frame, text="Sitzung speichern", command=self.save_session)
        self.save_button.pack(side="left", padx=(20, 4))

        self.export_button = tk.Button(top_frame, text="Exportieren", command=self.export_session)
        self.export_button.pack(side="left", padx=4)

        status_frame = tk.Frame(self.root)
        status_frame.pack(fill="x", padx=10, pady=(0, 6))
        tk.Label(status_frame, text="Status:", font=("Arial", 10, "bold")).pack(side="left")
        tk.Label(status_frame, textvariable=self.status_var).pack(side="left", padx=8)

        transcript_frame = tk.LabelFrame(self.root, text="Laufendes Transkript")
        transcript_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.transcript_text = ScrolledText(transcript_frame, wrap="word", height=16)
        self.transcript_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.transcript_text.configure(state="disabled")

        answer_frame = tk.LabelFrame(self.root, text="Letzte generierte Antwort")
        answer_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.answer_text = ScrolledText(answer_frame, wrap="word", height=10)
        self.answer_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.answer_text.configure(state="disabled")

    def set_status(self, text: str):
        self.status_var.set(text)

    def _format_project_display_name(self, filename: str) -> str:
        name, _ = os.path.splitext(filename)
        if name.lower().startswith("one-pager_"):
            name = name[len("One-Pager_"):]
        return name.replace("_", " ").strip()

    def refresh_project_dropdown(self):
        projects = self.project_description_service.list_projects()
        self.project_display_to_filename = {}
        display_names: List[str] = []

        for filename in projects:
            display_name = self._format_project_display_name(filename)
            if not display_name:
                display_name = filename
            if display_name in self.project_display_to_filename:
                display_name = f"{display_name} ({filename})"
            self.project_display_to_filename[display_name] = filename
            display_names.append(display_name)

        options = ["Kein Projekt", *display_names]
        self.project_dropdown["values"] = tuple(options)
        current = self.project_var.get()
        if current in options:
            self.project_var.set(current)
        else:
            self.project_var.set("Kein Projekt")

    def show_error(self, title: str, message: str):
        self.root.after(0, lambda: messagebox.showerror(title, message))

    def update_transcript_display(self):
        joined = self.conversation_manager.get_formatted_transcript()
        self.transcript_text.configure(state="normal")
        self.transcript_text.delete("1.0", tk.END)
        self.transcript_text.insert(tk.END, joined)
        self.transcript_text.configure(state="disabled")
        self.transcript_text.see(tk.END)

    def update_answer_display(self, text: str):
        self.answer_text.configure(state="normal")
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert(tk.END, text)
        self.answer_text.configure(state="disabled")
        self.answer_text.see(tk.END)

    # === Conversation ===
    def add_transcript_entry(self, text: str):
        self.conversation_manager.add_entry(text)
        self.root.after(0, self.update_transcript_display)
        if self.conversation_manager.needs_summarization():
            self.trigger_summarization()

    def trigger_summarization(self):
        with self.summary_lock:
            if self.is_summarizing:
                return
            self.is_summarizing = True
        threading.Thread(target=self._summarize_worker, daemon=True).start()

    def _summarize_worker(self):
        try:
            existing_summary, old_entries, summary_input = self.conversation_manager.get_entries_for_summarization()
            if not old_entries or not summary_input.strip():
                return
            self.root.after(0, lambda: self.set_status("Ältere Diskussion wird verdichtet ..."))
            new_summary = self.llm_service.summarize(existing_summary, summary_input)
            self.conversation_manager.apply_summary(new_summary)
            self.root.after(0, self.update_transcript_display)
            self.root.after(0, lambda: self.set_status("Kontext wurde verdichtet"))
        except RuntimeError as e:
            self.root.after(0, lambda: self.set_status("Zusammenfassung fehlgeschlagen"))
            self.show_error("Zusammenfassung", str(e))
        finally:
            with self.summary_lock:
                self.is_summarizing = False

    def clear_transcript(self):
        try:
            if self.audio_recorder.is_recording:
                self.stop_recording()
        except Exception:
            pass

        self.conversation_manager.clear()
        self.speech_buffer = []
        self.silence_start = None
        self.vad_triggered = False

        self.update_transcript_display()
        self.update_answer_display("")
        self.speak_button.configure(state="disabled")
        self.set_status("Transkript gelöscht")

    def get_context_for_prompt(self) -> str:
        return self.conversation_manager.get_prompt_context()

    def current_session_data(self) -> Dict[str, Any]:
        return self.conversation_manager.serialize(self.partei_var.get())

    # === Audio ===
    def start_recording(self):
        try:
            self.audio_recorder.start()
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.set_status("Aufnahme läuft")
        except RuntimeError as e:
            self.show_error("Fehler", str(e))
            self.set_status("Fehler beim Starten der Aufnahme")

    def stop_recording(self):
        try:
            self.audio_recorder.stop()
        except RuntimeError as e:
            self.show_error("Fehler", str(e))

        if self.speech_buffer:
            audio_chunk = np.concatenate(self.speech_buffer)
            self.speech_buffer = []
            self.vad_triggered = False
            self.silence_start = None
            threading.Thread(target=self.save_and_transcribe, args=(audio_chunk,), daemon=True).start()

        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.set_status("Aufnahme gestoppt")

    def preprocess_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        if audio_chunk.size == 0:
            return None

        audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(audio_chunk))))
        if rms < MIN_RMS_THRESHOLD:
            return None

        peak = float(np.max(np.abs(audio_chunk)))
        if peak > 0:
            gain = min(NORMALIZE_PEAK_TARGET / peak, 8.0)
            audio_chunk = np.clip(audio_chunk * gain, -1.0, 1.0)

        gate_threshold = max(MIN_RMS_THRESHOLD * 0.6, 0.002)
        audio_chunk[np.abs(audio_chunk) < gate_threshold] = 0.0
        return audio_chunk

    def process_audio(self):
        while True:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                mono = chunk[:, 0] if chunk.ndim > 1 else chunk
                resampled = torchaudio.functional.resample(
                    torch.from_numpy(mono),
                    orig_freq=INPUT_SAMPLE_RATE,
                    new_freq=SAMPLE_RATE,
                )
            except Exception as e:
                self.root.after(0, lambda: self.set_status(f"Audioverarbeitung fehlgeschlagen: {e}"))
                continue

            for i in range(0, len(resampled) - 512 + 1, 512):
                frame = resampled[i:i + 512].unsqueeze(0)
                try:
                    prob = self.model(frame, SAMPLE_RATE).item()
                except Exception as e:
                    self.root.after(0, lambda: self.set_status(f"VAD-Fehler: {e}"))
                    break

                if prob > VAD_THRESHOLD:
                    if not self.vad_triggered:
                        self.vad_triggered = True
                        self.root.after(0, lambda: self.set_status("Sprache erkannt"))
                    self.speech_buffer.append(frame.squeeze(0).numpy())
                    self.silence_start = None

                elif self.vad_triggered:
                    if self.silence_start is None:
                        self.silence_start = time.time()
                    elif time.time() - self.silence_start > SILENCE_THRESHOLD:
                        self.vad_triggered = False
                        self.root.after(0, lambda: self.set_status("Stille erkannt – transkribiere ..."))
                        if self.speech_buffer:
                            audio_chunk = np.concatenate(self.speech_buffer)
                            self.speech_buffer = []
                            self.silence_start = None
                            threading.Thread(target=self.save_and_transcribe, args=(audio_chunk,), daemon=True).start()

    # === Transkription ===
    def save_and_transcribe(self, audio_chunk: np.ndarray):
        duration_seconds = len(audio_chunk) / SAMPLE_RATE
        if duration_seconds < MIN_AUDIO_SECONDS:
            self.root.after(0, lambda: self.set_status("Audiosegment zu kurz – verworfen"))
            return

        audio_chunk = self.preprocess_audio_chunk(audio_chunk)
        if audio_chunk is None:
            self.root.after(0, lambda: self.set_status("Audiosegment zu leise – verworfen"))
            return

        filename = os.path.join(AUDIO_CHUNK_DIR, f"chunk_{int(time.time() * 1000)}.wav")
        try:
            sf.write(filename, audio_chunk, SAMPLE_RATE)
        except Exception as e:
            self.root.after(0, lambda: self.set_status("Audio konnte nicht gespeichert werden"))
            self.show_error("Fehler", f"Audiosegment konnte nicht gespeichert werden:\n{e}")
            return

        try:
            text = self.transcription_service.transcribe_file(filename)
            if text:
                self.add_transcript_entry(text)
                self.root.after(0, lambda: self.set_status("Transkript aktualisiert"))
            else:
                self.root.after(0, lambda: self.set_status("Leeres Transkript erhalten"))
        except RuntimeError as e:
            self.root.after(0, lambda: self.set_status("Transkription fehlgeschlagen"))
            self.show_error("Transkription", str(e))

    # === LLM ===
    def on_generate_response(self):
        partei = self.partei_var.get()
        if not partei:
            messagebox.showwarning("Fehler", "Bitte wähle eine Partei aus.")
            return

        context_text = self.get_context_for_prompt()
        if not context_text:
            messagebox.showwarning("Fehler", "Es ist noch kein Transkript vorhanden.")
            return

        self.generate_button.configure(state="disabled")
        self.set_status("LLM-Antwort wird generiert ...")
        threading.Thread(target=self._generate_response_worker, args=(context_text, partei), daemon=True).start()

    def _generate_response_worker(self, context_text: str, partei: str):
        try:
            project_name = self.project_var.get().strip()
            project_context = ""
            if project_name and project_name != "Kein Projekt":
                project_filename = self.project_display_to_filename.get(project_name, project_name)
                project_context = self.project_description_service.get_project_text(project_filename)

            response = self.llm_service.query(context_text, partei, VS_DICT, project_context)
            self.conversation_manager.set_last_response(response)
            self.root.after(0, lambda: self.update_answer_display(response))
            self.root.after(0, lambda: self.speak_button.configure(state="normal"))
            self.root.after(0, lambda: self.set_status("Antwort generiert"))
            self.auto_save_session(silent=True)
        except RuntimeError as e:
            self.root.after(0, lambda: self.set_status("LLM-Anfrage fehlgeschlagen"))
            self.show_error("LLM", str(e))
        finally:
            self.root.after(0, lambda: self.generate_button.configure(state="normal"))

    def on_speak_response(self):
        text = self.conversation_manager.get_last_response().strip()
        if not text:
            messagebox.showwarning("Fehler", "Es ist noch keine Antwort vorhanden.")
            return

        self.speak_button.configure(state="disabled")
        threading.Thread(target=self._speak_response_worker, args=(text,), daemon=True).start()

    # === TTS ===
    def _speak_response_worker(self, text: str):
        self.root.after(0, lambda: self.set_status("TTS-Ausgabe läuft ..."))
        try:
            self.tts_service.speak(text)
            self.root.after(0, lambda: self.set_status("Bereit"))
        except RuntimeError as e:
            self.root.after(0, lambda: self.set_status("TTS fehlgeschlagen"))
            self.show_error("TTS", str(e))
        finally:
            self.root.after(
                0,
                lambda: self.speak_button.configure(
                    state="normal" if self.conversation_manager.get_last_response().strip() else "disabled"
                ),
            )

    # === Sitzung / Export ===
    def auto_save_session(self, silent: bool = False):
        try:
            path = self.session_service.auto_save(self.current_session_data())
            if not silent:
                self.set_status(f"Sitzung gespeichert: {os.path.basename(path)}")
            return path
        except Exception as e:
            if not silent:
                self.show_error("Sitzung", f"Sitzung konnte nicht gespeichert werden:\n{e}")
            return None

    def save_session(self):
        self.auto_save_session(silent=False)

    def export_session(self):
        session_data = self.current_session_data()
        filepath = filedialog.asksaveasfilename(
            title="Sitzung exportieren",
            defaultextension=".txt",
            filetypes=[
                ("Textdatei", "*.txt"),
                ("JSON-Datei", "*.json"),
                ("PDF-Datei", "*.pdf"),
            ],
        )
        if not filepath:
            return

        try:
            lower = filepath.lower()
            if lower.endswith(".json"):
                self.session_service.export_json(filepath, session_data)
            elif lower.endswith(".pdf"):
                self.session_service.export_pdf(filepath, session_data)
            else:
                self.session_service.export_txt(filepath, session_data)
            self.set_status("Export erfolgreich")
        except Exception as e:
            self.show_error("Export", f"Export fehlgeschlagen:\n{e}")
            self.set_status("Export fehlgeschlagen")

    def on_close(self):
        try:
            if self.conversation_manager.get_entries() or self.conversation_manager.get_last_response():
                self.auto_save_session(silent=True)
        except Exception:
            pass
        try:
            self.audio_recorder.stop()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    try:
        app = DiscussionAssistantApp()
        app.run()
    except Exception as e:
        print(f"[FATAL] Anwendung konnte nicht gestartet werden: {e}")
