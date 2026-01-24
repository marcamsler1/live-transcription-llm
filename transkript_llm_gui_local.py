import os
import json
from dotenv import load_dotenv
load_dotenv()

import time
import queue
import threading
import sounddevice as sd
import numpy as np
import torch
import torchaudio
import requests
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import soundfile as sf
import subprocess
from openai import OpenAI

# === KONFIGURATION ===
WHISPER_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
TEMPERATURE = 0.7
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # Sekunden
SILENCE_THRESHOLD = 1.0  # Sekunden
PIPER_VOICE_PATH = "piper_voices/de_DE-thorsten-high.onnx"
PIPER_COMMAND = "piper"
AUDIO_CHUNK_DIR = "audio_chunks"
PIPER_OUTPUT_DIR = "piper_output"

os.makedirs(AUDIO_CHUNK_DIR, exist_ok=True)
os.makedirs(PIPER_OUTPUT_DIR, exist_ok=True)

vs_dict_json_string = os.getenv("VS_DICT_JSON", "{}")
VS_DICT = json.loads(vs_dict_json_string)

full_transcript = ""
audio_queue = queue.Queue()
vad_triggered = False
speech_buffer = []
silence_start = None

# === SILERO VAD MODELL LADEN ===
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
(get_speech_ts, _, read_audio, _, _) = utils

# === AUDIO CALLBACK ===
def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

# === AUDIOAUFNAHME STARTEN ===
def start_audio_stream():
    with sd.InputStream(samplerate=44100, channels=1, callback=audio_callback, blocksize=int(44100 * CHUNK_DURATION)):
        while True:
            time.sleep(0.1)

# === HINTERGRUNDVERARBEITUNG ===
def process_audio():
    global vad_triggered, speech_buffer, silence_start, full_transcript

    while True:
        if not audio_queue.empty():
            chunk = audio_queue.get()
            mono = chunk[:, 0] if chunk.ndim > 1 else chunk
            resampled = torchaudio.functional.resample(torch.from_numpy(mono), orig_freq=44100, new_freq=SAMPLE_RATE)

            for i in range(0, len(resampled) - 512 + 1, 512):
                frame = resampled[i:i+512].unsqueeze(0)
                prob = model(frame, SAMPLE_RATE).item()

                if prob > 0.5:
                    if not vad_triggered:
                        vad_triggered = True
                        print("[VAD] Sprache erkannt")
                    speech_buffer.append(frame.squeeze(0).numpy())
                    silence_start = None

                elif vad_triggered:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_THRESHOLD:
                        vad_triggered = False
                        print("[VAD] Stille erkannt – Chunk abschließen")
                        audio_chunk = np.concatenate(speech_buffer)
                        save_and_transcribe(audio_chunk)
                        speech_buffer = []
                        silence_start = None

# === AUDIOCHUNK ALS WAV SPEICHERN UND TRANSKRIBIEREN ===
def save_and_transcribe(audio_chunk):
    filename = os.path.join(AUDIO_CHUNK_DIR, f"chunk_{int(time.time())}.wav")
    sf.write(filename, audio_chunk, SAMPLE_RATE)
    print(f"[DATEI] Gespeichert: {filename}")
    
    headers = {
        "Authorization": f"Bearer {WHISPER_API_KEY}"
    }
    files = {
        'file': (filename, open(filename, 'rb')),
        'model': (None, 'whisper-1'),
        'language': (None, 'de')
    }
    response = requests.post("https://api.openai.com/v1/audio/transcriptions", headers=headers, files=files)

    if response.status_code == 200:
        text = response.json()['text']
        print(f"[TRANSKRIPT] {text}")
        global full_transcript
        full_transcript += text + "\n"
    else:
        print(f"[FEHLER] Whisper API: {response.text}")

# === OPENAI LLM AUFRUF ===
def query_llm(transcript, partei):
    client = OpenAI(api_key=WHISPER_API_KEY)
    prompt = f"Sie übernehmen die Rolle des Fraktionschefs der {partei} der Stadt St.Gallen. Sprechen Sie Hochdeutsch. Beziehen Sie sich auf die folgende Diskussion und formulieren Sie eine kurze Antwort aus Sicht der Partei:\n\n{transcript.strip()}"

    tools = []
    tool_choice = None
    if partei in VS_DICT:
        tools = [{
            "type": "file_search",
            "vector_store_ids": [VS_DICT[partei]],
            "max_num_results": 20
        }]
        tool_choice = "required"

    response = client.responses.create(
        model=OPENAI_MODEL,
        temperature=TEMPERATURE,
        tool_choice=tool_choice,
        tools=tools,
        input=prompt
    )

    try:
        reply = response.output_text
        print(f"[LLM-ANTWORT]\n{reply}\n")
        return reply
    except Exception as e:
        print(f"[FEHLER] OpenAI LLM: {e}")
        return "Fehler bei der LLM-Abfrage."

# === TEXT-TO-SPEECH ===
def speak_text(text):
    print("[TTS] Piper-Ausgabe starten...")
    input_txt_path = os.path.join(PIPER_OUTPUT_DIR, "piper_input.txt")
    output_wav_path = os.path.join(PIPER_OUTPUT_DIR, "response.wav")

    with open(input_txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    subprocess.run([
        PIPER_COMMAND,
        "--model", PIPER_VOICE_PATH,
        "--output-file", output_wav_path,
        "--input-file", input_txt_path
    ])

    import playsound
    playsound.playsound(output_wav_path)

# === GUI SETUP ===
def start_gui():
    def on_button_click():
        partei = partei_var.get()
        if not partei:
            messagebox.showwarning("Fehler", "Bitte wähle eine Partei aus.")
            return
        response = query_llm(full_transcript, partei)
        speak_text(response)

    root = tk.Tk()
    root.title("Diskussions-Assistent")

    partei_var = tk.StringVar()
    label = tk.Label(root, text="Partei wählen:")
    label.pack()

    dropdown = ttk.Combobox(root, textvariable=partei_var)
    dropdown['values'] = ("SP", "FDP", "SVP", "MITTE", "GLP", "GRUENE")
    dropdown.pack()

    button = tk.Button(root, text="Antwort generieren", command=on_button_click)
    button.pack(pady=10)

    root.mainloop()

# === HAUPTTHREADS STARTEN ===
threading.Thread(target=start_audio_stream, daemon=True).start()
threading.Thread(target=process_audio, daemon=True).start()

start_gui()
