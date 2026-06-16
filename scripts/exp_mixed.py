#!/usr/bin/env python3
"""Local smoke test: synthesize mixed Russian+English strings via server.synth_text.

Exercises the full mixed-language phoneme assembly (Cyrillic -> RUAccent + RU g2p,
Latin -> English g2p -> RU phoneme symbols, one ONNX call). Writes WAVs to
scripts/out/ for listening. No server, no network.
"""
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import server
import english_g2p

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

STRINGS = [
    "Я открыл Google и нашёл meeting.",
    "Скачай файл с GitHub завтра.",
    "My schedule is busy сегодня.",
    "Отправь message на email прямо сейчас.",
]


def save_wav(audio, path):
    if audio.dtype in (np.float32, np.float64):
        audio = (audio * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(server.SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


def main():
    print("Loading models ...", flush=True)
    server.load_models()
    print("Loaded.\n", flush=True)

    for i, s in enumerate(STRINGS):
        segs = english_g2p.split_by_script(s)
        audio = server.synth_text(s)
        tag = f"mixed_{i:02d}"
        save_wav(audio, os.path.join(OUT, f"{tag}.wav"))
        print(f"{s!r}\n  segments: {segs}\n  samples : {audio.size}\n", flush=True)

    print(f"WAVs in {OUT}", flush=True)


if __name__ == "__main__":
    main()
