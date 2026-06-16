#!/usr/bin/env python3
"""Local experiment: speak English words through the Russian glados VITS model.

Compares two paths per word:
  (a) naive   -- current server behaviour: transliterate Latin->Cyrillic, then RU g2p
  (b) phonetic-- English g2p_en (ARPAbet) -> mapped to the model's RU phoneme symbols

Writes WAVs to scripts/out/ for A/B listening. No server, no network.
"""
import os
import sys
import wave

import numpy as np
from TeraTTS import TTS
from g2p_en import G2p

MODEL = "TeraTTS/glados2-g2p-vits"
LENGTH_SCALE = 0.81
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# ARPAbet (stress digits stripped) -> list of model RU phoneme symbols.
# RU vowels carry stress as a trailing 0 (unstressed) / 1 (stressed); we attach
# it from the ARPAbet stress digit. Consonants are emitted hard (non-palatalized).
ARPA_VOWEL = {
    "AA": "a", "AE": "a", "AH": "a", "AO": "o",
    "EH": "e", "IH": "i", "IY": "i", "UH": "u", "UW": "u",
    # diphthongs -> vowel core (+glide handled below)
    "AW": "a", "AY": "a", "EY": "e", "OW": "o", "OY": "o", "ER": "e",
}
# diphthong glide tail (added after the stressed/unstressed core vowel)
ARPA_DIPH_TAIL = {
    "AW": ["u0"], "AY": ["j"], "EY": ["j"], "OW": ["u0"], "OY": ["j"], "ER": ["r"],
}
ARPA_CONS = {
    "B": ["b"], "CH": ["ch"], "D": ["d"], "DH": ["z"], "F": ["f"],
    "G": ["g"], "HH": ["h"], "JH": ["d", "zh"], "K": ["k"], "L": ["l"],
    "M": ["m"], "N": ["n"], "NG": ["n"], "P": ["p"], "R": ["r"],
    "S": ["s"], "SH": ["sh"], "T": ["t"], "TH": ["s"], "V": ["v"],
    "W": ["v"], "Y": ["j"], "Z": ["z"], "ZH": ["zh"],
}


def arpa_to_ru_symbols(arpa_phones):
    """Convert a list of ARPAbet phones (e.g. ['HH','AH0','L','OW1']) to RU model symbols."""
    out = []
    for p in arpa_phones:
        if not p or not p[0].isalpha():
            continue
        stress = ""
        base = p
        if p[-1].isdigit():
            base, stress = p[:-1], p[-1]
        if base in ARPA_VOWEL:
            ru_stress = "1" if stress in ("1", "2") else "0"
            out.append(ARPA_VOWEL[base] + ru_stress)
            out.extend(ARPA_DIPH_TAIL.get(base, []))
        elif base in ARPA_CONS:
            out.extend(ARPA_CONS[base])
    return out


def symbols_to_ids(symbols, phoneme_id_map):
    ids = []
    ids.extend(phoneme_id_map["^"])
    ids.extend(phoneme_id_map["_"])
    for s in symbols:
        if s in phoneme_id_map:
            ids.extend(phoneme_id_map[s])
            ids.extend(phoneme_id_map["_"])
    ids.extend(phoneme_id_map["$"])
    return ids


def synth_from_symbols(tts, symbols, length_scale=LENGTH_SCALE):
    """Run the ONNX model directly from RU phoneme symbols (bypasses translit + g2p)."""
    pim = tts.tokenizer.config["phoneme_id_map"]
    ids = symbols_to_ids(symbols, pim)
    ids = tts._intersperse(ids, 0)
    text = np.expand_dims(np.array(ids, dtype=np.int64), 0)
    text_lengths = np.array([text.shape[1]], dtype=np.int64)
    scales = np.array([0.667, length_scale, 0.8], dtype=np.float32)
    audio = tts.model.run(None, {"input": text, "input_lengths": text_lengths,
                                 "scales": scales, "sid": None})[0][0, 0][0]
    return tts._add_silent(audio, silence_duration=tts.add_time_to_end)


def save_wav(audio, path):
    if audio.dtype in (np.float32, np.float64):
        audio = (audio * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(audio.tobytes())


def main():
    words = sys.argv[1:] or [
        "hello", "meeting", "google", "schedule", "thought",
        "computer", "weekend", "message", "download", "the weather is nice",
    ]
    print(f"Loading model {MODEL} ...", flush=True)
    tts = TTS(MODEL, add_time_to_end=0.5, tokenizer_load_dict=True)
    g2p = G2p()
    print("Loaded.\n", flush=True)

    for i, w in enumerate(words):
        tag = f"{i:02d}_" + "".join(c if c.isalnum() else "_" for c in w)[:24]

        # (a) naive -- exactly what the server does today
        naive_audio = tts(w, lenght_scale=LENGTH_SCALE)
        save_wav(naive_audio, os.path.join(OUT, f"{tag}_naive.wav"))

        # (b) phonetic -- English g2p -> RU phoneme symbols
        arpa = g2p(w)
        # per-word: g2p returns word phones with spaces between words preserved as ' '
        syms = []
        for tok in arpa:
            if tok == " ":
                syms.append(" ")
            else:
                syms.extend(arpa_to_ru_symbols([tok]))
        phon_audio = synth_from_symbols(tts, syms)
        save_wav(phon_audio, os.path.join(OUT, f"{tag}_phonetic.wav"))

        print(f"{w!r}\n  arpa : {arpa}\n  ru   : {syms}", flush=True)

    print(f"\nWAVs in {OUT}", flush=True)


if __name__ == "__main__":
    main()
