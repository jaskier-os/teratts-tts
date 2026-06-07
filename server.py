#!/usr/bin/env python3
"""TeraTTS HTTP server for Russian speech synthesis (piper-compatible API).

Supports streaming: splits text into sentences, generates and streams
Opus chunks as each sentence is synthesized.
"""

import argparse
import io
import logging
import os
import subprocess
import wave

import numpy as np
from aiohttp import web
from TeraTTS import TTS
from ruaccent import RUAccent

logger = logging.getLogger("teratts-tts")

tts = None
accentizer = None
has_ffmpeg = False
SAMPLE_RATE = 22050
LENGTH_SCALE = 0.81


def wav_to_opus(wav_bytes: bytes) -> bytes:
    """Convert WAV to OGG/Opus using ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-i", "pipe:0", "-c:a", "libopus", "-b:a", "64k", "-f", "ogg", "pipe:1"],
        input=wav_bytes, capture_output=True, timeout=30
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()[:200]}")
    return proc.stdout


def audio_to_wav(audio: np.ndarray) -> bytes:
    """Convert float32/int16 audio array to WAV bytes."""
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        audio = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


async def handle_tts(request):
    """Generate audio from text. Always produces a single OGG/Opus or WAV response.
    The orchestrator handles sentence splitting -- this server should never re-split."""
    text = await request.text()
    if not text.strip():
        return web.Response(status=400, text="Empty text")

    try:
        processed = accentizer.process_all(text)
    except Exception as e:
        logger.warning("RUAccent failed for '%s': %s -- using unaccented text", text[:50], e)
        processed = text
    audio = tts(processed, lenght_scale=LENGTH_SCALE)
    wav_bytes = audio_to_wav(audio)

    accept = request.headers.get("Accept", "")
    if has_ffmpeg and ("opus" in accept or "ogg" in accept):
        try:
            opus_bytes = wav_to_opus(wav_bytes)
            return web.Response(body=opus_bytes, content_type="audio/ogg")
        except Exception as e:
            logger.warning("Opus conversion failed, returning WAV: %s", e)

    return web.Response(body=wav_bytes, content_type="audio/wav")


async def handle_health(request):
    return web.json_response({"status": "ok"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("TTS_PORT", "10018")))
    parser.add_argument("--host", default=os.environ.get("TTS_HOST", "0.0.0.0"))
    parser.add_argument("--model", default=os.environ.get("TTS_MODEL", "TeraTTS/glados2-g2p-vits"))
    parser.add_argument("--length-scale", type=float,
                        default=float(os.environ.get("TTS_LENGTH_SCALE", "0.81")))
    args = parser.parse_args()

    global tts, accentizer, has_ffmpeg, LENGTH_SCALE
    LENGTH_SCALE = args.length_scale

    logger.info("Loading RUAccent...")
    accentizer = RUAccent()
    accentizer.load(omograph_model_size='turbo', use_dictionary=True)
    logger.info("RUAccent loaded.")

    logger.info("Loading TeraTTS model: %s", args.model)
    tts = TTS(args.model, add_time_to_end=0.5, tokenizer_load_dict=True)
    logger.info("Model loaded.")

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        has_ffmpeg = True
        logger.info("ffmpeg available -- Opus output supported")
    except Exception:
        has_ffmpeg = False
        logger.info("ffmpeg not found -- WAV output only")

    app = web.Application()
    app.router.add_post("/", handle_tts)
    app.router.add_get("/health", handle_health)

    logger.info("Starting server on %s:%d", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[teratts-tts] %(message)s")
    main()
