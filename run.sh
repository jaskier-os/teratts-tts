#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Load environment variables from .env if present
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

if [ ! -d "venv" ]; then
  echo "[teratts-tts] Creating venv..."
  python3 -m venv venv
fi

source venv/bin/activate

echo "[teratts-tts] Installing dependencies..."
pip install -q --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch
pip install -q -r requirements.txt

echo "[teratts-tts] Starting server on ${TTS_HOST:-0.0.0.0}:${TTS_PORT:-10018}..."
exec python server.py
