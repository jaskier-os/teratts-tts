# teratts-tts

HTTP microservice for Russian text-to-speech built on
[TeraTTS](https://github.com/Tera2Space/TeraTTS) with
[RUAccent](https://github.com/Den4ikAI/ruaccent) for automatic stress placement.
POST plain UTF-8 text to `/`, get WAV (or OGG/Opus when the client sends
`Accept: audio/ogg` and `ffmpeg` is available). `GET /health` returns
`{"status": "ok"}`. Sentence splitting is left to the caller.

## Build / run

Local: `run.sh` makes a venv, installs deps (including the CPU build of PyTorch),
and starts the server from `.env`.

```bash
./run.sh
```

Manual:

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python server.py
```

CLI flags `--host`, `--port`, `--model`, `--length-scale` override the environment.

Docker (model pre-downloaded into the image at build time):

```bash
docker build -t teratts-tts .
docker run -p 10018:10018 teratts-tts
# different model: docker build --build-arg TTS_MODEL=<hf-model-id> -t teratts-tts .
```

Smoke test against a running server: `python test_server.py` (override base URL with `TTS_TEST_URL`).

## Configuration

Config is env vars. `.env.example` is the source of truth; copy and edit:

```bash
cp .env.example .env
```

Ones that matter: `TTS_PORT` (default 10018), `TTS_HOST`, `TTS_MODEL`
(HF model id, default `TeraTTS/glados2-g2p-vits`), `TTS_LENGTH_SCALE` (speed, lower = faster).

## Dependencies

Python 3.12, PyTorch (CPU), `TeraTTS`, `ruaccent`. Needs `libportaudio2` (pulled in by
`sounddevice`) and, optionally, `ffmpeg` for OGG/Opus. The model (~200 MB) is a public
HuggingFace model downloaded automatically on first run; `./fetch_models.sh` pre-caches
it for offline/container use.
