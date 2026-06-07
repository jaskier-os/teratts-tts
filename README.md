# teratts-tts

An HTTP server for Russian text-to-speech synthesis built on
[TeraTTS](https://github.com/Tera2Space/TeraTTS) with
[RUAccent](https://github.com/Den4ikAI/ruaccent) for automatic stress placement.
It exposes a small piper-compatible HTTP API: POST plain UTF-8 Russian text and
receive synthesized audio as WAV, or as OGG/Opus when the client requests it via
the `Accept` header. Sentence splitting is intentionally left to the caller --
each request is synthesized as a single audio stream.

## API

- `POST /` -- body is plain UTF-8 text. Returns `audio/wav` by default, or
  `audio/ogg` (Opus) if the request `Accept` header contains `opus` or `ogg`
  and `ffmpeg` is available. Empty text returns `400`.
- `GET /health` -- returns `{"status": "ok"}`.

## Prerequisites

- Python 3.12
- `ffmpeg` on the `PATH` (optional; required only for OGG/Opus output, falls
  back to WAV automatically if missing)
- System library `libportaudio2` (pulled in by `sounddevice`, a TeraTTS
  dependency)
- ~200 MB free disk for the model, which is downloaded on first run

## Setup

```bash
cp .env.example .env
# edit .env if you want non-default host/port/model/speed
```

Every variable is documented inline in `.env.example`. All of them have working
defaults, so an unedited copy runs out of the box.

## Build / install

The simplest path uses the bundled launcher, which creates a virtualenv,
installs dependencies (including the CPU build of PyTorch) and starts the
server:

```bash
./run.sh
```

To install manually instead:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Model weights

The model named by `TTS_MODEL` (default `TeraTTS/glados2-g2p-vits`) is a
standard pretrained model published on the HuggingFace Hub. It is **not** bundled
in this repository because of its size (~200 MB). The TeraTTS library downloads
it automatically on first startup. To pre-cache it (e.g. for offline or
container deployments) run:

```bash
./fetch_models.sh
```

## Run

```bash
# via launcher (reads .env)
./run.sh

# or directly
source venv/bin/activate
python server.py
```

Configuration is read from the environment (or `.env` via `run.sh`); CLI flags
`--host`, `--port`, `--model`, `--length-scale` override the environment if
given.

Quick smoke test once running:

```bash
curl -s http://localhost:10018/health
curl -s -X POST http://localhost:10018/ --data 'Привет мир' -o out.wav
```

## Test

With a server already running (defaults assume `http://localhost:10018`, override
with `TTS_TEST_URL`):

```bash
python test_server.py
```

The suite checks the health endpoint, empty-text handling, WAV output, Opus
output, single-stream OGG behavior, and realtime-factor performance.

## Docker

```bash
docker build -t teratts-tts .
docker run -p 10018:10018 teratts-tts
```

The model is pre-downloaded into the image at build time. Override the model
with `--build-arg TTS_MODEL=<hf-model-id>`.

## TLS / VPN

This service speaks plain HTTP and requires no certificates or VPN. Put it
behind a reverse proxy (nginx, Caddy, Traefik) if you need TLS termination or
authentication.

## Configuration reference

| Variable           | Default                     | Description                                  |
| ------------------ | --------------------------- | -------------------------------------------- |
| `TTS_HOST`         | `0.0.0.0`                   | Bind interface                               |
| `TTS_PORT`         | `10018`                     | Listen port                                  |
| `TTS_MODEL`        | `TeraTTS/glados2-g2p-vits`  | HuggingFace model id                         |
| `TTS_LENGTH_SCALE` | `0.81`                      | Speech speed (lower = faster)                |
| `TTS_TEST_URL`     | `http://localhost:10018`    | Base URL used by `test_server.py`            |
