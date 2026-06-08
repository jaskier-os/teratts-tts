# CLAUDE.md -- teratts-tts

This file provides guidance to Claude Code when working in this repository.

IMPORTANT: NEVER USE EMOJIS ANYWHERE IN LOGGING, CODE OR OTHER TEXT!

IMPORTANT: Treat this codebase as work in progress. Never do backwards-compatibility or legacy support unless explicitly asked to. Remove code that becomes redundant.

## What this is

HTTP microservice for Russian text-to-speech built on [TeraTTS](https://github.com/Tera2Space/TeraTTS) with [RUAccent](https://github.com/Den4ikAI/ruaccent) for automatic stress placement. POST plain UTF-8 text to `/`, get WAV (or OGG/Opus when the client sends `Accept: audio/ogg` and `ffmpeg` is available). `GET /health` returns `{"status": "ok"}`. Sentence splitting is left to the caller. Default port 10018.

It is one of several TTS backends in a larger AI orchestration system. For the whole-system map -- architecture, full port allocation, and every service -- see the `jaskier-os/orchestrator` repo's CLAUDE.md / docs. Don't duplicate that here.

## Key files

- `server.py` -- HTTP server (the whole service). CLI flags `--host`, `--port`, `--model`, `--length-scale` override the env.
- `run.sh` -- local run helper: makes a venv, installs deps (incl. CPU PyTorch), starts from `.env`
- `fetch_models.sh` -- pre-caches the public HuggingFace model (~200 MB) for offline/container use
- `test_server.py` -- smoke test against a running server (override base URL with `TTS_TEST_URL`)
- `Dockerfile` -- model pre-downloaded into the image at build time
- `.env.example` -- source of truth for configuration

## Build / run

See `README.md`. Quick local run: `./run.sh`. Config is env vars only; copy `.env.example` to `.env`. The ones that matter: `TTS_PORT` (default 10018), `TTS_HOST`, `TTS_MODEL` (HF model id, default `TeraTTS/glados2-g2p-vits`), `TTS_LENGTH_SCALE` (speed, lower = faster).

Python 3.12, PyTorch (CPU). Needs `libportaudio2` (via `sounddevice`) and optionally `ffmpeg`.

## Deployment

This service auto-deploys on push to `main`. Its own `.gitlab-ci.yml` builds the Dockerfile, pushes the image to `localhost:5000`, then commits an `image:` tag bump into the `infrastructure/deploy` repo, and FluxCD reconciles it onto the cluster.

- Do NOT build/push Docker images by hand. Commit + push and let CI do it.
- NEVER modify the Kubernetes cluster directly (`kubectl set env/edit/patch`). All deploy changes go through the `infrastructure/deploy` repo so Flux reconciles them; direct changes get overwritten and cause drift.
- `main` is protected; normal (non-force) commits by Maintainers are fine. NEVER force-push.
