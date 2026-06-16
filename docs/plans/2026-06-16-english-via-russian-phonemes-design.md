# English-via-Russian-phonemes TTS + configurable orchestrator routing

Date: 2026-06-16

## Goal

Make the Russian glados/TeraTTS model pronounce English words (in a Russian
accent) without retraining, and make the orchestrator able to route all TTS to
TeraTTS instead of splitting English/Russian across engines.

Two independent but related changes:

1. **teratts-tts server** gains a phonetic English path: English words are run
   through `g2p_en` (ARPAbet) and mapped to the model's own 62 Russian phoneme
   symbols, then fed to the ONNX model. Validated locally (A/B vs. the naive
   `translit` path); user confirmed the phonetic output is correct.
2. **orchestrator** gets a configurable TTS routing mode. Today it detects
   Cyrillic vs. Latin per segment and routes English -> Kokoro, Russian ->
   TeraTTS, concatenating per-segment WAVs for mixed text. A new
   `TTS_ROUTING_MODE` flag lets the whole pipeline route to TeraTTS only.

## Non-goals

- No model retraining.
- No deletion of the existing Kokoro / language-split code. It stays and is the
  default behavior.
- No tuning of the ARPAbet->RU map beyond the validated first pass (TH->s/z,
  W->v, JH->d+zh, NG->n, no unstressed-vowel reduction). Future refinement is
  out of scope.

## Approach

### teratts-tts server (the real new capability)

The TeraTTS tokenizer already does dictionary-first lookup, then maps phoneme
strings through `config.json`'s `phoneme_id_map`. The model is phoneme-based;
the Python API has no direct-phoneme flag, so we build phoneme IDs ourselves and
call the ONNX session directly (prefix `^` `_`, `_` separators, suffix `$`, then
`_intersperse(ids, 0)`), exactly as the validated `scripts/exp_english.py` does.

- New module `english_g2p.py`:
  - lazy-initialized `g2p_en.G2p`
  - the `ARPA_VOWEL` / `ARPA_DIPH_TAIL` / `ARPA_CONS` maps and
    `arpa_to_ru_symbols(arpa_phones) -> [ru_symbol, ...]` lifted verbatim from
    the validated experiment script.
- `server.py` mixed-language handling in `handle_tts`:
  1. Split incoming text into contiguous Latin vs. Cyrillic word-runs
     (non-alphabetic chars attach to the current run, matching how the
     orchestrator's `segmentByLanguage` behaves).
  2. Cyrillic runs -> existing path: `accentizer.process_all` then the
     tokenizer's `g2p()` to get RU phoneme symbols.
  3. Latin runs -> `g2p_en` -> `arpa_to_ru_symbols` -> RU phoneme symbols.
  4. Concatenate the symbol lists (with word-separator spaces), map through
     `phoneme_id_map`, intersperse, run the ONNX model ONCE, encode WAV/Opus as
     today.
- Output contract unchanged: POST text -> WAV, or OGG/Opus when
  `Accept: audio/ogg` and ffmpeg present. `/health` unchanged.
- New deps: `g2p_en` in `requirements.txt`; NLTK data
  (`averaged_perceptron_tagger`, `averaged_perceptron_tagger_eng`, `cmudict`,
  `punkt`, `punkt_tab`) pre-fetched in `fetch_models.sh` and baked into the
  Docker image so first request is not blocked on a download.

Gotcha: the installed TeraTTS lib uses the misspelled kwarg `lenght_scale`
(server.py already does too) -- keep using it.

### orchestrator routing (configurable, nothing deleted)

- `config.js`: add `ttsRoutingMode`, env `TTS_ROUTING_MODE`, Joi
  `.valid('language-split', 'teratts').default('language-split')`.
- `tts.js`: `generateAudio`, `generateNotifAudio`, and `streamTts` branch on the
  mode:
  - `language-split` (default): existing behavior, unchanged. English -> Kokoro,
    Russian -> TeraTTS, mixed -> per-segment concat.
  - `teratts`: skip `segmentByLanguage`; send the whole text / each sentence to
    the TeraTTS endpoint (`generateAudioPiperOpus` / `generateAudioPiper`, which
    already point at `piperTtsUrl` = teratts-tts:10018 in prod). Kokoro and the
    WAV-concat helpers remain in the file, just not exercised in this mode.
- `.env.example`: document `TTS_ROUTING_MODE` (default `language-split`).
- `orchestrator/CLAUDE.md`: document the flag and the two modes under the TTS
  notes.

Note: `PIPER_TTS_URL` already resolves to `teratts-tts.ai.svc.cluster.local:10018`
in the deploy manifests, so `teratts` mode needs no new endpoint wiring.

## Data flow

```
Device text -> orchestrator tts.js
  mode=language-split: segmentByLanguage -> Kokoro(en) / TeraTTS(ru) / concat(mixed)
  mode=teratts:        whole text -> TeraTTS (10018)

TeraTTS server (handle_tts):
  split Latin/Cyrillic runs
    Cyrillic -> RUAccent + tokenizer.g2p -> RU phoneme symbols
    Latin    -> g2p_en (ARPAbet) -> arpa_to_ru_symbols -> RU phoneme symbols
  concat symbols -> phoneme_id_map -> intersperse -> ONNX (one call) -> WAV/Opus
```

## Error handling

- teratts: if `g2p_en` yields no phones for a Latin run (e.g. digits already
  expanded upstream, or empty), skip that run; never crash the request. If
  English G2P import/model init fails, log and fall back to the existing
  RUAccent path for the whole text so the endpoint still returns audio.
- orchestrator: invalid `TTS_ROUTING_MODE` is rejected at startup by Joi (fail
  fast), consistent with the rest of `config.js`.

## Testing

- teratts (local, no network): reuse `scripts/exp_english.py` A/B (done). Add a
  mixed-string smoke run ("Я открыл Google и нашёл meeting" etc.) producing one
  clean WAV; bounce to Telegram for the user to confirm. `test_server.py`
  smoke test against a running server still passes for pure-Russian input.
- orchestrator: unit-level assertion that `teratts` mode never calls the Kokoro
  fetch and always hits the TeraTTS endpoint; `language-split` preserves current
  routing (English->Kokoro, Russian->TeraTTS).

## Deployment

Both services auto-deploy on push to `main` via their own GitLab CI (image
build -> tag bump in the deploy repo -> Flux). Do not build images by hand or
edit k8s directly. `TTS_ROUTING_MODE` defaults to `language-split`, so deploying
is a no-op for behavior until the env var is flipped to `teratts` in the
orchestrator deploy manifest.
