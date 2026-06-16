# English-via-Russian-phonemes TTS + configurable orchestrator routing — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the Russian glados/TeraTTS model pronounce English words via Russian phonemes, and add a configurable `TTS_ROUTING_MODE` flag so the orchestrator can route all TTS to TeraTTS instead of splitting English/Russian across engines.

**Architecture:** In `teratts-tts`, English words go through `g2p_en` (ARPAbet) and are mapped to the model's own 62 Russian phoneme symbols, concatenated with the Cyrillic RUAccent+g2p output, and fed to the ONNX model in one call. In `orchestrator/src/tts.js`, a new string-mode config flag (`language-split` default, `teratts` opt-in) gates whether the existing language-detection routing runs or everything goes to TeraTTS. No existing code is deleted.

**Tech Stack:** Python 3.12, TeraTTS, ruaccent, g2p_en, nltk, onnxruntime, aiohttp (server); Node.js 20, Koa, Joi (orchestrator).

**Repos / branches:** Work in place on `main` in each repo:
- `AI/infrastructure/teratts-tts`
- `AI/orchestrator`

**Reference design:** `AI/infrastructure/teratts-tts/docs/plans/2026-06-16-english-via-russian-phonemes-design.md`

**Validated prototype:** `AI/infrastructure/teratts-tts/scripts/exp_english.py` (the ARPAbet->RU map and direct-ONNX synth are already proven; lift from here).

---

## Part A — teratts-tts server

### Task A1: English G2P module with ARPAbet->RU phoneme map

**Files:**
- Create: `AI/infrastructure/teratts-tts/english_g2p.py`
- Create: `AI/infrastructure/teratts-tts/test_english_g2p.py`

**Step 1: Write the failing test**

```python
# test_english_g2p.py
from english_g2p import arpa_to_ru_symbols, english_to_ru_symbols


def test_arpa_vowel_stress_mapping():
    # HH AH0 L OW1 -> hard consonants + stressed/unstressed RU vowels + diphthong tail
    assert arpa_to_ru_symbols(["HH", "AH0", "L", "OW1"]) == ["h", "a0", "l", "o1", "u0"]


def test_arpa_consonant_clusters():
    # schedule: S K EH1 JH UH0 L -> JH = d+zh
    assert arpa_to_ru_symbols(["S", "K", "EH1", "JH", "UH0", "L"]) == \
        ["s", "k", "e1", "d", "zh", "u0", "l"]


def test_th_and_w_substitution():
    # thought: TH AO1 T -> TH->s ; weekend W IY1 ... -> W->v
    assert arpa_to_ru_symbols(["TH", "AO1", "T"]) == ["s", "o1", "t"]
    assert arpa_to_ru_symbols(["W", "IY1", "K"]) == ["v", "i1", "k"]


def test_english_word_end_to_end():
    # full word path including g2p_en (needs nltk data installed)
    syms = english_to_ru_symbols("computer")
    assert syms == ["k", "a0", "m", "p", "j", "u1", "t", "e0", "r"]


def test_empty_and_nonalpha():
    assert arpa_to_ru_symbols([]) == []
    assert arpa_to_ru_symbols(["", "  "]) == []
```

**Step 2: Run to verify it fails**

Run: `cd AI/infrastructure/teratts-tts && ./venv/bin/python -m pytest test_english_g2p.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'english_g2p'`.

**Step 3: Write minimal implementation**

Lift the maps and `arpa_to_ru_symbols` verbatim from `scripts/exp_english.py`, add a lazily-initialized `g2p_en` wrapper and a per-word helper.

```python
# english_g2p.py
"""English words -> Russian phoneme symbols for the glados/TeraTTS model.

English g2p_en (ARPAbet) is mapped to the model's 62 RU phoneme symbols so the
Russian VITS model can pronounce English words in a Russian accent. No retrain.
"""
from __future__ import annotations

_g2p = None


def _get_g2p():
    global _g2p
    if _g2p is None:
        from g2p_en import G2p
        _g2p = G2p()
    return _g2p


# ARPAbet (stress digit stripped) -> RU vowel core (stress attached from ARPAbet digit)
ARPA_VOWEL = {
    "AA": "a", "AE": "a", "AH": "a", "AO": "o",
    "EH": "e", "IH": "i", "IY": "i", "UH": "u", "UW": "u",
    "AW": "a", "AY": "a", "EY": "e", "OW": "o", "OY": "o", "ER": "e",
}
# diphthong glide tail appended after the vowel core
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
    """Convert ARPAbet phones (e.g. ['HH','AH0','L','OW1']) to model RU symbols."""
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


def english_to_ru_symbols(text):
    """Full path: English text -> ARPAbet (g2p_en) -> RU phoneme symbols.

    Word separators (g2p_en emits ' ' between words) become a literal ' '
    symbol so the tokenizer inserts a word gap.
    """
    arpa = _get_g2p()(text)
    syms = []
    for tok in arpa:
        if tok == " ":
            syms.append(" ")
        else:
            syms.extend(arpa_to_ru_symbols([tok]))
    return syms
```

**Step 4: Run to verify it passes**

Run: `./venv/bin/python -m pytest test_english_g2p.py -v`
Expected: PASS (5 tests). If `test_english_word_end_to_end` errors on NLTK data, run Task A0-nltk note below first.

> **A0 NLTK note:** g2p_en needs nltk data. Ensure installed once:
> `./venv/bin/python -c "import nltk; [nltk.download(r, quiet=True) for r in ['averaged_perceptron_tagger','averaged_perceptron_tagger_eng','cmudict','punkt','punkt_tab']]"`

**Step 5: Commit**

```bash
git add english_g2p.py test_english_g2p.py
git commit -m "Add English ARPAbet to Russian phoneme mapping module"
```

---

### Task A2: Mixed-language phoneme assembly in the tokenizer path

**Files:**
- Modify: `AI/infrastructure/teratts-tts/english_g2p.py` (add `split_by_script`)
- Modify: `AI/infrastructure/teratts-tts/test_english_g2p.py`

**Step 1: Write the failing test**

```python
from english_g2p import split_by_script


def test_split_pure_cyrillic():
    assert split_by_script("привет мир") == [("привет мир", "ru")]


def test_split_pure_latin():
    assert split_by_script("hello world") == [("hello world", "en")]


def test_split_mixed_attaches_spaces_and_punct():
    # "Я открыл Google и meeting" -> ru / en / ru / en runs; spaces stick to current run
    out = split_by_script("Я открыл Google и meeting")
    assert out == [("Я открыл ", "ru"), ("Google ", "en"), ("и ", "ru"), ("meeting", "en")]


def test_split_digits_attach():
    assert split_by_script("версия 5 build") == [("версия 5 ", "ru"), ("build", "en")]
```

**Step 2: Run to verify it fails**

Run: `./venv/bin/python -m pytest test_english_g2p.py::test_split_mixed_attaches_spaces_and_punct -v`
Expected: FAIL — `ImportError: cannot import name 'split_by_script'`.

**Step 3: Write minimal implementation**

Mirror the orchestrator's `segmentByLanguage` semantics (Cyrillic 0x0400-0x04FF, Latin A-Za-z; non-alpha attaches to current run; merge adjacent same-lang). Default lang for a leading non-alpha run is `ru` (server is Russian-first).

```python
def _is_cyrillic(ch):
    return "\u0400" <= ch <= "\u04FF"


def _is_latin(ch):
    return ("A" <= ch <= "Z") or ("a" <= ch <= "z")


def split_by_script(text):
    """Split text into contiguous (run_text, 'ru'|'en') segments.

    Non-alphabetic chars attach to the current run. Adjacent same-lang runs are
    merged. A leading non-alpha run defaults to 'ru' (Russian-first server)."""
    if not text:
        return []
    segments = []
    cur_lang = "ru"
    cur = ""
    for ch in text:
        if _is_cyrillic(ch):
            if cur_lang != "ru" and cur:
                segments.append((cur, cur_lang)); cur = ""
            cur_lang = "ru"; cur += ch
        elif _is_latin(ch):
            if cur_lang != "en" and cur:
                segments.append((cur, cur_lang)); cur = ""
            cur_lang = "en"; cur += ch
        else:
            cur += ch
    if cur:
        segments.append((cur, cur_lang))
    merged = []
    for seg in segments:
        if merged and merged[-1][1] == seg[1]:
            merged[-1] = (merged[-1][0] + seg[0], seg[1])
        else:
            merged.append(seg)
    return merged
```

**Step 4: Run to verify it passes**

Run: `./venv/bin/python -m pytest test_english_g2p.py -v`
Expected: PASS (9 tests).

**Step 5: Commit**

```bash
git add english_g2p.py test_english_g2p.py
git commit -m "Add script-based segmentation for mixed RU/EN text"
```

---

### Task A3: Wire mixed phonetic synthesis into server.py

**Files:**
- Modify: `AI/infrastructure/teratts-tts/server.py`
- Modify: `AI/infrastructure/teratts-tts/requirements.txt`

**Step 1: Write the failing test (integration-style, against the synth helper)**

Add `test_server_synth.py` that loads the model once and asserts a mixed string produces non-empty audio and that English-only input differs from the naive translit path.

```python
# test_server_synth.py
import numpy as np
import server


def setup_module(module):
    server.load_models()  # see Step 3: factor model loading out of main()


def test_mixed_returns_audio():
    audio = server.synth_text("Я открыл Google и нашёл meeting")
    assert isinstance(audio, np.ndarray) and audio.size > 0


def test_english_only_returns_audio():
    audio = server.synth_text("hello computer")
    assert isinstance(audio, np.ndarray) and audio.size > 0
```

**Step 2: Run to verify it fails**

Run: `./venv/bin/python -m pytest test_server_synth.py -v`
Expected: FAIL — `AttributeError: module 'server' has no attribute 'load_models'` / `synth_text`.

**Step 3: Implement**

In `server.py`:
1. Add `g2p_en` import path via `english_g2p`.
2. Factor model loading out of `main()` into `load_models()` (sets globals `tts`, `accentizer`, `has_ffmpeg`); call it from `main()`.
3. Add `synth_text(text)` that builds the combined RU phoneme-symbol list and runs the ONNX model once. Reuse the proven helpers from `scripts/exp_english.py` (`symbols_to_ids`, intersperse, model.run with scales). Cyrillic runs go through `accentizer.process_all` + `tts.tokenizer.g2p(run)[1]` (the phoneme strings), Latin runs through `english_g2p.english_to_ru_symbols`.
4. Rewrite `handle_tts` to call `synth_text` instead of `tts(processed, ...)`.

```python
import english_g2p

def synth_text(text, length_scale=LENGTH_SCALE):
    pim = tts.tokenizer.config["phoneme_id_map"]
    symbols = []
    for run, lang in english_g2p.split_by_script(text):
        if lang == "en":
            try:
                symbols.extend(english_g2p.english_to_ru_symbols(run))
            except Exception as e:
                logger.warning("English g2p failed for %r: %s -- accent fallback", run[:50], e)
                symbols.extend(_ru_symbols(run))
        else:
            symbols.extend(_ru_symbols(run))
    ids = []
    ids.extend(pim["^"]); ids.extend(pim["_"])
    for s in symbols:
        if s in pim:
            ids.extend(pim[s]); ids.extend(pim["_"])
    ids.extend(pim["$"])
    ids = tts._intersperse(ids, 0)
    arr = np.expand_dims(np.array(ids, dtype=np.int64), 0)
    lengths = np.array([arr.shape[1]], dtype=np.int64)
    scales = np.array([0.667, length_scale, 0.8], dtype=np.float32)
    audio = tts.model.run(None, {"input": arr, "input_lengths": lengths,
                                 "scales": scales, "sid": None})[0][0, 0][0]
    return tts._add_silent(audio, silence_duration=tts.add_time_to_end)


def _ru_symbols(run):
    try:
        run = accentizer.process_all(run)
    except Exception as e:
        logger.warning("RUAccent failed for %r: %s", run[:50], e)
    return tts.tokenizer.g2p(run)[1]  # phoneme strings (index 1), not ids
```

`handle_tts` body becomes: `audio = synth_text(text)` then existing `audio_to_wav` / Opus logic.

Add to `requirements.txt`: `g2p_en` and `nltk`.

**Step 4: Run to verify it passes**

Run: `./venv/bin/python -m pytest test_server_synth.py -v`
Expected: PASS (2 tests).

**Step 5: Commit**

```bash
git add server.py requirements.txt test_server_synth.py
git commit -m "Synthesize mixed RU/EN text via combined phoneme assembly"
```

---

### Task A4: Mixed-string audio smoke + Telegram bounce (manual verification)

**Files:**
- Create: `AI/infrastructure/teratts-tts/scripts/exp_mixed.py` (extends exp_english.py to mixed strings)

**Step 1:** Write `exp_mixed.py` synthesizing: "Я открыл Google и нашёл meeting.", "Скачай файл с GitHub завтра.", "My schedule is busy сегодня." via `server.synth_text` after `server.load_models()`. Save WAVs to `scripts/out/`.

**Step 2:** Run it:
`./venv/bin/python scripts/exp_mixed.py`
Expected: 3 WAVs, no exceptions, console prints the RU symbol list per string.

**Step 3:** Concatenate to one mp3 and bounce to Telegram (per user CLAUDE.md), quote the shortId in the report. Use `mcp__plugin_context-mode_context-mode__ctx_execute` for the curl (Bash curl is blocked).

**Step 4:** STOP and show the user the recording shortId for confirmation before proceeding to Part B deploy. (This is the "show intermediary results" checkpoint.)

**Step 5: Commit**

```bash
git add scripts/exp_mixed.py
git commit -m "Add mixed RU/EN synthesis smoke script"
```

---

### Task A5: Bake g2p_en + NLTK data into the Docker image

**Files:**
- Modify: `AI/infrastructure/teratts-tts/fetch_models.sh`
- Modify: `AI/infrastructure/teratts-tts/Dockerfile`

**Step 1:** In `fetch_models.sh`, after the RUAccent prefetch, add:

```bash
echo "[teratts-tts] Pre-downloading g2p_en NLTK data"
python -c "import nltk; [nltk.download(r, quiet=True) for r in ['averaged_perceptron_tagger','averaged_perceptron_tagger_eng','cmudict','punkt','punkt_tab']]"
python -c "from g2p_en import G2p; G2p()"
```

**Step 2:** Confirm the Dockerfile runs `fetch_models.sh` at build time (it already pre-downloads the model). If it pip-installs from `requirements.txt`, `g2p_en`+`nltk` are now included. Ensure NLTK data dir is on the image (default `~/.nltk_data`) and readable by the runtime user.

**Step 3:** Verify locally that a clean `./venv/bin/python -c "from g2p_en import G2p; G2p()('test')"` works offline after fetch (no network).
Expected: prints ARPAbet, no download attempts.

**Step 4: Commit**

```bash
git add fetch_models.sh Dockerfile
git commit -m "Bake g2p_en and NLTK data into the image"
```

> **Do NOT push yet** — push triggers auto-deploy. Hold until the user approves the Telegram smoke result from A4 and the orchestrator side (Part B) is ready, so the two deploy together cleanly. The default routing mode keeps Kokoro for English, so order is not behavior-critical, but coordinate the push with the user.

---

## Part B — orchestrator routing flag

### Task B1: Add `ttsRoutingMode` config

**Files:**
- Modify: `AI/orchestrator/src/config.js`
- Modify: `AI/orchestrator/.env.example`

**Step 1: Write the failing test**

```javascript
// AI/orchestrator/test/config-tts-mode.test.mjs  (node:test)
import { test } from 'node:test';
import assert from 'node:assert';

test('ttsRoutingMode defaults to language-split', async () => {
  delete process.env.TTS_ROUTING_MODE;
  const { default: config } = await import('../src/config.js?cfg1');
  assert.equal(config.ttsRoutingMode, 'language-split');
});

test('ttsRoutingMode honors env teratts', async () => {
  process.env.TTS_ROUTING_MODE = 'teratts';
  const { default: config } = await import('../src/config.js?cfg2');
  assert.equal(config.ttsRoutingMode, 'teratts');
});
```

**Step 2: Run to verify it fails**

Run: `cd AI/orchestrator && node --test test/config-tts-mode.test.mjs`
Expected: FAIL — `config.ttsRoutingMode` is undefined.

**Step 3: Implement**

In `src/config.js` schema add:
```javascript
ttsRoutingMode: Joi.string().valid('language-split', 'teratts').default('language-split'),
```
and in the values object:
```javascript
ttsRoutingMode: process.env.TTS_ROUTING_MODE,
```

In `.env.example` add:
```
# TTS routing: 'language-split' (default) detects EN/RU and routes EN->Kokoro,
# RU->TeraTTS, mixed->concat. 'teratts' sends everything to TeraTTS (which
# pronounces English via Russian phonemes).
TTS_ROUTING_MODE=language-split
```

**Step 4: Run to verify it passes**

Run: `node --test test/config-tts-mode.test.mjs`
Expected: PASS (2 tests).

**Step 5: Commit**

```bash
git add src/config.js .env.example test/config-tts-mode.test.mjs
git commit -m "Add TTS_ROUTING_MODE config flag"
```

---

### Task B2: Gate routing in tts.js on the mode

**Files:**
- Modify: `AI/orchestrator/src/tts.js`
- Test: `AI/orchestrator/test/tts-routing.test.mjs`

**Step 1: Write the failing test**

Mock `fetch` to record which TTS endpoint URL is hit. Assert: in `teratts` mode, a pure-English string hits `piperTtsUrl` (TeraTTS), NOT `ttsUrl` (Kokoro); in `language-split` mode the same string hits `ttsUrl` (Kokoro).

```javascript
// test/tts-routing.test.mjs
import { test } from 'node:test';
import assert from 'node:assert';

function installFetchSpy() {
  const hits = [];
  globalThis.fetch = async (url) => {
    hits.push(String(url));
    // minimal WAV so downstream code doesn't crash
    const wav = Buffer.alloc(44); wav.write('RIFF', 0); wav.write('WAVE', 8);
    return { ok: true, headers: { get: () => 'audio/wav' },
             arrayBuffer: async () => wav.buffer.slice(0, 44) };
  };
  return hits;
}

test('teratts mode routes English to TeraTTS not Kokoro', async () => {
  process.env.TTS_ROUTING_MODE = 'teratts';
  const hits = installFetchSpy();
  const { generateAudio } = await import('../src/tts.js?t1');
  await generateAudio('hello world');
  assert.ok(hits.some(u => u.includes('10018')), `expected TeraTTS hit, got ${hits}`);
  assert.ok(!hits.some(u => u.includes('10007')), `must not hit Kokoro, got ${hits}`);
});

test('language-split mode routes English to Kokoro', async () => {
  process.env.TTS_ROUTING_MODE = 'language-split';
  const hits = installFetchSpy();
  const { generateAudio } = await import('../src/tts.js?t2');
  await generateAudio('hello world');
  assert.ok(hits.some(u => u.includes('10007')), `expected Kokoro hit, got ${hits}`);
});
```

(Adjust port assertions if `.env`/config resolves different URLs in the test env; the intent is Kokoro=`ttsUrl` vs TeraTTS=`piperTtsUrl`.)

**Step 2: Run to verify it fails**

Run: `node --test test/tts-routing.test.mjs`
Expected: FAIL — in `teratts` mode English still routes to Kokoro.

**Step 3: Implement**

At top of `tts.js`, `import config from './config.js';` already present. Add a guard at the start of the three entry points:

- `generateAudio(text)`: if `config.ttsRoutingMode === 'teratts'`, `return generateAudioPiper(text);` (TeraTTS) before any `segmentByLanguage`.
- `generateNotifAudio(text)`: if `teratts`, `text = stripUrlsToDomain(text); return generateAudioPiperOpus(text);`.
- `streamTts` inner loop: if `teratts`, force the non-English branch — `const audioBuffer = await generateAudioPiperOpus(sentences[i]);` and skip the `isEnglish`/`segmentByLanguage` checks.

Leave all existing functions (`segmentByLanguage`, Kokoro fns, concat helpers) intact for `language-split`.

**Step 4: Run to verify it passes**

Run: `node --test test/tts-routing.test.mjs`
Expected: PASS (2 tests).

**Step 5: Commit**

```bash
git add src/tts.js test/tts-routing.test.mjs
git commit -m "Gate TTS language routing on TTS_ROUTING_MODE"
```

---

### Task B3: Document the flag in orchestrator CLAUDE.md

**Files:**
- Modify: `AI/orchestrator/CLAUDE.md`

**Step 1:** Under the Configuration section (near the peer-service endpoints bullet), add a short subsection:

```markdown
### TTS routing (`TTS_ROUTING_MODE`)

Controls how `src/tts.js` chooses a TTS engine:

- `language-split` (default) — detects Cyrillic vs. Latin per segment and routes
  English -> Kokoro (`TTS_URL`), Russian -> TeraTTS (`PIPER_TTS_URL`), mixed text
  -> per-segment synthesis concatenated into one WAV.
- `teratts` — bypasses language detection; all text goes to TeraTTS
  (`PIPER_TTS_URL`), which pronounces English words via Russian phonemes
  (English g2p -> Russian phoneme symbols, no model retrain). Kokoro is left
  unused in this mode.

The Kokoro/language-split code path is retained; switching the flag back to
`language-split` restores it with no code change.
```

**Step 2:** No test (docs). Verify it renders / reads correctly.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "Document TTS_ROUTING_MODE in orchestrator CLAUDE.md"
```

---

### Task B4: Push + deploy coordination (gated on user approval)

**Steps:**
1. Confirm the A4 Telegram smoke was approved by the user.
2. Push `teratts-tts` `main` (auto-build + deploy).
3. Push `orchestrator` `main` (auto-build + deploy). Default mode is
   `language-split`, so behavior is unchanged until the env flips.
4. To actually enable: set `TTS_ROUTING_MODE=teratts` in the orchestrator
   deployment manifest in `repository-deployment-catalog` (NOT via `kubectl`),
   commit + push that repo, let Flux reconcile. Per memory: never push the
   deploy repo to bump images; only the env-var change here, and only after the
   service images are live.
5. STOP before any push and get explicit user go-ahead (services are shared).

---

## Notes for the executor

- teratts venv is uv-managed: `export UV_CACHE_DIR=/media/varingait/Lobotomite/.uv-cache` before any `uv` op.
- Installed TeraTTS lib uses misspelled kwarg `lenght_scale` — keep it.
- Bash `curl`/`wget` are blocked; use `mcp__plugin_context-mode_context-mode__ctx_execute` for the Telegram bounce.
- Never restart services; deploy is via git push (auto). Get user approval before pushing (shared infra).
- Run Node tests with `node --test`; Python tests with `./venv/bin/python -m pytest`.
```
