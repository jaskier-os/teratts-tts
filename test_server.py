#!/usr/bin/env python3
"""Tests for TeraTTS server -- starts the server, runs requests, verifies behavior."""

import asyncio
import io
import os
import struct
import time
import wave

import aiohttp


BASE_URL = os.environ.get("TTS_TEST_URL", "http://localhost:10018")
TIMEOUT = aiohttp.ClientTimeout(total=60)


async def test_health():
    """Test /health endpoint."""
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        async with session.get(f"{BASE_URL}/health") as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
            data = await resp.json()
            assert data["status"] == "ok", f"Unexpected health response: {data}"
    print("[PASS] health check")


async def test_empty_text():
    """Test that empty text returns 400."""
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        async with session.post(f"{BASE_URL}/", data="   ") as resp:
            assert resp.status == 400, f"Expected 400 for empty text, got {resp.status}"
    print("[PASS] empty text returns 400")


async def test_wav_output():
    """Test WAV output for a short sentence."""
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        async with session.post(
            f"{BASE_URL}/",
            data="Привет мир!",
            headers={"Content-Type": "text/plain; charset=utf-8"},
        ) as resp:
            assert resp.status == 200, f"WAV request failed: {resp.status}"
            ct = resp.headers.get("Content-Type", "")
            assert "wav" in ct, f"Expected audio/wav, got {ct}"
            body = await resp.read()
            assert body[:4] == b"RIFF", "Response is not a valid WAV file"
            # Verify WAV properties
            buf = io.BytesIO(body)
            with wave.open(buf, "rb") as wf:
                assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()} channels"
                assert wf.getframerate() == 22050, f"Expected 22050 Hz, got {wf.getframerate()}"
                assert wf.getsampwidth() == 2, f"Expected 16-bit, got {wf.getsampwidth() * 8}-bit"
                duration = wf.getnframes() / wf.getframerate()
                assert duration > 0.5, f"Audio too short: {duration:.2f}s"
    print(f"[PASS] WAV output ({duration:.1f}s, {len(body)} bytes)")


async def test_opus_output():
    """Test Opus output when Accept header requests it."""
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        async with session.post(
            f"{BASE_URL}/",
            data="Тестирование формата опус.",
            headers={
                "Content-Type": "text/plain; charset=utf-8",
                "Accept": "audio/opus, audio/ogg",
            },
        ) as resp:
            assert resp.status == 200, f"Opus request failed: {resp.status}"
            ct = resp.headers.get("Content-Type", "")
            assert "ogg" in ct, f"Expected audio/ogg, got {ct}"
            body = await resp.read()
            assert body[:4] == b"OggS", f"Response is not a valid OGG file (got {body[:4]})"
            assert len(body) > 100, f"Opus output too small: {len(body)} bytes"
    print(f"[PASS] Opus output ({len(body)} bytes)")


async def test_single_ogg_for_text_with_periods():
    """Test that text containing periods produces a single OGG stream, not concatenated ones."""
    text = "Нужно обновить версию до 3.5 и проверить совместимость с т.д."
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        async with session.post(
            f"{BASE_URL}/",
            data=text,
            headers={
                "Content-Type": "text/plain; charset=utf-8",
                "Accept": "audio/opus, audio/ogg",
            },
        ) as resp:
            assert resp.status == 200, f"Request failed: {resp.status}"
            body = await resp.read()
            assert body[:4] == b"OggS", "Not a valid OGG file"
            # Count BOS pages (byte at offset 5 has bit 0x02 set for beginning-of-stream)
            bos_count = 0
            pos = 0
            while True:
                pos = body.find(b"OggS", pos)
                if pos == -1:
                    break
                if pos + 5 < len(body) and (body[pos + 5] & 0x02):
                    bos_count += 1
                pos += 4
            assert bos_count == 1, f"Expected 1 OGG stream (1 BOS page), got {bos_count} -- concatenated streams detected!"
    print(f"[PASS] single OGG stream for text with periods ({len(body)} bytes, {bos_count} BOS page)")


async def test_performance():
    """Measure generation speed for a typical sentence."""
    text = "Искусственный интеллект стремительно развивается и меняет нашу повседневную жизнь."
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        t0 = time.time()
        async with session.post(
            f"{BASE_URL}/",
            data=text,
            headers={"Content-Type": "text/plain; charset=utf-8"},
        ) as resp:
            body = await resp.read()
        elapsed = time.time() - t0
        buf = io.BytesIO(body)
        with wave.open(buf, "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()
        rtf = elapsed / duration
        assert rtf < 1.0, f"Not realtime: RTF={rtf:.3f} (generation {elapsed:.2f}s for {duration:.1f}s audio)"
    print(f"[PASS] performance: {duration:.1f}s audio in {elapsed:.2f}s (RTF: {rtf:.3f})")


async def main():
    print(f"Testing TeraTTS server at {BASE_URL}\n")
    await test_health()
    await test_empty_text()
    await test_wav_output()
    await test_opus_output()
    await test_single_ogg_for_text_with_periods()
    await test_performance()
    print("\nAll tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
