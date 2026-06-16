import numpy as np
import server


def setup_module(module):
    server.load_models()


def test_mixed_returns_audio():
    audio = server.synth_text("Я открыл Google и нашёл meeting")
    assert isinstance(audio, np.ndarray) and audio.size > 0


def test_english_only_returns_audio():
    audio = server.synth_text("hello computer")
    assert isinstance(audio, np.ndarray) and audio.size > 0


def test_russian_only_returns_audio():
    audio = server.synth_text("привет мир")
    assert isinstance(audio, np.ndarray) and audio.size > 0
