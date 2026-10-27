import sys

import pytest

from telefony_app.voice import SapiTts, make_tts


@pytest.mark.skipif(sys.platform != "win32", reason="SAPI is Windows-only")
async def test_sapi_speaks_at_the_link_rate():
    pcm = await SapiTts().synthesize("hello", 8000)
    assert 0.2 < len(pcm) / 8000 < 3  # about half a second of speech at 8 kHz
    assert abs(pcm).max() > 1000


def test_auto_without_piper_model_falls_back_past_piper():
    tts = make_tts("auto", piper_model=None)
    assert tts is None or isinstance(tts, SapiTts)


def test_none_means_no_speech():
    assert make_tts("none", "model.onnx") is None
