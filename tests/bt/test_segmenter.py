import numpy as np

from telefony_app.bt.segmenter import Segmenter

RATE = 16000


def speech(ms: int) -> np.ndarray:
    t = np.arange(RATE * ms // 1000) / RATE
    return (np.sin(2 * np.pi * 300 * t) * 8000).astype(np.int16)


def silence(ms: int) -> np.ndarray:
    return np.zeros(RATE * ms // 1000, dtype=np.int16)


def make() -> Segmenter:
    return Segmenter(rate=RATE, min_speech_ms=200, end_silence_ms=600)


def duration_ms(pcm: np.ndarray) -> int:
    return len(pcm) * 1000 // RATE


def test_utterance_emitted_after_trailing_silence():
    seg = make()
    assert seg.feed(silence(300)) == []
    assert seg.feed(speech(800)) == []
    out = seg.feed(silence(700))
    assert len(out) == 1
    assert 800 <= duration_ms(out[0]) <= 1500


def test_short_pause_does_not_split():
    out = make().feed(np.concatenate([speech(500), silence(300), speech(500), silence(700)]))
    assert len(out) == 1


def test_two_utterances_split_by_long_silence():
    out = make().feed(np.concatenate([speech(500), silence(700), speech(500), silence(700)]))
    assert len(out) == 2


def test_click_shorter_than_min_speech_is_dropped():
    assert make().feed(np.concatenate([speech(60), silence(700)])) == []


def test_flush_returns_unterminated_speech_once():
    seg = make()
    seg.feed(speech(500))
    assert len(seg.flush()) == 1
    assert seg.flush() == []
