import numpy as np
import pytest

from telefony_app.bt.dtmf import DtmfDecoder

RATE = 8000
KEYS = {
    "1": (697, 1209), "2": (697, 1336), "3": (697, 1477),
    "4": (770, 1209), "5": (770, 1336), "6": (770, 1477),
    "7": (852, 1209), "8": (852, 1336), "9": (852, 1477),
    "*": (941, 1209), "0": (941, 1336), "#": (941, 1477),
}  # fmt: skip


def tone(key: str, ms: int, amplitude: float = 6000) -> np.ndarray:
    t = np.arange(RATE * ms // 1000) / RATE
    low, high = KEYS[key]
    return amplitude * (np.sin(2 * np.pi * low * t) + np.sin(2 * np.pi * high * t))


def gap(ms: int) -> np.ndarray:
    return np.zeros(RATE * ms // 1000)


def vowel(ms: int) -> np.ndarray:
    """A voiced vowel: 140 Hz pitch with falling harmonics up to 3.4 kHz, which cross the DTMF bands."""
    t = np.arange(RATE * ms // 1000) / RATE
    return np.sum([9000 / k * np.sin(2 * np.pi * 140 * k * t) for k in range(1, 25)], axis=0)


def decode(*parts: np.ndarray, packet: int = 60) -> tuple[str, np.ndarray]:
    """Feed audio in SCO-sized packets, as the line does."""
    pcm = np.clip(np.concatenate(parts), -32768, 32767).astype(np.int16)
    decoder, digits, out = DtmfDecoder(RATE), "", []
    for start in range(0, len(pcm), packet):
        found, kept = decoder.feed(pcm[start : start + packet])
        digits += found
        out.append(kept)
    return digits, np.concatenate(out)


@pytest.mark.parametrize("key", list(KEYS))
def test_each_key_is_decoded(key: str):
    assert decode(gap(30), tone(key, 100), gap(60))[0] == key


def test_sequence_is_decoded_in_order():
    assert decode(tone("1", 80), gap(60), tone("#", 80), gap(60), tone("9", 80), gap(60))[0] == "1#9"


def test_held_key_is_one_digit_and_a_repeat_needs_a_gap():
    assert decode(tone("5", 1500), gap(60))[0] == "5"
    assert decode(tone("1", 80), gap(60), tone("1", 80), gap(60))[0] == "11"


def test_no_false_hits():
    rng = np.random.default_rng(1)
    t = np.arange(RATE) / RATE
    single = 8000 * np.sin(2 * np.pi * 697 * t)
    for audio in (gap(1000), rng.normal(0, 3000, RATE), single, vowel(1000)):
        assert decode(audio)[0] == ""


def test_too_short_burst_is_ignored():
    assert decode(gap(30), tone("3", 20), gap(60))[0] == ""


def test_survives_quiet_tone_and_line_noise():
    rng = np.random.default_rng(2)
    quiet = tone("8", 100, amplitude=1000)
    noisy = quiet + rng.normal(0, np.sqrt(np.mean(quiet**2)) / 5.6, len(quiet))  # ~15 dB SNR
    assert decode(gap(30), noisy, gap(60))[0] == "8"


def test_tones_are_muted_so_speech_recognition_never_hears_them():
    digits, kept = decode(tone("7", 200), gap(60))
    assert digits == "7"
    assert np.sqrt(np.mean(kept.astype(np.float64) ** 2)) < 500  # below the segmenter's speech threshold


def test_speech_passes_through_untouched():
    speech = np.clip(vowel(500), -32768, 32767).astype(np.int16)
    digits, kept = decode(speech)
    assert digits == ""
    assert np.array_equal(kept, speech[: len(kept)]) and len(speech) - len(kept) < 2 * 205  # one block held back, plus a partial one
