"""DTMF key presses in far-end call audio: decoded into digits, and muted so speech recognition never hears them."""

import numpy as np

_ROWS = (697, 770, 852, 941)
_COLUMNS = (1209, 1336, 1477)
_KEYS = ("123", "456", "789", "*0#")
_BLOCK_S = 205 / 8000  # the classic 205-sample block at 8 kHz: ~39 Hz bins, finer than the 73 Hz row spacing
_MIN_RMS = 100.0  # about -50 dBFS
_MIN_TONE_SHARE = 0.6  # of the block's energy in the two strongest DTMF bins; speech and noise spread wider
_MAX_TWIST = 10.0  # power ratio allowed between the row and column tones


class DtmfDecoder:
    """Feed downlink PCM; get back the digits pressed and the audio with tones zeroed, delayed by two blocks.

    A key counts once it is heard in two blocks in a row (~51 ms) and again only after a block without it.
    """

    def __init__(self, rate: int) -> None:
        self._block = round(rate * _BLOCK_S)
        n = np.arange(self._block)
        self._basis = np.exp(-2j * np.pi * np.outer(n, (*_ROWS, *_COLUMNS)) / rate)
        self._pending = np.zeros(0, dtype=np.int16)
        self._held: np.ndarray | None = None  # the last block, kept back until its successor is classified
        self._tones = [False, False]  # whether the block before the held one, and the held one, carry a tone
        self._previous: str | None = None
        self._pressed: str | None = None

    def feed(self, pcm: np.ndarray) -> tuple[str, np.ndarray]:
        self._pending = np.concatenate([self._pending, pcm.astype(np.int16)])
        digits, out = "", []
        while len(self._pending) >= self._block:
            block, self._pending = self._pending[: self._block], self._pending[self._block :]
            key = self._classify(block)
            if key != self._pressed:
                self._pressed = None  # released
            if key is not None and key == self._previous and self._pressed is None:
                digits += key
                self._pressed = key
            self._previous = key
            if self._held is not None:
                near_tone = self._tones[0] or self._tones[1] or key is not None
                out.append(np.zeros_like(self._held) if near_tone else self._held)
            self._held, self._tones = block, [self._tones[1], key is not None]
        return digits, np.concatenate(out) if out else np.zeros(0, dtype=np.int16)

    def _classify(self, block: np.ndarray) -> str | None:
        x = block.astype(np.float64)
        energy = float(np.dot(x, x))
        if energy < _MIN_RMS**2 * len(x):
            return None
        power = np.abs(x @ self._basis) ** 2
        row, column = int(np.argmax(power[:4])), int(np.argmax(power[4:]))
        low, high = power[row], power[4 + column]
        share = 2 * (low + high) / (len(x) * energy)  # 1.0 for a pure two-tone block
        if share < _MIN_TONE_SHARE or not 1 / _MAX_TWIST <= high / low <= _MAX_TWIST:
            return None
        return _KEYS[row][column]
