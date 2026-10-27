"""Energy-based utterance segmentation of far-end call audio, feeding whole utterances to STT."""

import numpy as np

_WINDOW_MS = 20


class Segmenter:
    def __init__(
        self, rate: int, min_speech_ms: int = 300, end_silence_ms: int = 700, threshold: float = 500.0
    ) -> None:
        self._window = rate * _WINDOW_MS // 1000
        self._min_speech = min_speech_ms // _WINDOW_MS
        self._end_silence = end_silence_ms // _WINDOW_MS
        self._threshold = threshold
        self._pending = np.zeros(0, dtype=np.int16)
        self._utterance: list[np.ndarray] = []
        self._speech_windows = 0
        self._silent_windows = 0

    def feed(self, pcm: np.ndarray) -> list[np.ndarray]:
        self._pending = np.concatenate([self._pending, pcm.astype(np.int16)])
        done = []
        while len(self._pending) >= self._window:
            window, self._pending = self._pending[: self._window], self._pending[self._window :]
            if self._is_speech(window):
                self._utterance.append(window)
                self._speech_windows += 1
                self._silent_windows = 0
            elif self._utterance:
                self._utterance.append(window)
                self._silent_windows += 1
                if self._silent_windows >= self._end_silence:
                    done.extend(self._close())
        return done

    def flush(self) -> list[np.ndarray]:
        return self._close()

    def _is_speech(self, window: np.ndarray) -> bool:
        return float(np.sqrt(np.mean(window.astype(np.float64) ** 2))) >= self._threshold

    def _close(self) -> list[np.ndarray]:
        done = [np.concatenate(self._utterance)] if self._speech_windows >= self._min_speech else []
        self._utterance, self._speech_windows, self._silent_windows = [], 0, 0
        return done
