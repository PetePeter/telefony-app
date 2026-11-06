"""Energy-based utterance segmentation of far-end call audio, feeding whole utterances to STT."""

from collections.abc import Callable

import numpy as np

_WINDOW_MS = 20
_SPEECH_ONSET_MS = 220  # DTMF classification buffers up to about 50 ms before VAD sees the audio.


class Segmenter:
    def __init__(
        self,
        rate: int,
        min_speech_ms: int = 220,
        end_silence_ms: int = 1500,  # a thinking pause stays in the turn; rolling windows keep the reply quick
        threshold: float = 500.0,
        on_speech_start: Callable[[], None] | None = None,
    ) -> None:
        self._window = rate * _WINDOW_MS // 1000
        self._min_speech = (min_speech_ms + _WINDOW_MS - 1) // _WINDOW_MS
        self._end_silence = end_silence_ms // _WINDOW_MS
        self._threshold = threshold
        self._on_speech_start = on_speech_start
        self._pending = np.zeros(0, dtype=np.int16)
        self._utterance: list[np.ndarray] = []
        self._speech_windows = 0
        self._onset_windows = 0
        self._onset_reported = False
        self._silent_windows = 0

    def feed(self, pcm: np.ndarray) -> list[np.ndarray]:
        self._pending = np.concatenate([self._pending, pcm.astype(np.int16)])
        done = []
        while len(self._pending) >= self._window:
            window, self._pending = self._pending[: self._window], self._pending[self._window :]
            if self._is_speech(window):
                self._utterance.append(window)
                self._speech_windows += 1
                self._onset_windows += 1
                self._silent_windows = 0
                if not self._onset_reported and self._onset_windows * _WINDOW_MS >= _SPEECH_ONSET_MS:
                    self._onset_reported = True
                    if self._on_speech_start:
                        self._on_speech_start()
            else:
                self._onset_windows = 0
                if self._utterance:
                    self._utterance.append(window)
                    self._silent_windows += 1
                    if self._silent_windows >= self._end_silence:
                        done.extend(self._close())
        return done

    @property
    def utterance_samples(self) -> int:
        return len(self._utterance) * self._window

    @property
    def in_progress(self) -> np.ndarray:
        """The utterance heard so far, before it has ended."""
        return np.concatenate(self._utterance) if self._utterance else np.zeros(0, dtype=np.int16)

    def flush(self) -> list[np.ndarray]:
        return self._close()

    def _is_speech(self, window: np.ndarray) -> bool:
        return float(np.sqrt(np.mean(window.astype(np.float64) ** 2))) >= self._threshold

    def _close(self) -> list[np.ndarray]:
        done = [np.concatenate(self._utterance)] if self._speech_windows >= self._min_speech else []
        self._utterance = []
        self._speech_windows = self._onset_windows = self._silent_windows = 0
        self._onset_reported = False
        return done
