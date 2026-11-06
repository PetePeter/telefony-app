"""Rolling transcription: while the caller talks, their utterance is transcribed in overlapping windows,
so at the pause only the last few seconds are left for STT. The overlap is merged here, in code."""

import re

import numpy as np

WINDOW_S = 3.0
OVERLAP_S = 1.0
_MAX_OVERLAP_WORDS = 8


def _norm(word: str) -> str:
    return re.sub(r"[^\w']", "", word).lower()


def merge_overlap(before: str, after: str) -> str:
    """Join two transcripts whose audio overlapped: drop the words at the start of `after` that end `before`."""
    old, new = before.split(), after.split()
    for k in range(min(len(old), len(new), _MAX_OVERLAP_WORDS), 0, -1):
        if [_norm(w) for w in old[-k:]] == [_norm(w) for w in new[:k]]:
            new = new[k:]
            break
    return " ".join(old + new)


class RollingTranscript:
    """Which window of a growing utterance to transcribe next, and the merged text so far."""

    def __init__(self, rate: int, window_s: float = WINDOW_S, overlap_s: float = OVERLAP_S) -> None:
        self._rate = rate
        self._window = int(rate * window_s)
        self._step = int(rate * (window_s - overlap_s))
        self._done = 0  # windows handed out
        self.text = ""

    def next_window(self, audio: np.ndarray) -> np.ndarray | None:
        """The next full window of the utterance so far, or None until there is one."""
        start = self._done * self._step
        if len(audio) < start + self._window:
            return None
        self._done += 1
        return audio[start : start + self._window]

    @property
    def covered_s(self) -> float:
        """Where the untranscribed tail starts, in seconds: the last window's overlap."""
        return self._done * self._step / self._rate

    def tail(self, audio: np.ndarray) -> np.ndarray:
        """What the windows have not covered yet, from the last overlap to the end."""
        return audio[self._done * self._step :]

    def add(self, text: str) -> str:
        self.text = merge_overlap(self.text, text) if text else self.text
        return self.text
