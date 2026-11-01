"""Write call audio as an unpublished WAV until the call reaches a terminal boundary."""

import contextlib
from pathlib import Path
import struct

import numpy as np

from telefony_app.phone import validate_recording_id

_MIN_ACTIVE_RMS = 200
_MIN_ACTIVE_PEAK = 400
_MAX_GAIN = 10.0
_NORMALISE_WINDOW_MS = 20
_NORMALISE_RAMP_MS = 5
_NORMALISE_BATCH_WINDOWS = 4096


class WavRecorder:
    """Pair receive/transmit PCM into a stereo WAV, published only after its header is complete."""

    def __init__(self, directory: Path, recording_id: str, sample_rate: int) -> None:
        validated_id = validate_recording_id(recording_id)
        assert validated_id is not None
        if sample_rate not in (8000, 16000):
            raise ValueError("SCO sample rate must be 8000 or 16000 Hz")

        self.directory = directory
        self.part_path = directory / f"{validated_id}.wav.part"
        self.final_path = directory / f"{validated_id}.wav"
        self._frame_count = 0
        self._finished = False

        directory.mkdir(parents=True, exist_ok=True)
        if self.final_path.exists():
            raise FileExistsError(self.final_path)
        self._stream = self.part_path.open("xb")
        try:
            self._sample_rate = sample_rate
            self._write_header()
        except BaseException:
            self._stream.close()
            self.part_path.unlink(missing_ok=True)
            raise

    def write_frame(self, received: np.ndarray, transmitted: np.ndarray) -> None:
        if self._finished:
            raise ValueError("recording is already finalized")
        receive = np.asarray(received, dtype=np.int16).reshape(-1)
        transmit = np.asarray(transmitted, dtype=np.int16).reshape(-1)
        count = max(len(receive), len(transmit))
        if count == 0:
            return
        stereo = np.zeros((count, 2), dtype="<i2")
        stereo[: len(receive), 0] = receive
        stereo[: len(transmit), 1] = transmit
        self._stream.write(stereo.tobytes())
        self._frame_count += count

    def finish(self) -> Path | None:
        """Close the WAV and publish it; empty recordings leave no artifact."""
        if self._finished:
            return self.final_path if self.final_path.exists() else None
        try:
            self._write_header()
            self._stream.flush()
            self._stream.close()
            if self._frame_count == 0:
                self.part_path.unlink(missing_ok=True)
                self._finished = True
                return None
            self._normalise_sides()
            if self.final_path.exists():
                raise FileExistsError(self.final_path)
            self.part_path.rename(self.final_path)
            self._finished = True
            return self.final_path
        except BaseException:
            self.abort()
            raise

    def _normalise_sides(self) -> None:
        """Match active receive/transmit RMS in the saved WAV, with bounded gain."""
        window_frames = max(1, self._sample_rate * _NORMALISE_WINDOW_MS // 1000)
        full_windows, tail_frames = divmod(self._frame_count, window_frames)
        window_count = full_windows + bool(tail_frames)
        active_windows = np.zeros((window_count, 2), dtype=bool)
        totals = np.zeros(2, dtype=np.float64)
        active_frames = np.zeros(2, dtype=np.int64)
        peaks = np.zeros(2, dtype=np.int32)
        audio = np.memmap(self.part_path, dtype="<i2", mode="r+", offset=44, shape=(self._frame_count, 2))

        def measure(windows: np.ndarray, first_window: int) -> None:
            values = windows.astype(np.float64)
            sums = np.einsum("ijk,ijk->ik", values, values, optimize=True)
            rms = np.sqrt(sums / windows.shape[1])
            peak = np.max(np.abs(windows.astype(np.int32)), axis=1)
            active = (rms >= _MIN_ACTIVE_RMS) & (peak >= _MIN_ACTIVE_PEAK)
            count = windows.shape[0]
            active_windows[first_window:first_window + count] = active
            totals[:] += np.sum(np.where(active, sums, 0.0), axis=0)
            active_frames[:] += np.sum(active, axis=0) * windows.shape[1]
            peaks[:] = np.maximum(peaks, np.max(np.where(active, peak, 0), axis=0))

        for first in range(0, full_windows, _NORMALISE_BATCH_WINDOWS):
            count = min(_NORMALISE_BATCH_WINDOWS, full_windows - first)
            start_frame = first * window_frames
            end_frame = (first + count) * window_frames
            measure(audio[start_frame:end_frame].reshape(count, window_frames, 2), first)
        if tail_frames:
            measure(audio[full_windows * window_frames:].reshape(1, tail_frames, 2), full_windows)

        active = active_frames > 0
        if not np.any(active):
            audio.flush()
            return

        rms = np.zeros(2, dtype=np.float64)
        rms[active] = np.sqrt(totals[active] / active_frames[active])
        target_rms = float(np.mean(rms[active]))
        gains = np.ones(2, dtype=np.float64)
        for side in range(2):
            if active[side]:
                gain = min(target_rms / rms[side], _MAX_GAIN)
                gains[side] = min(gain, 32767.0 / peaks[side])

        prior_active = np.vstack((np.zeros((1, 2), dtype=bool), active_windows[:-1]))
        next_active = np.vstack((active_windows[1:], np.zeros((1, 2), dtype=bool)))
        starts = active_windows & ~prior_active
        ends = active_windows & ~next_active
        ramp_frames = max(1, self._sample_rate * _NORMALISE_RAMP_MS // 1000)

        def apply(windows: np.ndarray, first_window: int) -> None:
            count = windows.shape[0]
            for side, gain in enumerate(gains):
                mask = active_windows[first_window:first_window + count, side]
                if gain == 1.0 or not np.any(mask):
                    continue
                selected = np.flatnonzero(mask)
                source = windows[selected, :, side].astype(np.float64)
                scaled = source * gain
                start_rows = np.flatnonzero(starts[first_window:first_window + count, side])
                end_rows = np.flatnonzero(ends[first_window:first_window + count, side])
                start_rows = np.searchsorted(selected, start_rows)
                end_rows = np.searchsorted(selected, end_rows)
                for row in start_rows:
                    length = min(ramp_frames, windows.shape[1] // 2)
                    if length:
                        scaled[row, :length] = source[row, :length] * np.linspace(1.0, gain, length)
                for row in end_rows:
                    length = min(ramp_frames, windows.shape[1] // 2)
                    if length:
                        scaled[row, -length:] = source[row, -length:] * np.linspace(gain, 1.0, length)
                windows[selected, :, side] = np.clip(np.rint(scaled), -32768, 32767).astype("<i2")

        for first in range(0, full_windows, _NORMALISE_BATCH_WINDOWS):
            count = min(_NORMALISE_BATCH_WINDOWS, full_windows - first)
            start_frame = first * window_frames
            end_frame = (first + count) * window_frames
            apply(audio[start_frame:end_frame].reshape(count, window_frames, 2), first)
        if tail_frames:
            apply(audio[full_windows * window_frames:].reshape(1, tail_frames, 2), full_windows)
        audio.flush()

    def abort(self) -> None:
        """Discard a failed capture without affecting call audio."""
        with contextlib.suppress(Exception):
            self._stream.close()
        with contextlib.suppress(OSError):
            self.part_path.unlink()
        self._finished = True

    def _write_header(self) -> None:
        data_size = self._frame_count * 4
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,
            2,
            self._sample_rate,
            self._sample_rate * 4,
            4,
            16,
            b"data",
            data_size,
        )
        self._stream.seek(0)
        self._stream.write(header)
        self._stream.seek(0, 2)
