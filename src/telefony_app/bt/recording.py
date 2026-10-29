"""Write call audio as an unpublished WAV until the call reaches a terminal boundary."""

import contextlib
from pathlib import Path
import struct

import numpy as np

from telefony_app.phone import validate_recording_id


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
            if self.final_path.exists():
                raise FileExistsError(self.final_path)
            self.part_path.rename(self.final_path)
            self._finished = True
            return self.final_path
        except BaseException:
            self.abort()
            raise

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
