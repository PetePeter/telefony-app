"""WAV recording format and publication behavior."""

import wave

import numpy as np
import pytest

from telefony_app.bt.recording import WavRecorder


def read_frames(path):
    with wave.open(str(path), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth()) == (2, 2)
        frames = wav.readframes(wav.getnframes())
        return wav.getframerate(), np.frombuffer(frames, dtype="<i2").reshape(-1, 2)


@pytest.mark.parametrize("rate", [8000, 16000])
def test_publishes_stereo_pcm_only_after_finalize(tmp_path, rate):
    recorder = WavRecorder(tmp_path, "interaction_1", rate)
    part = tmp_path / "interaction_1.wav.part"
    final = tmp_path / "interaction_1.wav"
    assert part.exists() and not final.exists()

    recorder.write_frame(np.array([1, 2], dtype=np.int16), np.array([10, 20], dtype=np.int16))
    assert not final.exists()
    recorder.finish()

    assert not part.exists()
    actual_rate, frames = read_frames(final)
    assert actual_rate == rate
    assert frames.tolist() == [[1, 10], [2, 20]]


def test_pads_the_missing_direction_with_silence(tmp_path):
    recorder = WavRecorder(tmp_path, "call", 8000)
    recorder.write_frame(np.array([1, 2], dtype=np.int16), np.array([10], dtype=np.int16))
    recorder.finish()
    _, frames = read_frames(tmp_path / "call.wav")
    assert frames.tolist() == [[1, 10], [2, 0]]


def test_no_audio_removes_partial_without_publishing(tmp_path):
    recorder = WavRecorder(tmp_path, "empty", 8000)
    recorder.finish()
    assert not (tmp_path / "empty.wav.part").exists()
    assert not (tmp_path / "empty.wav").exists()


def test_creates_configured_directory_when_recording_starts(tmp_path):
    directory = tmp_path / "not-created-yet"
    recorder = WavRecorder(directory, "first", 8000)
    recorder.write_frame(np.array([1], dtype=np.int16), np.array([2], dtype=np.int16))
    recorder.finish()
    assert (directory / "first.wav").exists()


@pytest.mark.parametrize("suffix", [".wav.part", ".wav"])
def test_never_overwrites_existing_recording_files(tmp_path, suffix):
    existing = tmp_path / f"taken{suffix}"
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        WavRecorder(tmp_path, "taken", 8000)
    assert existing.read_bytes() == b"keep"


@pytest.mark.parametrize("recording_id", ["../escape", "a/b", "é", "", "x" * 129])
def test_rejects_invalid_recording_id(recording_id, tmp_path):
    with pytest.raises(ValueError):
        WavRecorder(tmp_path, recording_id, 8000)
