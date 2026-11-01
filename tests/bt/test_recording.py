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


def test_normalizes_both_directions_to_shared_active_loudness(tmp_path):
    sample_count = 8000
    phase = np.arange(sample_count) * (2 * np.pi / 100)
    received = (8000 * np.sin(phase)).astype(np.int16)
    transmitted = (4000 * np.sin(phase)).astype(np.int16)
    recorder = WavRecorder(tmp_path, "balanced", 8000)

    recorder.write_frame(received, transmitted)
    recorder.finish()

    _, frames = read_frames(tmp_path / "balanced.wav")
    levels = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=0))
    assert abs(levels[0] - levels[1]) / levels[0] < 0.03
    assert np.max(np.abs(frames.astype(np.int32))) <= 32767
    assert np.max(np.abs(frames[:, 0].astype(np.int32))) < received.max()
    assert np.max(np.abs(frames[:, 1].astype(np.int32))) > transmitted.max()
    assert received.max() == 8000
    assert transmitted.max() == 4000


def test_caps_gain_and_leaves_silence_and_noise_below_the_floor_unchanged(tmp_path):
    sample_count = 8000
    phase = np.arange(sample_count) * (2 * np.pi / 100)
    reference = (12000 * np.sin(phase)).astype(np.int16)
    quiet_speech = (1200 * np.sin(phase)).astype(np.int16)
    recorder = WavRecorder(tmp_path, "gated", 8000)

    recorder.write_frame(reference, quiet_speech)
    recorder.finish()

    _, frames = read_frames(tmp_path / "gated.wav")
    quiet_peak = np.max(np.abs(frames[:, 1].astype(np.int32)))
    assert 3600 < quiet_peak <= 4800

    noise = np.resize(np.array([-100, 100], dtype=np.int16), sample_count)
    silence = np.zeros(sample_count, dtype=np.int16)
    low_level = WavRecorder(tmp_path, "low-level", 8000)
    low_level.write_frame(noise, silence)
    low_level.finish()
    _, low_frames = read_frames(tmp_path / "low-level.wav")
    np.testing.assert_array_equal(low_frames[:, 0], noise)
    np.testing.assert_array_equal(low_frames[:, 1], silence)


def test_does_not_boost_noise_or_silence_windows_around_active_speech(tmp_path):
    window = 160
    active_frames = window * 10
    phase = np.arange(active_frames) * (2 * np.pi / 20)
    received = np.concatenate(((8000 * np.sin(phase)).astype(np.int16), np.zeros(window, dtype=np.int16)))
    noise = np.resize(np.array([-100, 100], dtype=np.int16), window)
    transmitted = np.concatenate(((4000 * np.sin(phase)).astype(np.int16), noise))
    recorder = WavRecorder(tmp_path, "gated-windows", 8000)

    recorder.write_frame(received, transmitted)
    recorder.finish()

    _, frames = read_frames(tmp_path / "gated-windows.wav")
    active_rms = np.sqrt(np.mean(frames[:active_frames].astype(np.float64) ** 2, axis=0))
    assert abs(active_rms[0] - active_rms[1]) / active_rms[0] < 0.03
    np.testing.assert_array_equal(frames[active_frames:, 0], np.zeros(window, dtype=np.int16))
    np.testing.assert_array_equal(frames[active_frames:, 1], noise)


def test_ramps_gain_at_active_voice_edges_without_boosting_the_noise_window(tmp_path):
    window = 160
    phase = np.arange(window) * (2 * np.pi / 20)
    received = (12000 * np.sin(phase)).astype(np.int16)
    transmitted = (4000 * np.sin(phase)).astype(np.int16)
    noise = np.resize(np.array([-100, 100], dtype=np.int16), window)
    recorder = WavRecorder(tmp_path, "ramped", 8000)
    recorder.write_frame(np.concatenate((received, np.zeros(window, dtype=np.int16))),
                         np.concatenate((transmitted, noise)))

    recorder.finish()

    _, frames = read_frames(tmp_path / "ramped.wav")
    original_boundary = abs(int(transmitted[-1]) - int(noise[0]))
    processed_boundary = abs(int(frames[window - 1, 1]) - int(frames[window, 1]))
    assert processed_boundary <= original_boundary + 2
    np.testing.assert_array_equal(frames[window:, 1], noise)


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
