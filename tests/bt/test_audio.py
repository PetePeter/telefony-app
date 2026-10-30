import numpy as np
from bumble.hci import HCI_SynchronousDataPacket
from typing import cast

from telefony_app.bt.audio import AudioPump, resample, shape_for_phone
from telefony_app.bt.dtmf import DtmfDecoder

Status = HCI_SynchronousDataPacket.Status
GOOD = Status.CORRECTLY_RECEIVED_DATA


def tone(rate: int, seconds: float, hz: float = 440.0) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (np.sin(2 * np.pi * hz * t) * 10000).astype(np.int16)


def dominant_hz(pcm: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(pcm.astype(np.float64)))
    return np.fft.rfftfreq(len(pcm), 1 / rate)[np.argmax(spectrum)]


def test_resample_piper_rate_to_cvsd_keeps_length_and_pitch():
    out = resample(tone(22050, 1.0), 22050, 8000)
    assert out.dtype == np.int16
    assert len(out) == 8000
    assert abs(dominant_hz(out, 8000) - 440) < 5


def test_resample_filters_content_above_new_nyquist():
    # 6 kHz cannot exist at 8 kHz; it must be filtered out, not aliased to 2 kHz.
    out = resample(tone(22050, 1.0, hz=6000), 22050, 8000)
    assert np.abs(out[100:-100]).max() < 100  # edges carry filter start-up transients


def packet(samples: int, value: int = 100) -> bytes:
    return np.full(samples, value, dtype=np.int16).tobytes()


def test_uplink_sends_same_sample_count_as_received_padding_with_silence():
    pump = AudioPump()
    pump.queue_uplink(np.arange(1, 46, dtype=np.int16))  # 45 samples
    first = np.frombuffer(pump.on_packet(packet(30), GOOD), dtype=np.int16)
    second = np.frombuffer(pump.on_packet(packet(30), GOOD), dtype=np.int16)
    assert len(first) == len(second) == 30
    assert list(first) == list(range(1, 31))
    assert list(second[:15]) == list(range(31, 46))
    assert not second[15:].any()
    assert pump.uplink_pending == 0


def test_uplink_idle_sends_silence():
    assert AudioPump().on_packet(packet(30), GOOD) == bytes(60)


def test_downlink_no_data_packet_becomes_silence_of_same_length():
    pump = AudioPump()
    pump.on_packet(packet(30, 7), GOOD)
    pump.on_packet(packet(30, 9), Status.NO_DATA)
    pump.on_packet(packet(30, 7), GOOD)
    pcm = pump.take_downlink()
    assert len(pcm) == 90
    assert not pcm[30:60].any()
    assert (pcm[:30] == 7).all() and (pcm[60:] == 7).all()
    assert len(pump.take_downlink()) == 0


def test_downlink_keeps_flagged_but_present_data():
    # The phone flagged ~8% of packets on a real call; zeroing them made audible dropouts.
    pump = AudioPump()
    pump.on_packet(packet(30, 5), Status.POSSIBLY_INVALID_DATA)
    pump.on_packet(packet(30, 6), Status.DATA_PARTIALLY_LOST)
    pcm = pump.take_downlink()
    assert (pcm[:30] == 5).all() and (pcm[30:] == 6).all()


def test_recording_tap_pairs_raw_receive_with_actual_padded_transmit_frame():
    class Filter:
        def feed(self, pcm):
            return [], np.zeros_like(pcm)

    captured = []
    pump = AudioPump(cast(DtmfDecoder, Filter()))
    pump.on_recording_frame = lambda received, transmitted: captured.append((received.copy(), transmitted.copy()))
    pump.queue_uplink(np.array([8], dtype=np.int16))
    pump.on_packet(packet(2, 4), GOOD)
    pump.on_packet(packet(2, 5), Status.NO_DATA)

    assert captured[0][0].tolist() == [4, 4]
    assert captured[0][1].tolist() == [8, 0]
    assert captured[1][0].tolist() == [0, 0]
    assert captured[1][1].tolist() == [0, 0]


def test_live_audio_emits_20_ms_interleaved_stereo_pcm16_frames():
    frames = []
    pump = AudioPump()
    pump.set_live_audio_callback(frames.append)
    pump.queue_uplink(np.full(192, -321, dtype=np.int16))

    for _ in range(4):
        pump.on_packet(packet(48, 123), GOOD)

    assert len(frames) == 1
    samples = np.frombuffer(frames[0], dtype="<i2")
    assert len(frames[0]) == 640  # 160 stereo samples, 20 ms at 8 kHz
    assert (samples[::2] == 123).all()
    assert (samples[1::2] == -321).all()


def test_live_audio_subscription_drops_pre_subscription_samples():
    pump = AudioPump()
    pump.on_packet(packet(80, 7), GOOD)
    frames = []
    pump.set_live_audio_callback(frames.append)

    pump.on_packet(packet(80, 8), GOOD)
    assert frames == []
    pump.on_packet(packet(80, 9), GOOD)

    samples = np.frombuffer(frames[0], dtype="<i2")
    assert (samples[::2][:80] == 8).all()
    assert (samples[::2][80:] == 9).all()


def test_live_audio_callback_failure_does_not_escape_audio_pump(caplog):
    def broken(_frame):
        raise RuntimeError("listener failed")

    pump = AudioPump()
    pump.set_live_audio_callback(broken)

    assert pump.on_packet(packet(160, 1), GOOD) == bytes(320)
    assert "live audio callback failed" in caplog.text


def gain_difference_db(hz_a: float, hz_b: float) -> float:
    """Shape a two-tone mix (output is peak-normalised) and return level(a) - level(b) in dB."""
    out = shape_for_phone(tone(8000, 1.0, hz=hz_a) // 2 + tone(8000, 1.0, hz=hz_b) // 2, 8000)
    spectrum = np.abs(np.fft.rfft(out[400:-400].astype(np.float64)))  # skip filter start-up
    freqs = np.fft.rfftfreq(len(out) - 800, 1 / 8000)
    level = lambda hz: spectrum[np.argmin(np.abs(freqs - hz))]  # noqa: E731
    return 20 * np.log10(level(hz_a) / level(hz_b))


def test_phone_shaping_cuts_rumble_below_the_telephone_band():
    assert gain_difference_db(1000, 120) > 15


def test_phone_shaping_lifts_presence_over_low_mids():
    # A listener reported the Piper voice as muffled: 2-3.4 kHz sat 16 dB below the low-mids.
    assert gain_difference_db(2800, 500) > 3


def test_phone_shaping_leaves_headroom_so_cvsd_does_not_clip():
    loud = (tone(8000, 1.0, hz=1000).astype(np.int32) * 3).clip(-32768, 32767).astype(np.int16)
    out = shape_for_phone(loud, 8000)
    assert out.dtype == np.int16
    assert np.abs(out).max() <= 16500  # about -6 dBFS
