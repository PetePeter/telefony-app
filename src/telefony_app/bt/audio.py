"""PCM plumbing between the SCO link and TTS/STT. SCO carries 16-bit mono PCM (8 kHz CVSD, 16 kHz mSBC)."""

from collections.abc import Callable
from fractions import Fraction
import logging

import numpy as np
from bumble.hci import HCI_SynchronousDataPacket
from scipy.signal import butter, resample_poly, sosfilt

from telefony_app.bt.dtmf import DtmfDecoder

_PEAK = 16384  # -6 dBFS: CVSD's slope overload distorts near full scale
LIVE_AUDIO_RATE = 8000
LIVE_AUDIO_FRAME_SAMPLES = 160  # 20 ms at 8 kHz

log = logging.getLogger(__name__)


STT_RATE = 16000  # whisper's input rate


def resample(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return pcm
    ratio = Fraction(dst_rate, src_rate)
    out = resample_poly(pcm.astype(np.float32), ratio.numerator, ratio.denominator)
    return np.clip(out, -32768, 32767).astype(np.int16)


def shape_for_phone(pcm: np.ndarray, rate: int) -> np.ndarray:
    """EQ synthetic speech for a narrowband call: drop sub-300 Hz rumble, lift 1.8-3.4 kHz presence."""
    x = np.asarray(sosfilt(butter(4, 300, "highpass", fs=rate, output="sos"), pcm.astype(np.float64)))
    x = x + 2 * np.asarray(sosfilt(butter(2, [1800, 3400], "bandpass", fs=rate, output="sos"), x))  # ~+9.5 dB in band
    peak = np.abs(x).max()
    return (x * (_PEAK / peak) if peak else x).astype(np.int16)


class AudioPump:
    """SCO is paced by the controller: for every packet received, send the same number of samples back."""

    def __init__(self, dtmf: DtmfDecoder | None = None) -> None:
        self._uplink = np.zeros(0, dtype=np.int16)
        self._downlink: list[np.ndarray] = []
        self._dtmf = dtmf  # takes key presses out of the downlink
        self.on_dtmf: Callable[[str], None] | None = None
        self.tap: Callable[[np.ndarray], None] | None = None  # sees all downlink audio, e.g. a transcript
        self.on_recording_frame: Callable[[np.ndarray, np.ndarray], None] | None = None
        self._live_audio_callback: Callable[[bytes], None] | None = None
        self._live_audio_buffer = np.empty((LIVE_AUDIO_FRAME_SAMPLES, 2), dtype="<i2")
        self._live_audio_samples = 0

    @property
    def uplink_pending(self) -> int:
        return len(self._uplink)

    def queue_uplink(self, pcm: np.ndarray) -> None:
        self._uplink = np.concatenate([self._uplink, pcm.astype(np.int16)])

    def set_live_audio_callback(self, callback: Callable[[bytes], None] | None) -> None:
        """Capture future 20 ms stereo PCM16LE frames; changing listeners drops partial audio."""
        self._live_audio_callback = callback
        self._live_audio_samples = 0

    def on_packet(self, data: bytes, status: int) -> bytes:
        received = np.frombuffer(data, dtype=np.int16)
        # Flagged packets still carry mostly-right audio; only NO_DATA is garbage.
        no_data = status == HCI_SynchronousDataPacket.Status.NO_DATA
        heard = np.zeros_like(received) if no_data else received.copy()
        recording_receive = heard.copy()
        if self._dtmf:
            digits, heard = self._dtmf.feed(heard)
            for digit in digits:
                if self.on_dtmf:
                    self.on_dtmf(digit)
        self._downlink.append(heard)
        if self.tap:
            self.tap(heard)
        out = np.zeros(len(received), dtype=np.int16)
        take = min(len(received), len(self._uplink))
        out[:take] = self._uplink[:take]
        self._uplink = self._uplink[take:]
        if self.on_recording_frame:
            self.on_recording_frame(recording_receive, out)
        self._capture_live_audio(recording_receive, out)
        return out.tobytes()

    def _capture_live_audio(self, received: np.ndarray, transmitted: np.ndarray) -> None:
        if self._live_audio_callback is None:
            return
        offset = 0
        while offset < len(received):
            count = min(LIVE_AUDIO_FRAME_SAMPLES - self._live_audio_samples, len(received) - offset)
            end = offset + count
            target = self._live_audio_samples
            self._live_audio_buffer[target : target + count, 0] = received[offset:end]
            self._live_audio_buffer[target : target + count, 1] = transmitted[offset:end]
            self._live_audio_samples += count
            offset = end
            if self._live_audio_samples == LIVE_AUDIO_FRAME_SAMPLES:
                callback = self._live_audio_callback
                frame = self._live_audio_buffer.tobytes()
                self._live_audio_samples = 0
                if callback is not None:
                    try:
                        callback(frame)
                    except Exception:
                        log.exception("live audio callback failed; continuing SCO audio")

    def take_downlink(self) -> np.ndarray:
        pcm = np.concatenate(self._downlink) if self._downlink else np.zeros(0, dtype=np.int16)
        self._downlink = []
        return pcm
