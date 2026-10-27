"""PCM plumbing between the SCO link and TTS/STT. SCO carries 16-bit mono PCM (8 kHz CVSD, 16 kHz mSBC)."""

from collections.abc import Callable
from fractions import Fraction

import numpy as np
from bumble.hci import HCI_SynchronousDataPacket
from scipy.signal import butter, resample_poly, sosfilt

from telefony_app.bt.dtmf import DtmfDecoder

_PEAK = 16384  # -6 dBFS: CVSD's slope overload distorts near full scale


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

    @property
    def uplink_pending(self) -> int:
        return len(self._uplink)

    def queue_uplink(self, pcm: np.ndarray) -> None:
        self._uplink = np.concatenate([self._uplink, pcm.astype(np.int16)])

    def on_packet(self, data: bytes, status: int) -> bytes:
        received = np.frombuffer(data, dtype=np.int16)
        # Flagged packets still carry mostly-right audio; only NO_DATA is garbage.
        no_data = status == HCI_SynchronousDataPacket.Status.NO_DATA
        heard = np.zeros_like(received) if no_data else received.copy()
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
        return out.tobytes()

    def take_downlink(self) -> np.ndarray:
        pcm = np.concatenate(self._downlink) if self._downlink else np.zeros(0, dtype=np.int16)
        self._downlink = []
        return pcm
