"""Continuous SCO uplink over USB isochronous OUT.

Bumble's USB sink keeps one OUT transfer in flight, sharing it with ACL. Every resubmit gap
costs USB frames, and on a real call ~6% of uplink audio never reached the dongle (choppy for
the far end). This streamer keeps several SCO transfers queued back to back and pads underruns
with silent SCO packets so the isochronous stream never pauses.
"""

import collections
import logging
import math
import threading
import time
from typing import Any

import usb1
from bumble import hci

log = logging.getLogger(__name__)

_STATS_EVERY_SECONDS = 10.0


def iso_lengths(total: int, frame_size: int) -> list[int]:
    """Split a transfer into USB frame-sized iso packets (libusb rejects zero-length ones)."""
    full, rest = divmod(total, frame_size)
    return [frame_size] * full + ([rest] if rest else [])


class ScoOutQueue:
    """Outgoing HCI SCO packets (header + payload, no packet-type byte), padded with silence on underrun."""

    def __init__(self, max_packets: int) -> None:
        self._packets: collections.deque[bytes] = collections.deque(maxlen=max_packets)  # drops oldest
        self._silence = b""

    def push(self, packet: bytes) -> None:
        self._silence = packet[:3] + bytes(len(packet) - 3)  # same handle and length, zero samples
        self._packets.append(packet)

    def take(self, count: int) -> bytes:
        if not self._silence:
            return b""
        return b"".join(self._packets.popleft() if self._packets else self._silence for _ in range(count))


class UsbScoStreamer:
    """Takes over SCO packets sent to a bumble UsbPacketSink; everything else passes through."""

    def __init__(self, sink: Any, packets_per_transfer: int = 4, in_flight: int = 3, max_queued: int = 50) -> None:
        self._sink = sink
        self._forward = sink.on_packet
        self._endpoint = sink.isochronous_out
        self._per_transfer = packets_per_transfer
        self._in_flight = in_flight
        self._queue = ScoOutQueue(max_queued)
        self._lock = threading.Lock()  # pushes come from the asyncio thread, refills from libusb's
        self._started = False
        self._sent = self._failed = 0
        self._stats_at = time.monotonic()
        sink.on_packet = self.on_packet  # the host holds the sink object, so patch the instance

    def on_packet(self, packet: bytes) -> None:
        if packet[0] != hci.HCI_SYNCHRONOUS_DATA_PACKET:
            self._forward(packet)
            return
        with self._lock:
            self._queue.push(packet[1:])
            start = not self._started
            self._started = True
        if start:
            frames = math.ceil(len(packet[1:]) * self._per_transfer / self._endpoint.getMaxPacketSize())
            for _ in range(self._in_flight):
                self._submit(self._sink.device.getTransfer(iso_packets=frames))

    def _submit(self, transfer: Any) -> None:
        if self._sink.closed:
            return
        with self._lock:
            data = self._queue.take(self._per_transfer)
        transfer.setIsochronous(
            self._endpoint.getAddress(),
            data,
            callback=self._on_done,
            iso_transfer_length_list=iso_lengths(len(data), self._endpoint.getMaxPacketSize()),
        )
        transfer.submit()

    def _on_done(self, transfer: Any) -> None:
        # usb1 exports its status constants dynamically, so pyright cannot see them.
        stopped = (usb1.TRANSFER_CANCELLED, usb1.TRANSFER_NO_DEVICE)  # pyright: ignore[reportAttributeAccessIssue]
        if transfer.getStatus() in stopped:
            return
        for status, buffer in transfer.iterISO():
            if status:
                self._failed += 1
            else:
                self._sent += len(buffer)
        if (now := time.monotonic()) - self._stats_at >= _STATS_EVERY_SECONDS:
            log.debug("sco out: %d bytes sent, %d iso packets failed", self._sent, self._failed)
            self._stats_at = now
        self._submit(transfer)
