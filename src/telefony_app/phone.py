"""The port between the PhAP service and a real phone line (Bluetooth HFP today; SIP or a modem later)."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Protocol

from telefony_app.bt.call_log import CallLogEntry
from telefony_app.bt.contacts import Contact

__all__ = ["CallLogEntry", "CallMaybePlaced", "CallNotPlaced", "Contact", "LinkDown", "Phone", "PhoneError", "PhoneEvents", "Retryable", "SmsMaybeSent", "SmsMessage"]

_RECORDING_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z", re.ASCII)


def validate_recording_id(recording_id: object) -> str | None:
    if recording_id is None:
        return None
    if not isinstance(recording_id, str) or not _RECORDING_ID.fullmatch(recording_id):
        raise ValueError("recordingId must be 1-128 ASCII letters, digits, underscores, or hyphens")
    return recording_id


class PhoneError(Exception):
    """The phone refused a command; the message carries its reason (AT or OBEX)."""


class Retryable(PhoneError):
    """Nothing reached the far end, so the command is safe to repeat."""


class LinkDown(Retryable):
    """The link to the phone is down, or the phone stopped answering; worth retrying once it is back."""


class CallNotPlaced(Retryable):
    """The phone refused the ATD at once (ERROR, NO CARRIER): no call went out."""


class CallMaybePlaced(PhoneError):
    """ATD went out and then the link failed: the phone may be calling, so never dial again."""


class SmsMaybeSent(PhoneError):
    """The phone got the message but its reply was lost, so it may go out: never resend it."""


@dataclass
class SmsMessage:
    handle: str
    number: str
    name: str | None
    text: str  # untrusted
    ts: str  # ISO-8601 UTC


class PhoneEvents(Protocol):
    """What the line reports. call_state always refers to the call most recently dialed or rung."""

    def ringing(self, number: str, name: str | None) -> None: ...
    def call_state(self, state: str, reason: str | None = None) -> None: ...
    def utterance(self, text: str, start_ms: int, end_ms: int) -> None: ...
    def speech(self, speaking: bool) -> None: ...
    def partial(self, text: str) -> None: ...  # the utterance so far, while the far end is still talking  # the far end started talking, or paused long enough to end a turn
    def dtmf(self, digit: str) -> None: ...
    def sms_received(self, handle: str, number: str, text: str) -> None: ...
    def link(self, up: bool) -> None: ...


class Phone(ABC):
    """One phone. Commands act on its current call; consult/merge build a three-way conference."""

    tts_available: bool
    stt_available: bool
    live_audio_available = False
    link_up: bool = True

    @abstractmethod
    def attach(self, events: PhoneEvents) -> None: ...

    def set_live_audio_callback(self, callback: Callable[[bytes], None] | None) -> None:
        """Subscribe to 20 ms stereo PCM16LE/8 kHz frames, or raise when unsupported."""
        if callback is not None:
            raise PhoneError("live audio is unsupported")

    @abstractmethod
    async def dial(self, number: str, recording_id: str | None = None) -> None: ...

    @abstractmethod
    async def answer(self, recording_id: str | None = None) -> None: ...

    @abstractmethod
    async def reject(self) -> None: ...

    @abstractmethod
    async def hangup(self) -> None: ...

    @abstractmethod
    async def hold(self) -> None: ...

    @abstractmethod
    async def resume(self) -> None: ...

    @abstractmethod
    async def consult(self, number: str) -> None:
        """Hold the current call and dial number."""

    @abstractmethod
    async def release_active(self) -> None:
        """End the active call and take the held one back."""

    @abstractmethod
    async def merge(self) -> None:
        """Join the held and active calls into one conference."""

    @abstractmethod
    async def speak(self, text: str) -> None:
        """Synthesise text into the call; returns once it has been played."""

    @abstractmethod
    async def send_dtmf(self, digits: str) -> None:
        """Press keys (0-9 * #) on the current call; ',' pauses."""

    @abstractmethod
    async def sms_send(self, number: str, text: str) -> str: ...

    @abstractmethod
    async def sms_list(self, limit: int) -> list[SmsMessage]: ...

    @abstractmethod
    async def sms_read(self, handle: str) -> SmsMessage: ...

    @abstractmethod
    async def contacts(self) -> list[Contact]: ...

    @abstractmethod
    async def call_log(self, limit: int) -> list[CallLogEntry]: ...
