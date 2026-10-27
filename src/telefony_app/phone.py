"""The port between the PhAP service and a real phone line (Bluetooth HFP today; SIP or a modem later)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from telefony_app.bt.contacts import Contact

__all__ = ["Contact", "LinkDown", "Phone", "PhoneError", "PhoneEvents", "SmsMessage"]


class PhoneError(Exception):
    """The phone refused a command; the message carries its reason (AT or OBEX)."""


class LinkDown(PhoneError):
    """The link to the phone is down."""


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
    def dtmf(self, digit: str) -> None: ...
    def sms_received(self, handle: str, number: str, text: str) -> None: ...
    def link(self, up: bool) -> None: ...


class Phone(ABC):
    """One phone. Commands act on its current call; consult/merge build a three-way conference."""

    tts_available: bool
    stt_available: bool
    link_up: bool = True

    @abstractmethod
    def attach(self, events: PhoneEvents) -> None: ...

    @abstractmethod
    async def dial(self, number: str) -> None: ...

    @abstractmethod
    async def answer(self) -> None: ...

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
