"""An in-memory phone behind the Phone port, scripted by tests to ring, answer, speak and text."""

import asyncio

from telefony_app.phone import Contact, Phone, PhoneEvents, SmsMessage


class FakePhone(Phone):
    def __init__(self, *, tts: bool = True, stt: bool = True) -> None:
        self.tts_available = tts
        self.stt_available = stt
        self.events: PhoneEvents | None = None
        self.commands: list[tuple] = []  # every command the service sent, in order
        self.spoken: list[str] = []
        self.sms: dict[str, SmsMessage] = {}
        self.contacts_list: list[Contact] = []
        self.speak_gate: asyncio.Event | None = None  # set by a test to hold say() mid-playback
        self.dial_gate: asyncio.Event | None = None  # set by a test to hold dial() until it is set
        self.failing: dict[str, Exception] = {}  # command name -> what it raises

    def attach(self, events: PhoneEvents) -> None:
        self.events = events

    async def dial(self, number: str) -> None:
        self.commands.append(("dial", number))
        if self.dial_gate:
            await self.dial_gate.wait()
        self._fail("dial")

    async def answer(self) -> None:
        self.commands.append(("answer",))

    async def reject(self) -> None:
        self.commands.append(("reject",))

    async def hangup(self) -> None:
        self.commands.append(("hangup",))

    async def hold(self) -> None:
        self.commands.append(("hold",))

    async def resume(self) -> None:
        self.commands.append(("resume",))

    async def consult(self, number: str) -> None:
        self.commands.append(("consult", number))
        self._fail("consult")

    async def merge(self) -> None:
        self.commands.append(("merge",))

    async def speak(self, text: str) -> None:
        self.commands.append(("speak", text))
        if self.speak_gate:
            await self.speak_gate.wait()
        self._fail("speak")
        self.spoken.append(text)

    async def sms_send(self, number: str, text: str) -> str:
        self.commands.append(("sms_send", number, text))
        return f"h{len(self.commands)}"

    async def sms_list(self, limit: int) -> list[SmsMessage]:
        return list(self.sms.values())[:limit]

    async def sms_read(self, handle: str) -> SmsMessage:
        return self.sms[handle]

    async def contacts(self) -> list[Contact]:
        return self.contacts_list

    def _fail(self, command: str) -> None:
        if command in self.failing:
            raise self.failing[command]

    # --- the far end, driven by tests ---

    def ring(self, number: str) -> None:
        assert self.events
        self.events.ringing(number, None)

    def remote_answers(self) -> None:
        assert self.events
        self.events.call_state("active")

    def remote_hangs_up(self) -> None:
        assert self.events
        self.events.call_state("ended", "remote")

    def caller_says(self, text: str) -> None:
        assert self.events
        self.events.utterance(text, 0, 1000)
