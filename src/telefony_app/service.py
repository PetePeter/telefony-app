"""The phone service behind both protocols: calls, allowlist, events and fallback. PhAP and MCP are thin views of it."""

import asyncio
import contextlib
import itertools
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from telefony_app.allowlist import is_allowed
from telefony_app.bt import contacts as contact_book
from telefony_app.bt.dialing import normalize_number
from telefony_app.events import EventLog
from telefony_app.phone import LinkDown, Phone, PhoneError

log = logging.getLogger(__name__)

PREVIEW_LENGTH = 160
DTMF_DIGITS = re.compile(r"[0-9*#,]{1,32}")

# PhAP error codes (spec/2026-10-27/errors.md)
INVALID_PARAMS = -32602
NOT_ALLOWED = 1001
NO_LINK = 1002
NO_SUCH_CALL = 1003
BUSY = 1004
UNSUPPORTED = 1006
PHONE_ERROR = 1007


class PhapError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code, self.message, self.data = code, message, data


@dataclass
class Policy:
    allowlist: list[str] = field(default_factory=list)
    fallback_policy: Literal["ring", "reject"] = "ring"
    ring_timeout_s: float = 15
    country_code: str = "+61"  # how national numbers (0..) are read when matching the allowlist
    outbound: Literal["allowlist", "all"] = "allowlist"  # all: the controller alone decides who is called or texted


@dataclass
class Call:
    call_id: str
    number: str
    direction: Literal["inbound", "outbound"]
    state: str
    name: str | None = None
    merge_on_answer: bool = False  # a consult call: join the held call once it answers
    conference: bool = False
    transcript: list[dict] = field(default_factory=list)

    def info(self) -> dict:
        info = {"callId": self.call_id, "number": self.number, "direction": self.direction, "state": self.state}
        return info | ({"name": self.name} if self.name else {})


class PhoneService:
    def __init__(self, phone: Phone, policy: Policy, event_buffer: int = 1000) -> None:
        self.phone = phone
        self.policy = policy
        self.events = EventLog(event_buffer)
        self.calls: dict[str, Call] = {}
        self.engine = "ready"
        self._current: Call | None = None  # the call the phone's call_state reports refer to
        self._ids = itertools.count(1)
        self._speaking = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self.says: dict[str, asyncio.Task] = {}  # sayId -> playback; its result is None, or why it failed
        phone.attach(self)

    # --- state ---

    def state(self) -> dict:
        live = [c.info() for c in self.calls.values() if c.state != "ended"]
        return {"link": "up" if self.phone.link_up else "down", "engine": self.engine, "calls": live}

    def capabilities(self) -> dict:
        return {
            "calls": {},
            "conference": {},
            "sms": {},
            "contacts": {},
            "voice": {"tts": self.phone.tts_available, "stt": self.phone.stt_available},
        }

    # --- calls ---

    async def dial(self, number: str | None = None, contact: str | None = None) -> str:
        number = await self._reachable(number, contact)
        self._ready()
        if any(c.state != "ended" for c in self.calls.values()):
            raise PhapError(BUSY, "busy", {"detail": "a call is already up; use conference.create"})
        call = self._new_call(number, "outbound", "dialing")
        await self._failing(call, self.phone.dial(number))
        return call.call_id

    async def answer(self, call_id: str) -> None:
        call = self._call(call_id, "ringing")
        await self._phone(self.phone.answer())
        self._set_state(call, "active")

    async def reject(self, call_id: str) -> None:
        call = self._call(call_id, "ringing")
        await self._phone(self.phone.reject())
        self._set_state(call, "ended", "rejected")

    async def hangup(self, call_id: str) -> None:
        call = self._call(call_id)
        if call.state == "ringing":
            raise PhapError(INVALID_PARAMS, "invalid params", {"detail": "call is ringing; use call.reject", "state": call.state})
        await self._phone(self.phone.hangup())  # the phone drops a whole conference at once
        for each in [c for c in self.calls.values() if c is call or (call.conference and c.conference)]:
            self._set_state(each, "ended", "local")

    async def hold(self, call_id: str) -> None:
        call = self._call(call_id, "active")
        await self._phone(self.phone.hold())
        self._set_state(call, "held")

    async def resume(self, call_id: str) -> None:
        call = self._call(call_id, "held")
        await self._phone(self.phone.resume())
        self._set_state(call, "active")

    async def say(self, call_id: str, text: str) -> str:
        """Queue speech; returns its sayId at once. call.spoken is emitted once it has been played."""
        if not self.phone.tts_available:
            raise PhapError(UNSUPPORTED, "unsupported", {"detail": "no text-to-speech engine configured"})
        call = self._call(call_id, "active")
        say_id = f"s{next(self._ids)}"
        self.says[say_id] = self._spawn(self._speak(call, say_id, text))
        return say_id

    async def _speak(self, call: Call, say_id: str, text: str) -> str | None:
        async with self._speaking:
            if call.state == "ended":
                return "call ended"
            try:
                await self.phone.speak(text)
            except Exception as failed:
                log.exception("say %s failed", say_id)
                return str(failed) or type(failed).__name__
            call.transcript.append({"speaker": "local", "text": text, "startMs": 0, "endMs": 0})
            self.events.emit("call.spoken", callId=call.call_id, sayId=say_id)
            return None

    async def send_dtmf(self, call_id: str, digits: str) -> None:
        if not isinstance(digits, str) or not DTMF_DIGITS.fullmatch(digits):
            raise PhapError(INVALID_PARAMS, "invalid params", {"detail": "digits must be 1-32 of 0-9 * # ,"})
        self._call(call_id, "active")
        await self._phone(self.phone.send_dtmf(digits))

    async def conference_create(self, call_id: str, number: str | None = None, contact: str | None = None) -> str:
        number = await self._reachable(number, contact)
        first = self._call(call_id, "active")
        consult = self._new_call(number, "outbound", "dialing", merge_on_answer=True)
        try:
            await self._failing(consult, self.phone.consult(number))
        except Exception:
            self._current = first
            raise
        self._set_state(first, "held")
        return consult.call_id

    async def conference_join(self, call_id: str) -> None:
        held = self._call(call_id, "held")
        await self._phone(self.phone.merge())
        self._set_state(held, "active", conference=True)

    # --- SMS and contacts ---

    async def sms_send(self, number: str, text: str) -> str:
        if not self._may_reach(number):
            raise PhapError(NOT_ALLOWED, "not_allowed")
        self._ready()
        return await self._phone(self.phone.sms_send(number, text))

    async def sms_list(self, limit: int = 20) -> list[dict]:
        self._ready()
        messages = await self._phone(self.phone.sms_list(limit))
        return [_summary(m.handle, m.number, m.name, m.text) | {"ts": m.ts} for m in messages]

    async def sms_read(self, handle: str) -> dict:
        self._ready()
        try:
            message = await self._phone(self.phone.sms_read(handle))
        except KeyError:
            raise PhapError(INVALID_PARAMS, "invalid params", {"detail": f"no message {handle!r}"}) from None
        return {"number": message.number, "text": message.text, "ts": message.ts} | (
            {"name": message.name} if message.name else {}
        )

    async def contacts_search(self, query: str) -> list[dict]:
        self._ready()
        found = contact_book.search(await self._phone(self.phone.contacts()), query)
        return [_contact(c) for c in found]

    def transcript(self, call_id: str) -> list[dict]:
        return list(self._call(call_id).transcript)

    # --- engine ---

    def engine_start(self) -> None:
        self._set_engine("ready")

    def engine_stop(self) -> None:
        self._set_engine("stopped")

    def _set_engine(self, status: str) -> None:
        if self.engine != status:
            self.engine = status
            self.events.emit("engine", status=status)

    # --- PhoneEvents: what the phone reports ---

    def ringing(self, number: str, name: str | None) -> None:
        call = self._new_call(number, "inbound", "ringing", name=name, announce=False)
        allowed = is_allowed(number, self.policy.allowlist, self.policy.country_code)
        self.events.emit("call.ringing", callId=call.call_id, number=number, name=name, allowed=allowed)
        self._spawn(self._fallback(call))

    def call_state(self, state: str, reason: str | None = None) -> None:
        call = self._current
        if call is None or call.state == "ended" or call.state == state:
            return
        self._set_state(call, state, reason)
        if state == "ended" and call.conference:  # the phone reports one end for the whole conference
            for other in self.calls.values():
                if other.conference and other.state != "ended":
                    self._set_state(other, "ended", reason)
        if state == "ended" and call.merge_on_answer:  # the consulted party never joined: back to the first call
            call.merge_on_answer = False
            held = [c for c in self.calls.values() if c.state == "held"]
            if held:
                self._current = held[-1]
                self._set_state(held[-1], "active")
        if state == "active" and call.merge_on_answer:
            call.merge_on_answer = False
            self._spawn(self._merge())

    def utterance(self, text: str, start_ms: int, end_ms: int) -> None:
        call = self._current
        if call is None or call.state != "active":
            return
        call.transcript.append({"speaker": "remote", "text": text, "startMs": start_ms, "endMs": end_ms})
        self.events.emit("call.utterance", callId=call.call_id, text=text, startMs=start_ms, endMs=end_ms)

    def dtmf(self, digit: str) -> None:
        call = self._current
        if call is not None and call.state == "active":
            self.events.emit("call.dtmf", callId=call.call_id, digit=digit)

    def sms_received(self, handle: str, number: str, text: str) -> None:
        self.events.emit("sms.received", **_summary(handle, number, None, text))

    def link(self, up: bool) -> None:
        self.events.emit("link", status="up" if up else "down")
        if not up:
            for call in self.calls.values():
                if call.state != "ended":
                    self._set_state(call, "ended", "link_lost")

    # --- internals ---

    async def _fallback(self, call: Call) -> None:
        await asyncio.sleep(self.policy.ring_timeout_s)
        if call.state != "ringing" or self.policy.fallback_policy != "reject":
            return
        with contextlib.suppress(PhoneError):
            await self.phone.reject()
        self._set_state(call, "ended", "fallback")

    async def _merge(self) -> None:
        try:
            await self.phone.merge()
        except PhoneError:
            return
        for call in self.calls.values():
            if call.state in ("held", "active"):
                self._set_state(call, "active", conference=True, force=True)

    async def _reachable(self, number: str | None, contact: str | None) -> str:
        if contact is not None:
            try:
                number = contact_book.resolve_number(await self._phone(self.phone.contacts()), contact)
            except contact_book.Ambiguous as ambiguous:
                matches = [_contact(c) for c in ambiguous.candidates]
                raise PhapError(INVALID_PARAMS, "invalid params", {"detail": str(ambiguous), "matches": matches}) from None
            except LookupError as missing:
                raise PhapError(INVALID_PARAMS, "invalid params", {"detail": str(missing)}) from None
        if number is None:
            raise PhapError(INVALID_PARAMS, "invalid params", {"detail": "number or contact is required"})
        if not self._may_reach(number):
            raise PhapError(NOT_ALLOWED, "not_allowed")
        return number

    def _may_reach(self, number: str) -> bool:
        """Outbound filter. Even with outbound = all, only well-formed numbers pass: nothing that could carry AT syntax."""
        if self.policy.outbound == "all":
            try:
                normalize_number(number)
            except ValueError:
                return False
            return True
        return is_allowed(number, self.policy.allowlist, self.policy.country_code)

    def _ready(self) -> None:
        if not self.phone.link_up or self.engine != "ready":
            raise PhapError(NO_LINK, "no_link")

    def _call(self, call_id: str, state: str | None = None) -> Call:
        call = self.calls.get(call_id)
        if call is None or call.state == "ended":
            raise PhapError(NO_SUCH_CALL, "no_such_call")
        if state and call.state != state:
            raise PhapError(INVALID_PARAMS, "invalid params", {"detail": f"call is {call.state}, not {state}", "state": call.state})
        return call

    def _new_call(self, number: str, direction, state: str, announce: bool = True, **extra: Any) -> Call:
        call = Call(f"c{next(self._ids)}", number, direction, state, **extra)
        self.calls[call.call_id] = self._current = call
        if announce:
            self.events.emit("call.state", callId=call.call_id, state=state)
        return call

    def _set_state(
        self, call: Call, state: str, reason: str | None = None, conference: bool | None = None, force: bool = False
    ) -> None:
        if call.state == state and not force:
            return
        call.state = state
        call.conference = call.conference or bool(conference)
        self.events.emit("call.state", callId=call.call_id, state=state, reason=reason, conference=conference)

    async def _phone(self, command):
        try:
            return await command
        except LinkDown as down:
            raise PhapError(NO_LINK, "no_link", {"detail": str(down)}) from None
        except PhoneError as refused:
            raise PhapError(PHONE_ERROR, "phone_error", {"detail": str(refused)}) from None

    async def _failing(self, call: Call, command) -> None:
        """Run a command that places call; if the phone refuses, the call ended before it began."""
        try:
            await self._phone(command)
        except Exception:
            self._set_state(call, "ended", "failed")
            raise

    def _spawn(self, coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task


def _summary(handle: str, number: str, name: str | None, text: str) -> dict:
    summary = {"handle": handle, "number": number, "preview": text[:PREVIEW_LENGTH]}
    return summary | ({"name": name} if name else {})


def _contact(contact: contact_book.Contact) -> dict:
    return {"name": contact.name, "numbers": [{"number": n, "label": t} for t, n in contact.numbers]}
