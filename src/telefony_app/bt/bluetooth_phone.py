"""The Phone port over Bluetooth: HFP call control and SCO audio (HfLine), PBAP contacts and call log, MAP SMS."""

import asyncio
import contextlib
import logging
import wave
from datetime import datetime

from bumble import hfp, rfcomm
from bumble.device import Connection

from telefony_app.bt.call_state import CallEnded, CallState
from telefony_app.bt.dialing import normalize_number
from telefony_app.bt.hf_line import HfLine, LinkLost
from telefony_app.bt.map_client import map_session
from telefony_app.bt.pbap import pull_call_log, pull_contacts
from telefony_app.phone import CallLogEntry, Contact, LinkDown, Phone, PhoneError, PhoneEvents, SmsMessage
from telefony_app.voice import Tts, WhisperStt

log = logging.getLogger(__name__)
_UTTERANCE_MAX_S = 30.0


class BluetoothPhone(Phone):
    def __init__(self, line: HfLine, dlc: rfcomm.DLC, tts: Tts | None, stt: WhisperStt | None) -> None:
        self.tts, self.stt = tts, stt
        self.tts_available, self.stt_available = tts is not None, stt is not None
        self.events: PhoneEvents | None = None
        self.link_down = asyncio.Event()  # set while there is no link; main waits on it to reconnect
        self._hung_up = False
        self._ringing = False  # the current call came in rather than going out
        self._tasks: dict[str, asyncio.Task] = {}
        self._bind(line, dlc)

    def _bind(self, line: HfLine, dlc: rfcomm.DLC) -> None:
        self.line, self.dlc = line, dlc
        self.link_up = True
        self.link_down.clear()
        self._contacts: list[Contact] | None = None  # the phone may have changed them while we were away
        line.on_note = self._on_note
        line.pump.on_dtmf = self._on_dtmf
        connection = self._connection
        connection.on(Connection.EVENT_DISCONNECTION, lambda reason: self._on_disconnect(connection))

    def relink(self, line: HfLine, dlc: rfcomm.DLC) -> None:
        """Carry on over a new HFP line after the old link dropped."""
        self.line.stop()
        self._bind(line, dlc)
        if self.events:
            self.events.link(True)
            self._watch_incoming()

    @property
    def _connection(self) -> Connection:
        return self.dlc.multiplexer.l2cap_channel.connection

    def attach(self, events: PhoneEvents) -> None:
        self.events = events
        self._watch_incoming()

    # --- calls ---

    async def dial(self, number: str) -> None:
        self._stop("incoming")
        self._hung_up = self._ringing = False
        await self._at(self.line.dial, number)

    async def answer(self) -> None:
        await self._at(self.line.answer())

    async def reject(self) -> None:
        await self._at(self.line.reject())

    async def hangup(self) -> None:
        self._hung_up = True
        await self._at(self.line.hangup_all())  # never leave a held call behind

    async def hold(self) -> None:
        await self._at(self.line.protocol.execute_command("AT+CHLD=2"))

    async def resume(self) -> None:
        await self._at(self.line.protocol.execute_command("AT+CHLD=2"))  # swap the held call back in

    async def consult(self, number: str) -> None:
        await self._at(self.line.protocol.execute_command("AT+CHLD=2"))
        self._hung_up = False
        await self._at(self.line.dial, number)

    async def merge(self) -> None:
        await self._at(self.line.protocol.execute_command("AT+CHLD=3"))

    async def speak(self, text: str) -> None:
        if self.tts is None:
            raise PhoneError("no text-to-speech engine")
        try:
            await self._at(self.line.play(await self.tts.synthesize(text, self.line.rate)))
        except (OSError, RuntimeError, CallEnded, wave.Error) as failed:
            raise PhoneError(str(failed) or type(failed).__name__) from None

    async def send_dtmf(self, digits: str) -> None:
        await self._at(self.line.send_dtmf(digits))

    # --- SMS and contacts (OBEX on the HFP RFCOMM session; iOS refuses a second one) ---

    async def sms_send(self, number: str, text: str) -> str:
        async with self._map() as session:
            return await session.send_sms(normalize_number(number), text)

    async def sms_list(self, limit: int) -> list[SmsMessage]:
        async with self._map() as session:
            listed = await session.list_messages(count=limit)
        return [SmsMessage(m.handle, m.sender, None, m.subject, _iso(m.datetime)) for m in listed]

    async def sms_read(self, handle: str) -> SmsMessage:
        async with self._map() as session:
            message = await session.read_message(handle)
        return SmsMessage(handle, message.sender, None, message.text, _iso(""))

    async def contacts(self) -> list[Contact]:
        if self._contacts is None:
            self._contacts = await pull_contacts(self._connection, self.dlc.multiplexer)
        return self._contacts or []

    async def call_log(self, limit: int) -> list[CallLogEntry]:
        return await pull_call_log(self._connection, self.dlc.multiplexer, limit)

    @contextlib.asynccontextmanager
    async def _map(self):
        if not self.link_up:
            raise LinkDown("Bluetooth link to the phone is down")
        try:
            async with map_session(self._connection, self.dlc.multiplexer) as session:
                yield session
        except (OSError, RuntimeError) as failed:
            raise PhoneError(str(failed)) from None

    # --- what the phone reports ---

    def _on_note(self, note: str) -> None:
        if not self.events:
            return
        if note == "answered":
            self.events.call_state("active")
            self._start("listen", self._listen())
        elif note in ("call ended", "not answered"):
            self._stop("listen")
            if note == "not answered":
                reason = "remote" if self._ringing else "failed"
            else:
                reason = "local" if self._hung_up else "remote"
            self.events.call_state("ended", reason)
            self._watch_incoming()

    def _on_dtmf(self, digit: str) -> None:
        if self.events:
            self.events.dtmf(digit)

    def _on_disconnect(self, connection: Connection) -> None:
        if connection is not self._connection:  # a link we already replaced
            return
        self.link_up = False
        self.link_down.set()
        for name in list(self._tasks):
            self._stop(name)
        if self.events:
            self.events.link(False)

    def _watch_incoming(self) -> None:
        self._start("incoming", self._incoming())

    async def _incoming(self) -> None:
        with contextlib.suppress(LinkLost, CallEnded):
            number = await self.line.wait_incoming()
            self._hung_up, self._ringing = False, True
            if self.events:
                self.events.ringing(number or "withheld", None)

    async def _listen(self) -> None:
        """Turn the far end's speech into utterance events for as long as the call is up."""
        if self.stt is None:
            return
        started = asyncio.get_running_loop().time()
        with contextlib.suppress(LinkLost, CallEnded):
            while self.line.tracker.state is CallState.ACTIVE:
                pcm = await self.line.listen_utterance(_UTTERANCE_MAX_S)
                if not len(pcm):
                    continue
                text = await asyncio.to_thread(self.stt.transcribe, pcm)
                end = int((asyncio.get_running_loop().time() - started) * 1000)
                if text and self.events:
                    self.events.utterance(text, max(0, end - len(pcm) * 1000 // 16000), end)

    # --- internals ---

    async def _at(self, command, number: str | None = None):
        try:
            if number is not None:
                command = command(normalize_number(number))
            return await command
        except ValueError as bad:
            raise PhoneError(str(bad)) from None
        except LinkLost as lost:
            raise LinkDown(str(lost)) from None
        except (hfp.HfpProtocolError, TimeoutError, RuntimeError) as refused:
            raise PhoneError(str(refused) or type(refused).__name__) from None

    def _start(self, name: str, coroutine) -> None:
        self._stop(name)
        self._tasks[name] = asyncio.create_task(coroutine)

    def _stop(self, name: str) -> None:
        if task := self._tasks.pop(name, None):
            task.cancel()


def _iso(phone_time: str) -> str:
    """MAP's phone-local 20261026T102233 as ISO-8601; now when the phone gave none."""
    try:
        moment = datetime.strptime(phone_time[:15], "%Y%m%dT%H%M%S")
    except ValueError:
        moment = datetime.now()
    return moment.astimezone().isoformat(timespec="seconds")

