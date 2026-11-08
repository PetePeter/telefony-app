"""The Phone port over Bluetooth: HFP call control and SCO audio (HfLine), PBAP contacts and call log, MAP SMS."""

import asyncio
import contextlib
import logging
import wave
from collections.abc import Callable
from datetime import datetime

import numpy as np
from bumble import hfp, rfcomm
from bumble.device import Connection

from telefony_app.bt.audio import STT_RATE, resample
from telefony_app.bt.call_state import CallEnded, CallState
from telefony_app.bt.dialing import normalize_number
from telefony_app.bt.hf_line import _ACTIVE_AND_HELD, HfLine, LinkLost
from telefony_app.bt.map_client import map_session
from telefony_app.bt.obex import ObexError, ObexTimeout
from telefony_app.bt.pbap import pull_call_log, pull_contacts
from telefony_app.bt.rolling import RollingTranscript
from telefony_app.phone import (
    CallLogEntry,
    CallMaybePlaced,
    CallNotPlaced,
    Contact,
    LinkDown,
    Phone,
    PhoneError,
    PhoneEvents,
    SmsMessage,
)
from telefony_app.voice import Stt, Tts

log = logging.getLogger(__name__)
_ATD_NOT_PLACED = {"ERROR", "NO CARRIER"}  # an immediate refusal; BUSY and NO ANSWER mean the call went out
_UTTERANCE_MAX_S = 30.0
_CONSULT_ANSWER_S = 60.0  # a consulted party who has not answered by then is dropped
_FINAL_UTTERANCE_S = 20.0  # whisper on a 30 s utterance takes a few seconds on CPU


class BluetoothPhone(Phone):
    live_audio_available = True
    line: HfLine  # bound by the first link; PhoneService checks link_up before using it
    dlc: rfcomm.DLC

    def __init__(self, line: HfLine | None, dlc: rfcomm.DLC | None, tts: Tts | None, stt: Stt | None) -> None:
        """Without a line the phone starts unlinked (it was unreachable at startup); relink() brings it up."""
        self.tts, self.stt = tts, stt
        self.tts_available, self.stt_available = tts is not None, stt is not None
        self.events: PhoneEvents | None = None
        self.link_up = False
        self.link_down = asyncio.Event()  # set while there is no link; main waits on it to reconnect
        self.link_down.set()
        self._hung_up = False
        self._ringing = False  # the current call came in rather than going out
        self._tasks: dict[str, asyncio.Task] = {}
        self._speech_generation = 0
        self._live_audio_callback: Callable[[bytes], None] | None = None
        self._obex = asyncio.Lock()  # one OBEX session at a time: bumble opens one DLC at a time on the shared multiplexer
        if line is not None and dlc is not None:
            self._bind(line, dlc)

    def _bind(self, line: HfLine, dlc: rfcomm.DLC) -> None:
        self.line, self.dlc = line, dlc
        self.link_up = True
        self.link_down.clear()
        line.set_live_audio_callback(self._live_audio_callback)
        self._contacts: list[Contact] | None = None  # the phone may have changed them while we were away
        line.on_note = self._on_note
        line.pump.on_dtmf = self._on_dtmf
        connection = self._connection
        connection.on(Connection.EVENT_DISCONNECTION, lambda reason: self._on_disconnect(connection))
        self._start("contacts", self._pull_contacts())  # also makes iOS offer its "Sync Contacts" switch

    def relink(self, line: HfLine, dlc: rfcomm.DLC) -> None:
        """Carry on over a new HFP line after the old link dropped, or bring up the first one."""
        if hasattr(self, "line"):  # not on the first link
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
        if self.link_up:
            self._watch_incoming()

    def set_live_audio_callback(self, callback: Callable[[bytes], None] | None) -> None:
        self._live_audio_callback = callback
        if self.link_up:
            self.line.set_live_audio_callback(callback)

    # --- calls ---

    async def dial(self, number: str, recording_id: str | None = None) -> None:
        self._stop("incoming")
        self._hung_up = self._ringing = False
        self.line.set_recording_id(recording_id)
        try:
            await self._at(self._place, number)
        except LinkDown as lost:  # ATD went out first: the phone may be calling
            raise CallMaybePlaced(f"call may have been placed: {lost}") from None

    async def _place(self, number: str) -> None:
        try:
            await self.line.dial(number)
        except hfp.HfpProtocolError as refused:
            if refused.error_name in _ATD_NOT_PLACED:
                raise CallNotPlaced(refused.error_name) from None
            raise

    async def answer(self, recording_id: str | None = None) -> None:
        self.line.set_recording_id(recording_id)
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
        finished = self.line.tracker.setups_finished
        await self._at(self.line.dial, number)
        self._start("consult", self._watch_consult(finished))

    async def _watch_consult(self, finished: int) -> None:
        """Report the consulted call answered or failed: the call indicator stays 1 throughout, so only the
        setup finishing says which (one active and one held = answered)."""
        tracker = self.line.tracker
        with contextlib.suppress(CallEnded, LinkLost):
            try:
                await self.line._wait_until(lambda: tracker.setups_finished > finished, _CONSULT_ANSWER_S)
            except TimeoutError:
                await self._at(self.line.protocol.execute_command("AT+CHLD=1"))  # drop the ringing call, resume the held one
            else:
                if tracker.held == _ACTIVE_AND_HELD:
                    if self.events:
                        self.events.call_state("active")
                    return
                await self._at(self.line.protocol.execute_command("AT+CHLD=2"))  # take the held call back
            if self.events:
                self.events.call_state("ended", "failed")

    async def release_active(self) -> None:
        self._stop("consult")
        await self._at(self.line.protocol.execute_command("AT+CHLD=1"))

    async def merge(self) -> None:
        await self._at(self.line.protocol.execute_command("AT+CHLD=3"))

    async def speak(self, text: str) -> None:
        if self.tts is None:
            raise PhoneError("no text-to-speech engine")
        generation = self._speech_generation
        try:
            pcm = await self.tts.synthesize(text, self.line.rate)
            if generation != self._speech_generation:  # the far end spoke first: nothing was heard
                raise PhoneError("interrupted before it played")
            await self._at(self.line.play(pcm))
        except (OSError, RuntimeError, CallEnded, wave.Error) as failed:
            raise PhoneError(str(failed) or type(failed).__name__) from None

    def _on_speech_onset(self) -> None:
        """The far end started talking: stop our speech and say so at once, before any transcript."""
        self._speech_generation += 1
        self.line.stop_playback()
        if self.events:
            self.events.speech(True)

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
        async with self._obex:
            if self._contacts is None:
                self._contacts = await pull_contacts(self._connection, self.dlc.multiplexer)
        return self._contacts or []

    async def _pull_contacts(self) -> None:
        try:
            await self.contacts()
        except (OSError, RuntimeError, LookupError, ObexError) as failed:
            log.info("phone contacts not available yet: %s", failed)

    async def call_log(self, limit: int) -> list[CallLogEntry]:
        async with self._obex:
            return await pull_call_log(self._connection, self.dlc.multiplexer, limit)

    @contextlib.asynccontextmanager
    async def _map(self):
        if not self.link_up:
            raise LinkDown("Bluetooth link to the phone is down")
        try:
            async with self._obex, map_session(self._connection, self.dlc.multiplexer) as session:
                yield session
        except ObexTimeout as silent:  # the link is going or gone; the drop usually follows
            raise LinkDown(str(silent)) from None
        except (OSError, RuntimeError, LookupError, ObexError) as failed:  # refused or not offered, e.g. iOS before Show Notifications
            raise PhoneError(str(failed)) from None

    # --- what the phone reports ---

    def _on_note(self, note: str) -> None:
        if not self.events:
            return
        if note == "answered":
            self.events.call_state("active")
            self._start("listen", self._listen())
        elif note in ("call ended", "not answered"):
            if note == "not answered":
                reason = "remote" if self._ringing else "failed"
            else:
                reason = "local" if self._hung_up else "remote"
            self._start("ending", self._end_call(reason))

    async def _end_call(self, reason: str) -> None:
        """Report the end only after the last words are transcribed: a message is often cut off by the hangup."""
        if listen := self._tasks.pop("listen", None):
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(listen, _FINAL_UTTERANCE_S)
        if self.events:
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
            if name != "ending":  # the end of a call still gets reported
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
        """Turn the far end's speech into utterance events for as long as the call is up. While they talk,
        the utterance is transcribed in rolling windows (partial events), so at the pause only the tail is left."""
        if self.stt is None:
            return
        started = asyncio.get_running_loop().time()
        with contextlib.suppress(LinkLost, CallEnded):
            while self.line.tracker.state is CallState.ACTIVE:
                rolling = RollingTranscript(self.line.rate)
                window_work: list[asyncio.Task] = []
                latest: list[np.ndarray] = []  # the utterance so far at the link rate, as the windows see it

                def on_progress(heard: np.ndarray) -> None:
                    latest[:] = [heard]
                    if window_work and not window_work[-1].done():
                        return  # one window at a time; the next catches up
                    if (window := rolling.next_window(heard)) is not None:
                        window_work.append(asyncio.create_task(self._transcribe_window(rolling, window)))

                pcm = await self.line.listen_utterance(_UTTERANCE_MAX_S, self._on_speech_onset, on_progress)
                if not len(pcm):
                    continue
                if self.events:
                    self.events.speech(False)
                began = asyncio.get_running_loop().time()
                if window_work:
                    for failed in await asyncio.gather(*window_work, return_exceptions=True):
                        if isinstance(failed, Exception):
                            log.warning("rolling transcription failed: %s", failed)
                # Windows a slow STT could not take while they talked: every call stays one window long, as
                # whisper.cpp hears only its audio context and would drop the rest of a long tail unheard.
                while latest and (window := rolling.next_window(latest[0])) is not None:
                    try:
                        await self._transcribe_window(rolling, window)
                    except Exception as failed:
                        log.warning("rolling transcription failed: %s", failed)
                tail = pcm[int(rolling.covered_s * STT_RATE) :]
                text = rolling.add(await asyncio.to_thread(self.stt.transcribe, tail)) if len(tail) else rolling.text
                log.info("transcribed %.1f s of speech; %.2f s after the pause", len(pcm) / STT_RATE, asyncio.get_running_loop().time() - began)
                end = int((asyncio.get_running_loop().time() - started) * 1000)
                if self.events:  # empty text too: it tells the client the turn is fully transcribed
                    self.events.utterance(text, max(0, end - len(pcm) * 1000 // STT_RATE), end)

    async def _transcribe_window(self, rolling: RollingTranscript, window: np.ndarray) -> None:
        assert self.stt is not None
        text = await asyncio.to_thread(self.stt.transcribe, resample(window, self.line.rate, STT_RATE))
        if text and self.events:
            self.events.partial(rolling.add(text))

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

