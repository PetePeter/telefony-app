"""A phone line over Bumble's HFP Hands-Free role: AT call control plus SCO audio."""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
import math
import logging
from pathlib import Path

import numpy as np
from bumble import hci, hfp
from bumble.device import Connection, Device, ScoLink

from telefony_app.bt.audio import LIVE_AUDIO_RATE, STT_RATE, AudioPump, resample
from telefony_app.bt.call_state import CallEnded, CallOutcome, CallState, CallTracker
from telefony_app.bt.dtmf import DtmfDecoder
from telefony_app.bt.recording import WavRecorder
from telefony_app.bt.segmenter import Segmenter
from telefony_app.phone import validate_recording_id

_AUDIO_GRACE_SECONDS = 5.0  # time allowed for the phone to open the SCO link
_CALLER_ID_SECONDS = 3.0  # +CLIP follows the first RING, shortly after callsetup=1
HUNG_UP = "telefony hung up"
log = logging.getLogger(__name__)

_CONSULT_OPERATIONS = {
    hfp.CallHoldOperation.RELEASE_ALL_ACTIVE_CALLS,
    hfp.CallHoldOperation.HOLD_ALL_ACTIVE_CALLS,
    hfp.CallHoldOperation.ADD_HELD_CALL,
}
_ACTIVE_AND_HELD = 1  # callheld indicator: one call active, another on hold

# CVSD only for now: the controller transcodes, so the host sees 8 kHz linear PCM.
HF_CONFIGURATION = hfp.HfConfiguration(
    supported_hf_features=[
        hfp.HfFeature.REMOTE_VOLUME_CONTROL,
        hfp.HfFeature.ENHANCED_CALL_STATUS,
        hfp.HfFeature.THREE_WAY_CALLING,
        hfp.HfFeature.ESCO_S4_SETTINGS_SUPPORTED,
    ],
    supported_hf_indicators=[hfp.HfIndicator.BATTERY_LEVEL],
    supported_audio_codecs=[hfp.AudioCodec.CVSD],
)


class ConsultFailed(Exception):
    """The consulted party could not be added; the original call is back off hold."""


class LinkLost(Exception):
    """The Bluetooth link to the phone dropped."""


class HfLine:
    def __init__(self, protocol: hfp.HfProtocol, recording_directory: Path | None = None) -> None:
        self.protocol = protocol
        self.tracker = CallTracker()
        self.rate = LIVE_AUDIO_RATE
        self.pump = AudioPump(DtmfDecoder(self.rate))
        self.recording_directory = recording_directory
        self._recording_id: str | None = None
        self._recorder: WavRecorder | None = None
        self.pump.on_recording_frame = self._record_frame
        self.on_audio_link_lost: Callable[[], None] | None = None
        self._changed = asyncio.Event()
        self._audio = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._caller: str | None = None  # untrusted caller ID of the ringing call
        self._caller_known = asyncio.Event()
        self.on_note: Callable[[str], None] | None = None  # sees call milestones, e.g. for a transcript
        self.tracker.on_change = self._on_state
        protocol.on(protocol.EVENT_AG_INDICATOR, self._on_ag_indicator)
        protocol.on(protocol.EVENT_CLI_NOTIFICATION, self._on_caller_id)
        self._link_lost = asyncio.Event()
        protocol.dlc.multiplexer.l2cap_channel.connection.on(Connection.EVENT_DISCONNECTION, self._on_link_lost)

    async def start(self) -> None:
        """Bring up the service level connection, then handle the phone's unsolicited events."""
        await self.protocol.initiate_slc()
        try:
            await self.protocol.execute_command("AT+CLIP=1")  # caller ID with each RING
        except hfp.HfpProtocolError:
            # iPhones answer ERROR here; wait_incoming falls back to the AT+CLCC call list.
            log.warning("phone refused caller ID notifications (AT+CLIP)")
        if self.protocol.supports_ag_feature(hfp.AgFeature.EC_NR):
            # telefony has no speaker or mic, so there is no echo to cancel; the phone's echo suppression
            # was cutting telefony's words whenever the callee spoke over them.
            try:
                await self.protocol.execute_command("AT+NREC=0")
            except (hfp.HfpProtocolError, TimeoutError):
                log.warning("phone refused to turn off its echo cancelling (AT+NREC=0)")
        self._task = asyncio.create_task(self.protocol.run())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def dial(self, number: str) -> None:
        self.tracker.begin_call()
        log.info("ATD %s (placed by telefony)", number)  # any other outgoing leg was started on the phone itself
        self._note(f"dialed {number}")
        # Real phones may only reply OK once the network has set the call up, so never time out here;
        # callers bound the whole dial-to-answer wait instead.
        await self._unless_link_lost(self.protocol.execute_command(f"ATD{number};", timeout=math.inf))

    async def wait_incoming(self) -> str | None:
        """Wait for the phone to ring; return the caller's number, or None if withheld."""
        while True:
            self.tracker.begin_call()
            self._caller, self._caller_known = None, asyncio.Event()
            try:
                await self.wait_for(CallState.INCOMING, math.inf)
                break
            except CallEnded:
                continue  # a call placed from the phone itself came and went; keep waiting
        try:
            async with asyncio.timeout(_CALLER_ID_SECONDS):
                await self._caller_known.wait()
        except TimeoutError:
            return await self._incoming_number_from_call_list()
        return self._caller

    async def _incoming_number_from_call_list(self) -> str | None:
        try:
            calls = await self.protocol.query_current_calls()
        except (hfp.HfpProtocolError, TimeoutError):
            return None
        incoming = (c for c in calls if c.status is hfp.CallInfoStatus.INCOMING)
        return next((c.number for c in incoming if c.number), None)

    async def answer(self) -> None:
        await self.protocol.answer_incoming_call()

    @property
    def recording_id(self) -> str | None:
        return self._recording_id

    def set_recording_id(self, recording_id: str | None) -> None:
        self._recording_id = validate_recording_id(recording_id)

    def set_live_audio_callback(self, callback: Callable[[bytes], None] | None) -> None:
        """Capture live call frames independently from the optional WAV recorder."""
        self.pump.set_live_audio_callback(callback)

    async def reject(self) -> None:
        await self.protocol.reject_incoming_call()

    async def hangup(self) -> None:
        if self.tracker.state in (CallState.IDLE, CallState.ENDED):
            return
        self._note(HUNG_UP)
        try:
            await self.protocol.terminate_call()
        except TimeoutError:
            # Some phones end the call (+CIEV call=0) but never OK the AT+CHUP.
            if self.tracker.state is not CallState.ENDED:
                raise

    async def send_dtmf(self, digits: str, pause_s: float = 0.5) -> None:
        """Press keys on the call, one AT+VTS each; ',' waits pause_s."""
        for digit in digits:
            if digit == ",":
                await asyncio.sleep(pause_s)
            else:
                await self._unless_link_lost(self.protocol.execute_command(f"AT+VTS={digit}"))

    async def consult(self, number: str, answer_timeout: float) -> None:
        """Hold the current call, dial number and, once answered, merge everyone into one conference."""
        if not _CONSULT_OPERATIONS <= set(self.protocol.supported_ag_call_hold_operations):
            raise ConsultFailed("phone does not support three-way calling")
        await self.protocol.execute_command("AT+CHLD=2")
        finished = self.tracker.setups_finished
        try:
            # As in a first call, the phone may only OK the ATD once the network sets the call up.
            async with asyncio.timeout(answer_timeout):
                await self.dial(number)
                # Wait for the call setup to finish: iPhones report callheld=1 while still dialing.
                await self._wait_until(lambda: self.tracker.setups_finished > finished, math.inf)
        except TimeoutError:
            await self.protocol.execute_command("AT+CHLD=1")  # drop the ringing call, resume the held one
            raise ConsultFailed(f"{number} did not answer") from None
        if self.tracker.held != _ACTIVE_AND_HELD:
            await self.protocol.execute_command("AT+CHLD=2")  # the dialed call failed; take the held one back
            raise ConsultFailed(f"call to {number} failed")
        await self.protocol.execute_command("AT+CHLD=3")
        self._note(f"{number} joined")

    async def hangup_all(self, settle_seconds: float = 3.0) -> None:
        """End every call on the phone, held ones included, so no call outlives a run."""
        if self._link_lost.is_set():
            raise LinkLost("cannot hang up: Bluetooth link to the phone dropped")
        if await self.protocol.query_current_calls():
            self._note(HUNG_UP)
        # Escalate: iPhones may refuse AT+CHUP for a call left over from a conference.
        for _ in range(3):
            for command in ("AT+CHUP", "AT+CHLD=1", "specific", "AT+CHLD=0"):
                calls = await self.protocol.query_current_calls()
                if not calls:
                    return
                # Indexes are read fresh: earlier commands may have ended or renumbered calls.
                for each in [f"AT+CHLD=1{c.index}" for c in calls] if command == "specific" else [command]:
                    with contextlib.suppress(TimeoutError, hfp.HfpProtocolError):
                        await self.protocol.execute_command(each)
                if await self._calls_cleared_within(settle_seconds):
                    return
        raise RuntimeError("calls still up on the phone after hanging up")

    async def _calls_cleared_within(self, seconds: float) -> bool:
        """The phone OKs a hang-up before the call leaves AT+CLCC; give it a moment before escalating."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + seconds
        while await self.protocol.query_current_calls():
            if loop.time() >= deadline:
                return False
            await asyncio.sleep(min(0.25, seconds))
        return True

    async def wait_until_one_party_left(self, poll_seconds: float = 2.0) -> None:
        """Return once a conference is down to a single call. Leaving a conference changes no indicator, so poll AT+CLCC."""
        while len(await self.protocol.query_current_calls()) > 1:
            try:
                await self._wait_until(lambda: False, poll_seconds)
            except TimeoutError:
                pass
        self._note("a party left")

    async def wait_for(self, state: CallState, timeout: float) -> None:
        await self._wait_until(lambda: self.tracker.state is state, timeout)

    async def _unless_link_lost(self, command: Awaitable[object]) -> None:
        """Await an AT command that would otherwise wait forever for a reply over a dead link."""
        lost = asyncio.ensure_future(self._link_lost.wait())
        reply = asyncio.ensure_future(command)
        try:
            await asyncio.wait([reply, lost], return_when=asyncio.FIRST_COMPLETED)
        finally:
            lost.cancel()
        if not reply.done():
            reply.cancel()
            raise LinkLost("Bluetooth link to the phone dropped")
        reply.result()

    async def _wait_until(self, done: Callable[[], bool], timeout: float) -> None:
        async with asyncio.timeout(timeout):
            while not done():
                if self._link_lost.is_set():
                    raise LinkLost("Bluetooth link to the phone dropped")
                if self.tracker.state is CallState.ENDED:
                    raise CallEnded(self.tracker.outcome)
                self._changed.clear()
                await self._changed.wait()

    async def play(self, pcm: np.ndarray) -> None:
        """Queue PCM at the link rate and wait until the controller has consumed it."""
        self.pump.queue_uplink(pcm)
        async with asyncio.timeout(len(pcm) / self.rate + _AUDIO_GRACE_SECONDS):
            while self.pump.uplink_pending:
                await self._next_audio()

    async def listen_utterance(self, max_seconds: float) -> np.ndarray:
        """Return the far end's first utterance as 16 kHz PCM for STT."""
        # Segment at the link rate and resample the whole utterance once: resampling each
        # tiny SCO packet separately would add a filter transient every few milliseconds.
        segmenter = Segmenter(rate=self.rate)
        # Drop audio heard while we were speaking; barge-in is out of scope for goal 1.
        self.pump.take_downlink()
        try:
            async with asyncio.timeout(max_seconds):
                while not (done := segmenter.feed(self.pump.take_downlink())):
                    await self._next_audio()
        except TimeoutError:
            done = segmenter.flush()
        return resample(done[0], self.rate, STT_RATE) if done else np.zeros(0, dtype=np.int16)

    # --- hardware wiring (verified by manual QA, not unit tests) ---

    def attach_sco(self, connection: Connection) -> None:
        """Accept the phone's SCO request and route its audio through the pump."""
        # Bumble emits these on the connection but defines the names on Device.
        connection.on(Device.EVENT_SCO_REQUEST, lambda link_type: self._accept_sco(connection, link_type))
        connection.on(Device.EVENT_SCO_CONNECTION, self._on_sco_connection)

    def _accept_sco(self, connection: Connection, link_type: int) -> None:
        params = hfp.ESCO_PARAMETERS[
            hfp.DefaultCodecParameters.SCO_CVSD_D1
            if link_type == hci.HCI_Connection_Complete_Event.LinkType.SCO
            else hfp.DefaultCodecParameters.ESCO_CVSD_S4
        ]
        if connection.device.host.supports_command(hci.HCI_ENHANCED_ACCEPT_SYNCHRONOUS_CONNECTION_REQUEST_COMMAND):
            command = hci.HCI_Enhanced_Accept_Synchronous_Connection_Request_Command(
                bd_addr=connection.peer_address, **params.asdict()
            )
        else:  # older controllers, e.g. the Actions 10D7:B012 dongle
            command = hci.HCI_Accept_Synchronous_Connection_Request_Command(
                bd_addr=connection.peer_address,
                transmit_bandwidth=params.transmit_bandwidth,
                receive_bandwidth=params.receive_bandwidth,
                max_latency=params.max_latency,
                voice_setting=int(
                    hci.VoiceSetting(
                        input_sample_size=hci.VoiceSetting.InputSampleSize.SIZE_16_BITS,
                        input_data_format=hci.VoiceSetting.InputDataFormat.TWOS_COMPLEMENT,
                    )
                ),
                retransmission_effort=params.retransmission_effort,
                packet_type=params.packet_type,
            )
        connection.cancel_on_disconnection(connection.device.send_async_command(command))

    def _on_sco_connection(self, link: ScoLink) -> None:
        log.info("SCO audio link up: %s", link)
        host = link.device.host
        link.on(ScoLink.EVENT_DISCONNECTION, self._on_sco_disconnection)

        def on_packet(packet: hci.HCI_SynchronousDataPacket) -> None:
            sdu = self.pump.on_packet(packet.data, packet.packet_status)
            host.send_sco_sdu(connection_handle=packet.connection_handle, sdu=sdu)
            self._audio.set()

        link.sink = on_packet

    def _on_sco_disconnection(self, reason: int = 0) -> None:
        log.warning("SCO audio link dropped (reason %d)", reason)
        if self.tracker.state is not CallState.ACTIVE:
            return
        self._finish_recording()
        self._recording_id = None
        if self.on_audio_link_lost:
            self.on_audio_link_lost()
        self.tracker._end()

    async def _next_audio(self) -> None:
        self._audio.clear()
        await self._audio.wait()

    def _on_state(self, state: CallState) -> None:
        log.info("call state: %s", state.value)
        if state is CallState.ACTIVE:
            self._start_recording()
            self._note("answered")
        elif state is CallState.ENDED:
            self._finish_recording()
            self._recording_id = None
            self._note("call ended" if self.tracker.outcome is CallOutcome.COMPLETED else "not answered")
        self._changed.set()

    def _on_link_lost(self, reason: int) -> None:
        log.warning("Bluetooth link to the phone dropped (reason %d)", reason)
        if self.tracker.state is CallState.ACTIVE:
            self._finish_recording()
            self._recording_id = None
        self._link_lost.set()
        self._note("Bluetooth link to the phone lost")
        self._changed.set()

    def _start_recording(self) -> None:
        if self._recorder is not None or self._recording_id is None:
            return
        if self.recording_directory is None:
            log.error("recording %s requested but no recordings directory is configured", self._recording_id)
            self._recording_id = None
            return
        try:
            self._recorder = WavRecorder(self.recording_directory, self._recording_id, self.rate)
        except Exception:
            log.exception("could not start call recording %s", self._recording_id)
            self._recording_id = None

    def _record_frame(self, received: np.ndarray, transmitted: np.ndarray) -> None:
        if self._recorder is None:
            return
        try:
            self._recorder.write_frame(received, transmitted)
        except Exception:
            log.exception("call recording failed; continuing the call without recording")
            self._abort_recording()

    def _finish_recording(self) -> None:
        recorder, self._recorder = self._recorder, None
        if recorder is None:
            return
        try:
            recorder.finish()
        except Exception:
            log.exception("could not finalize call recording")
            recorder.abort()

    def _abort_recording(self) -> None:
        recorder, self._recorder = self._recorder, None
        if recorder is not None:
            recorder.abort()

    def _note(self, event: str) -> None:
        if self.on_note:
            self.on_note(event)

    def _on_caller_id(self, cli: hfp.CallLineIdentification) -> None:
        self._caller = cli.number or None
        self._caller_known.set()

    def _on_ag_indicator(self, state: hfp.AgIndicatorState) -> None:
        self.tracker.on_indicator(state.indicator.value, state.current_status)
        self._changed.set()  # held/setup changes matter to consult without changing the call state
