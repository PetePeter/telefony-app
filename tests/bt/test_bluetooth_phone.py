"""BluetoothPhone surviving a dropped link: the same object is handed a new HFP line and works again."""

import asyncio
from typing import cast

import numpy as np
import pytest
from bumble import hci, hfp

from telefony_app.bt.bluetooth_phone import BluetoothPhone
from telefony_app.bt.call_state import CallOutcome, CallState
from telefony_app.phone import CallMaybePlaced, CallNotPlaced, PhoneError, Retryable
from telefony_app.voice import WhisperStt
from tests.bt.test_hf_line import connect


class Events:
    def __init__(self) -> None:
        self.links: list[bool] = []
        self.call_states: list[tuple[str, str | None]] = []
        self.utterances: list[str] = []

    def link(self, up: bool) -> None:
        self.links.append(up)

    def ringing(self, number: str, name: str | None) -> None: ...
    def call_state(self, state: str, reason: str | None = None) -> None:
        self.call_states.append((state, reason))
    def utterance(self, text: str, start_ms: int, end_ms: int) -> None:
        self.utterances.append(text)
    def speech(self, speaking: bool) -> None: ...
    def partial(self, text: str) -> None: ...
    def dtmf(self, digit: str) -> None: ...
    def sms_received(self, handle: str, number: str, text: str) -> None: ...


def connection_of(line):
    return line.protocol.dlc.multiplexer.l2cap_channel.connection


async def until(condition) -> None:
    async with asyncio.timeout(2):
        while not condition():
            await asyncio.sleep(0.01)


async def test_relinked_phone_reports_link_up_and_dials_on_the_new_line():
    _, old_line = await connect()
    phone, events = BluetoothPhone(old_line, old_line.protocol.dlc, None, None), Events()
    phone.attach(events)
    await connection_of(old_line).disconnect()
    await until(lambda: not phone.link_up)
    assert phone.link_down.is_set()

    ag, new_line = await connect()
    phone.relink(new_line, new_line.protocol.dlc)

    assert phone.link_up and not phone.link_down.is_set()
    assert events.links == [False, True]
    dialed = asyncio.get_running_loop().create_future()
    ag.once(ag.EVENT_DIAL, dialed.set_result)
    await phone.dial("0491570110")
    assert await asyncio.wait_for(dialed, 1) == "0491570110;"
    new_line.stop()


async def test_a_phone_that_never_linked_reports_down_then_works_on_its_first_line():
    # Seen live: with the phone unreachable at startup, PhAP never opened, so Timbre could not even see "link down".
    phone, events = BluetoothPhone(None, None, None, None), Events()
    phone.attach(events)
    phone.set_live_audio_callback(None)
    assert not phone.link_up and phone.link_down.is_set()

    ag, line = await connect()
    phone.relink(line, line.protocol.dlc)

    assert phone.link_up and events.links == [True]
    dialed = asyncio.get_running_loop().create_future()
    ag.once(ag.EVENT_DIAL, dialed.set_result)
    await phone.dial("0491570110")
    assert await asyncio.wait_for(dialed, 1) == "0491570110;"
    line.stop()


async def test_late_disconnect_of_the_old_link_does_not_take_the_new_one_down():
    _, old_line = await connect()
    phone = BluetoothPhone(old_line, old_line.protocol.dlc, None, None)
    phone.attach(Events())
    _, new_line = await connect()
    phone.relink(new_line, new_line.protocol.dlc)

    await connection_of(old_line).disconnect()
    await asyncio.sleep(0.2)

    assert phone.link_up
    new_line.stop()


async def test_call_recording_id_is_set_before_dial():
    _, line = await connect()
    phone = BluetoothPhone(line, line.protocol.dlc, None, None)
    await phone.dial("0491570110", "interaction-2")
    assert line.recording_id == "interaction-2"
    line.stop()


async def test_sco_loss_keeps_service_call_until_hfp_call_indicator_ends_it():
    ag, line = await connect()
    phone, events = BluetoothPhone(line, line.protocol.dlc, None, None), Events()
    phone.events = events
    phone._watch_incoming = lambda: None
    hangups = []
    ag.on(ag.EVENT_HANG_UP, lambda: hangups.append(True))
    try:
        ag.update_ag_indicator(hfp.AgIndicator.CALL, 1)
        await line.wait_for(CallState.ACTIVE, 1)
        line._on_sco_disconnection(0x13)

        assert line.tracker.state is CallState.ACTIVE
        assert events.call_states == [("active", None)]
        assert hangups == []

        ag.update_ag_indicator(hfp.AgIndicator.CALL, 0)
        await line.wait_for(CallState.ENDED, 1)
        assert events.call_states == [("active", None), ("ended", "remote")]
        assert hangups == []
    finally:
        phone._stop("incoming")
        phone._stop("listen")
        line.stop()


async def test_phone_without_message_access_fails_sms_as_a_phone_error():
    _, line = await connect()  # the fake AG offers no MAP server
    phone = BluetoothPhone(line, line.protocol.dlc, None, None)

    with pytest.raises(PhoneError):
        await phone.sms_list(5)
    line.stop()


async def test_voiced_onset_interrupts_playback_and_still_reaches_stt():
    class Tts:
        async def synthesize(self, text: str, rate: int) -> np.ndarray:
            return np.ones(rate * 20, dtype=np.int16)

    class Stt:
        def transcribe(self, pcm: np.ndarray) -> str:
            return "Please answer this question."

    _, line = await connect()
    phone = BluetoothPhone(line, line.protocol.dlc, Tts(), cast(WhisperStt, Stt()))
    events = Events()
    phone._watch_incoming = lambda: None
    phone.attach(events)
    line.tracker.state = CallState.ACTIVE
    phone._on_note("answered")

    try:
        speaking = asyncio.create_task(phone.speak("A long opening message."))
        await until(lambda: line.pump.uplink_pending > 0)

        voiced = np.full(160, 8000, dtype=np.int16).tobytes()
        for _ in range(13):  # enough frames for a 220 ms VAD onset after DTMF buffering
            line.pump.on_packet(voiced, hci.HCI_SynchronousDataPacket.Status.CORRECTLY_RECEIVED_DATA)
            line._downlink_audio.set()
            await asyncio.sleep(0.01)

        await asyncio.wait_for(speaking, 1)
        assert line.pump.uplink_pending == 0

        for _ in range(12):
            line.pump.on_packet(voiced, hci.HCI_SynchronousDataPacket.Status.CORRECTLY_RECEIVED_DATA)
            line._downlink_audio.set()
            await asyncio.sleep(0.01)
        silence = bytes(160 * 2)
        for _ in range(80):  # 1.5 s end silence plus DTMF decoder buffering
            line.pump.on_packet(silence, hci.HCI_SynchronousDataPacket.Status.CORRECTLY_RECEIVED_DATA)
            line._downlink_audio.set()
            await asyncio.sleep(0.01)

        await until(lambda: bool(events.utterances))
        assert events.utterances == ["Please answer this question."]
    finally:
        phone._stop("listen")
        phone._stop("contacts")
        line.stop()


@pytest.mark.parametrize("result", ["ERROR", "NO CARRIER"])
async def test_atd_refused_at_once_means_no_call_was_placed(result):
    ag, line = await connect()
    ag._on_d = lambda number: ag.send_response(result)
    phone = BluetoothPhone(line, line.protocol.dlc, None, None)
    with pytest.raises(CallNotPlaced):
        await phone.dial("0491570110")
    line.stop()


@pytest.mark.parametrize("result", ["BUSY", "NO ANSWER"])
async def test_busy_or_no_answer_is_a_placed_call_and_never_retryable(result):
    ag, line = await connect()
    ag._on_d = lambda number: ag.send_response(result)
    phone = BluetoothPhone(line, line.protocol.dlc, None, None)
    with pytest.raises(PhoneError) as failed:
        await phone.dial("0491570110")
    assert not isinstance(failed.value, Retryable)
    line.stop()


async def test_link_lost_after_atd_went_out_means_the_call_may_have_been_placed():
    ag, line = await connect()
    ag._on_d = lambda number: None  # the phone never answers the ATD
    phone = BluetoothPhone(line, line.protocol.dlc, None, None)
    dialing = asyncio.create_task(phone.dial("0491570110"))
    await asyncio.sleep(0.1)
    await connection_of(line).disconnect()
    with pytest.raises(CallMaybePlaced):
        await asyncio.wait_for(dialing, 2)


async def test_speech_interrupted_before_it_plays_fails_rather_than_counting_as_spoken():
    # Seen live: the far end kept saying "Hello?" while a phrase rendered; it was dropped yet reported spoken.
    _, line = await connect()
    phone: BluetoothPhone

    class Tts:
        async def synthesize(self, text: str, rate: int) -> np.ndarray:
            phone._on_speech_onset()  # the far end starts talking while we render
            return np.ones(rate, dtype=np.int16)

    phone = BluetoothPhone(line, line.protocol.dlc, Tts(), None)
    with pytest.raises(PhoneError, match="interrupted"):
        await phone.speak("Hi, it is Oscar.")
    line.stop()


class OrderedEvents(Events):
    """Events in arrival order, so a test can see an utterance land before the call ends."""

    def __init__(self) -> None:
        super().__init__()
        self.log: list[str] = []

    def call_state(self, state: str, reason: str | None = None) -> None:
        super().call_state(state, reason)
        self.log.append(f"state:{state}")

    def utterance(self, text: str, start_ms: int, end_ms: int) -> None:
        super().utterance(text, start_ms, end_ms)
        self.log.append(f"utterance:{text}")

    def speech(self, speaking: bool) -> None:
        self.log.append("speaking" if speaking else "paused")


async def _hang_up_after_speech(stt, pause_frames: int = 0) -> OrderedEvents:
    """The caller speaks, maybe pauses, and hangs up: the voicemail case."""
    _, line = await connect()
    phone = BluetoothPhone(line, line.protocol.dlc, None, cast(WhisperStt, stt))
    events = OrderedEvents()
    phone._watch_incoming = lambda: None
    phone.attach(events)
    line.tracker.state = CallState.ACTIVE
    phone._on_note("answered")
    try:
        voiced = np.full(160, 8000, dtype=np.int16).tobytes()
        for _ in range(40):
            line.pump.on_packet(voiced, hci.HCI_SynchronousDataPacket.Status.CORRECTLY_RECEIVED_DATA)
            line._downlink_audio.set()
            await asyncio.sleep(0.01)
        silence = bytes(160 * 2)
        for _ in range(pause_frames):
            line.pump.on_packet(silence, hci.HCI_SynchronousDataPacket.Status.CORRECTLY_RECEIVED_DATA)
            line._downlink_audio.set()
            await asyncio.sleep(0.01)
        line.tracker.state, line.tracker.outcome = CallState.ENDED, CallOutcome.COMPLETED
        line._on_state(CallState.ENDED)
        await until(lambda: "state:ended" in events.log)
        return events
    finally:
        phone._stop("listen")
        line.stop()


async def test_speech_cut_off_by_hangup_is_still_transcribed_before_the_call_ends():
    class Stt:
        def transcribe(self, pcm: np.ndarray) -> str:
            return "Call me back."

    events = await _hang_up_after_speech(Stt())
    assert events.log == ["state:active", "speaking", "paused", "utterance:Call me back.", "state:ended"]


async def test_transcription_in_flight_at_hangup_lands_before_the_call_ends():
    class SlowStt:
        def transcribe(self, pcm: np.ndarray) -> str:
            import time
            time.sleep(0.5)  # whisper still working when the caller hangs up
            return "Call me back."

    events = await _hang_up_after_speech(SlowStt(), pause_frames=80)
    assert events.log == ["state:active", "speaking", "paused", "utterance:Call me back.", "state:ended"]


async def test_long_speech_streams_partials_and_ends_with_one_merged_utterance():
    class WindowStt:
        """Each call hears the next overlapping stretch of what the caller said."""

        def __init__(self) -> None:
            self.heard = iter(["could you call", "call the dentist", "dentist for me please"])

        def transcribe(self, pcm: np.ndarray) -> str:
            return next(self.heard, "")

    class Recorder(OrderedEvents):
        def partial(self, text: str) -> None:
            self.log.append(f"partial:{text}")

    _, line = await connect()
    phone = BluetoothPhone(line, line.protocol.dlc, None, cast(WhisperStt, WindowStt()))
    events = Recorder()
    phone._watch_incoming = lambda: None
    phone.attach(events)
    line.tracker.state = CallState.ACTIVE
    phone._on_note("answered")
    try:
        packet_s = 160 / line.rate
        voiced = np.full(160, 8000, dtype=np.int16).tobytes()
        for _ in range(int(5.5 / packet_s)):  # 5.5 s of speech: two rolling windows, then a tail
            line.pump.on_packet(voiced, hci.HCI_SynchronousDataPacket.Status.CORRECTLY_RECEIVED_DATA)
            line._downlink_audio.set()
            await asyncio.sleep(0.002)
        silence = bytes(160 * 2)
        for _ in range(int(1.6 / packet_s)):
            line.pump.on_packet(silence, hci.HCI_SynchronousDataPacket.Status.CORRECTLY_RECEIVED_DATA)
            line._downlink_audio.set()
            await asyncio.sleep(0.002)
        await until(lambda: bool(events.utterances))

        assert events.utterances == ["could you call the dentist for me please"]
        assert [entry for entry in events.log if entry.startswith("partial:")] == [
            "partial:could you call", "partial:could you call the dentist",
        ]
        assert events.log.index("paused") < events.log.index("utterance:could you call the dentist for me please")
    finally:
        phone._stop("listen")
        line.stop()
