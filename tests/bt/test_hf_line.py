"""HfLine against Bumble's real Audio Gateway (phone side) over an in-memory Bluetooth link."""

import asyncio
import wave
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from bumble import hfp, rfcomm
from bumble.device import ScoLink
from bumble.testing.test_utils import TwoDevices
from bumble.hci import HCI_SynchronousDataPacket

from telefony_app.bt.call_state import CallEnded, CallState
from telefony_app.bt.hf_line import HF_CONFIGURATION, ConsultFailed, HfLine, LinkLost

AG_CONFIGURATION = hfp.AgConfiguration(
    supported_ag_features=[hfp.AgFeature.ENHANCED_CALL_STATUS, hfp.AgFeature.REJECT_CALL],
    supported_ag_indicators=[
        hfp.AgIndicatorState.service(),
        hfp.AgIndicatorState.call(),
        hfp.AgIndicatorState.callsetup(),
        hfp.AgIndicatorState.callheld(),
        hfp.AgIndicatorState.signal(),
        hfp.AgIndicatorState.roam(),
        hfp.AgIndicatorState.battchg(),
    ],
    supported_hf_indicators=[hfp.HfIndicator.BATTERY_LEVEL],
    supported_ag_call_hold_operations=[],
    supported_audio_codecs=[hfp.AudioCodec.CVSD],
)


async def connect(
    ag_configuration: hfp.AgConfiguration = AG_CONFIGURATION,
    clip_supported: bool = True,
    recording_directory=None,
) -> tuple[hfp.AgProtocol, HfLine]:
    devices = TwoDevices()
    await devices.setup_connection()
    dlc_future = asyncio.get_running_loop().create_future()
    channel = rfcomm.Server(devices[0]).listen(dlc_future.set_result)
    mux = await rfcomm.Client(devices.connections[1]).start()
    hf_dlc = await mux.open_dlc(channel)
    ag = hfp.AgProtocol(await dlc_future, ag_configuration)
    if not clip_supported:  # seen on an iPhone: AT+CLIP=1 answered with ERROR
        ag._on_clip = lambda enabled: ag.send_error()
    slc = asyncio.get_running_loop().create_future()
    ag.once(ag.EVENT_SLC_COMPLETE, lambda: slc.set_result(None))
    line = HfLine(hfp.HfProtocol(hf_dlc, HF_CONFIGURATION), recording_directory=recording_directory)
    await asyncio.wait_for(line.start(), 2)  # returns only once the SLC is up
    assert slc.done()
    return ag, line


@pytest.fixture
async def phone_and_line():
    ag, line = await connect()
    yield ag, line
    line.stop()


async def test_phone_rejecting_caller_id_still_connects_and_reports_caller_via_clcc():
    ag, line = await connect(clip_supported=False)
    try:
        ag.calls = [
            hfp.CallInfo(
                index=1,
                direction=hfp.CallInfoDirection.MOBILE_TERMINATED_CALL,
                status=hfp.CallInfoStatus.INCOMING,
                mode=hfp.CallInfoMode.VOICE,
                multi_party=hfp.CallInfoMultiParty.NOT_IN_CONFERENCE,
                number="0491570157",
            )
        ]
        incoming = asyncio.create_task(line.wait_incoming())
        await asyncio.sleep(0.05)
        ag.update_ag_indicator(hfp.AgIndicator.CALL_SETUP, 1)
        assert await asyncio.wait_for(incoming, 10) == "0491570157"
    finally:
        line.stop()


async def test_active_call_records_at_the_audio_pump_and_finalizes_normally(tmp_path):
    _, line = await connect(recording_directory=tmp_path)
    try:
        line.set_recording_id("active")
        line._on_state(CallState.ACTIVE)
        line.pump.queue_uplink(np.array([8, 9], dtype=np.int16))
        line.pump.on_packet(np.array([1, 2], dtype=np.int16).tobytes(), HCI_SynchronousDataPacket.Status.CORRECTLY_RECEIVED_DATA)
        line._on_state(CallState.ENDED)

        with wave.open(str(tmp_path / "active.wav"), "rb") as wav:
            assert wav.getparams()[:3] == (2, 2, 8000)
            assert np.frombuffer(wav.readframes(2), dtype="<i2").reshape(-1, 2).tolist() == [[1, 8], [2, 9]]
    finally:
        line.stop()


async def test_sco_disconnect_finalizes_partial_and_reports_link_lost(tmp_path):
    class Link:
        def __init__(self):
            self.device = SimpleNamespace(host=SimpleNamespace(send_sco_sdu=lambda **kwargs: None))
            self.handlers = {}
            self.sink = None

        def on(self, event, callback):
            self.handlers[event] = callback

    _, line = await connect(recording_directory=tmp_path)
    lost = []
    line.on_audio_link_lost = lambda: lost.append(True)
    line.set_recording_id("partial")
    line.tracker.state = CallState.ACTIVE
    line._on_state(CallState.ACTIVE)
    link = Link()
    line._on_sco_connection(cast(ScoLink, link))
    line.pump.on_packet(np.array([3, 4], dtype=np.int16).tobytes(), HCI_SynchronousDataPacket.Status.CORRECTLY_RECEIVED_DATA)
    link.handlers[ScoLink.EVENT_DISCONNECTION](0x13)

    try:
        assert lost == [True]
        assert line.tracker.state is CallState.ENDED
        with wave.open(str(tmp_path / "partial.wav"), "rb") as wav:
            assert np.frombuffer(wav.readframes(2), dtype="<i2").reshape(-1, 2).tolist() == [[3, 0], [4, 0]]
    finally:
        line.stop()


async def test_acl_disconnect_finalizes_partial_recording(tmp_path):
    _, line = await connect(recording_directory=tmp_path)
    line.set_recording_id("acl_partial")
    line.tracker.state = CallState.ACTIVE
    line._on_state(CallState.ACTIVE)
    line.pump.on_packet(np.array([5], dtype=np.int16).tobytes(), HCI_SynchronousDataPacket.Status.CORRECTLY_RECEIVED_DATA)

    try:
        line._on_link_lost(0x13)
        with wave.open(str(tmp_path / "acl_partial.wav"), "rb") as wav:
            assert np.frombuffer(wav.readframes(1), dtype="<i2").reshape(-1, 2).tolist() == [[5, 0]]
    finally:
        line.stop()


async def test_recording_start_failure_does_not_break_sco_audio(tmp_path):
    existing_part = tmp_path / "collision.wav.part"
    existing_part.write_bytes(b"preserve")
    _, line = await connect(recording_directory=tmp_path)
    try:
        line.set_recording_id("collision")
        line.tracker.state = CallState.ACTIVE
        line._on_state(CallState.ACTIVE)
        out = line.pump.on_packet(np.array([7], dtype=np.int16).tobytes(), HCI_SynchronousDataPacket.Status.CORRECTLY_RECEIVED_DATA)
        assert out == bytes(2)
        assert existing_part.read_bytes() == b"preserve"
    finally:
        line.stop()


async def test_unanswered_or_unrecorded_call_does_not_create_a_recording(tmp_path):
    _, line = await connect(recording_directory=tmp_path)
    try:
        line.set_recording_id("unanswered")
        line._on_state(CallState.INCOMING)
        line._on_state(CallState.ENDED)
        line.set_recording_id(None)
        line._on_state(CallState.ACTIVE)
        line.pump.on_packet(np.array([1], dtype=np.int16).tobytes(), HCI_SynchronousDataPacket.Status.CORRECTLY_RECEIVED_DATA)
        line._on_state(CallState.ENDED)
        assert list(tmp_path.iterdir()) == []
    finally:
        line.stop()


async def test_dial_sends_atd_to_phone(phone_and_line):
    ag, line = phone_and_line
    dialed = asyncio.get_running_loop().create_future()
    ag.once(ag.EVENT_DIAL, dialed.set_result)
    await line.dial("+447700900123")
    # ATD's trailing ';' (voice call) reaches the phone verbatim.
    assert await asyncio.wait_for(dialed, 1) == "+447700900123;"


async def test_phone_indicators_drive_call_state(phone_and_line):
    ag, line = phone_and_line
    ag.update_ag_indicator(hfp.AgIndicator.CALL_SETUP, 2)
    await line.wait_for(CallState.DIALING, 1)
    ag.update_ag_indicator(hfp.AgIndicator.CALL, 1)
    await line.wait_for(CallState.ACTIVE, 1)


async def test_rejected_call_ends_wait_immediately(phone_and_line):
    ag, line = phone_and_line
    ag.update_ag_indicator(hfp.AgIndicator.CALL_SETUP, 2)
    ag.update_ag_indicator(hfp.AgIndicator.CALL_SETUP, 3)
    ag.update_ag_indicator(hfp.AgIndicator.CALL_SETUP, 0)
    with pytest.raises(CallEnded):
        await line.wait_for(CallState.ACTIVE, timeout=5)


async def test_second_call_not_failed_by_previous_ended_state(phone_and_line):
    ag, line = phone_and_line
    for indicator, value in [(hfp.AgIndicator.CALL_SETUP, 2), (hfp.AgIndicator.CALL_SETUP, 0)]:
        ag.update_ag_indicator(indicator, value)
    await line.wait_for(CallState.ENDED, 1)
    # ATD's OK can arrive before the phone's first callsetup update for the new call.
    await line.dial("123")
    ag.update_ag_indicator(hfp.AgIndicator.CALL_SETUP, 2)
    ag.update_ag_indicator(hfp.AgIndicator.CALL, 1)
    await line.wait_for(CallState.ACTIVE, 1)


async def test_hangup_succeeds_when_phone_ends_call_without_ok(phone_and_line):
    # Seen on a real Android phone: AT+CHUP gets +CIEV call=0 but never an OK.
    ag, line = phone_and_line
    ag.update_ag_indicator(hfp.AgIndicator.CALL, 1)
    await line.wait_for(CallState.ACTIVE, 1)
    ag._on_chup = lambda: ag.update_ag_indicator(hfp.AgIndicator.CALL, 0)
    await asyncio.wait_for(line.hangup(), 3)
    assert line.tracker.state is CallState.ENDED


async def test_hangup_with_no_call_is_a_no_op(phone_and_line):
    _, line = phone_and_line
    await line.hangup()


async def test_hangup_sends_chup_to_phone(phone_and_line):
    ag, line = phone_and_line
    ag.update_ag_indicator(hfp.AgIndicator.CALL, 1)
    await line.wait_for(CallState.ACTIVE, 1)
    hung = asyncio.get_running_loop().create_future()
    ag.once(ag.EVENT_HANG_UP, lambda: hung.set_result(True))
    await line.hangup()
    assert await asyncio.wait_for(hung, 1)


THREE_WAY_AG_CONFIGURATION = hfp.AgConfiguration(
    supported_ag_features=[*AG_CONFIGURATION.supported_ag_features, hfp.AgFeature.THREE_WAY_CALLING],
    supported_ag_indicators=AG_CONFIGURATION.supported_ag_indicators,
    supported_hf_indicators=AG_CONFIGURATION.supported_hf_indicators,
    supported_ag_call_hold_operations=[
        hfp.CallHoldOperation.RELEASE_ALL_HELD_CALLS,
        hfp.CallHoldOperation.RELEASE_ALL_ACTIVE_CALLS,
        hfp.CallHoldOperation.RELEASE_SPECIFIC_CALL,
        hfp.CallHoldOperation.HOLD_ALL_ACTIVE_CALLS,
        hfp.CallHoldOperation.ADD_HELD_CALL,
    ],
    supported_audio_codecs=AG_CONFIGURATION.supported_audio_codecs,
)


async def active_three_way_call() -> tuple[hfp.AgProtocol, HfLine, list[str]]:
    """A call in progress on a three-way capable phone; returns the log of CHLD/ATD commands it receives."""
    ag, line = await connect(THREE_WAY_AG_CONFIGURATION)
    commands: list[str] = []
    ag.on(ag.EVENT_CALL_HOLD, lambda operation, _index: commands.append(f"CHLD={operation.value}"))
    ag.on(ag.EVENT_DIAL, lambda number: commands.append(f"ATD{number}"))
    ag.update_ag_indicator(hfp.AgIndicator.CALL, 1)
    await line.wait_for(CallState.ACTIVE, 1)
    return ag, line, commands


async def test_consult_holds_dials_and_merges_once_answered():
    ag, line, commands = await active_three_way_call()
    try:
        consult = asyncio.create_task(line.consult("0491570157", answer_timeout=5))
        await asyncio.sleep(0.1)
        ag.update_ag_indicator(hfp.AgIndicator.CALL_HELD, 2)
        ag.update_ag_indicator(hfp.AgIndicator.CALL_SETUP, 2)
        ag.update_ag_indicator(hfp.AgIndicator.CALL_SETUP, 3)
        ag.update_ag_indicator(hfp.AgIndicator.CALL_HELD, 1)  # answered: one active, one held
        ag.update_ag_indicator(hfp.AgIndicator.CALL_SETUP, 0)
        await asyncio.wait_for(consult, 3)
        assert commands == ["CHLD=2", "ATD0491570157;", "CHLD=3"]
    finally:
        line.stop()


async def test_consult_waits_for_setup_to_finish_when_phone_reports_held_while_dialing():
    # Seen on an iPhone: callheld=1 arrives during dialing; merging then is answered with ERROR.
    ag, line, commands = await active_three_way_call()
    try:
        consult = asyncio.create_task(line.consult("132861", answer_timeout=5))
        await asyncio.sleep(0.1)
        ag.update_ag_indicator(hfp.AgIndicator.CALL_SETUP, 2)
        ag.update_ag_indicator(hfp.AgIndicator.CALL_HELD, 1)
        await asyncio.sleep(0.2)
        assert commands == ["CHLD=2", "ATD132861;"]
        ag.update_ag_indicator(hfp.AgIndicator.CALL_SETUP, 3)
        ag.update_ag_indicator(hfp.AgIndicator.CALL_SETUP, 0)
        await asyncio.wait_for(consult, 3)
        assert commands[-1] == "CHLD=3"
    finally:
        line.stop()


async def test_consult_not_answered_resumes_held_call_without_merging():
    ag, line, commands = await active_three_way_call()
    try:
        consult = asyncio.create_task(line.consult("0491570157", answer_timeout=5))
        await asyncio.sleep(0.1)
        ag.update_ag_indicator(hfp.AgIndicator.CALL_HELD, 2)
        ag.update_ag_indicator(hfp.AgIndicator.CALL_SETUP, 2)
        ag.update_ag_indicator(hfp.AgIndicator.CALL_SETUP, 0)  # busy / declined
        with pytest.raises(ConsultFailed):
            await asyncio.wait_for(consult, 3)
        assert commands == ["CHLD=2", "ATD0491570157;", "CHLD=2"]
        assert line.tracker.state is CallState.ACTIVE
    finally:
        line.stop()


async def test_consult_unanswered_in_time_drops_dialed_call_and_resumes_held():
    ag, line, commands = await active_three_way_call()
    try:
        consult = asyncio.create_task(line.consult("0491570157", answer_timeout=0.5))
        await asyncio.sleep(0.1)
        ag.update_ag_indicator(hfp.AgIndicator.CALL_HELD, 2)
        ag.update_ag_indicator(hfp.AgIndicator.CALL_SETUP, 3)  # keeps ringing
        with pytest.raises(ConsultFailed):
            await asyncio.wait_for(consult, 3)
        assert commands == ["CHLD=2", "ATD0491570157;", "CHLD=1"]
    finally:
        line.stop()


async def test_consult_on_phone_without_three_way_calling_fails_fast(phone_and_line):
    ag, line = phone_and_line
    ag.update_ag_indicator(hfp.AgIndicator.CALL, 1)
    await line.wait_for(CallState.ACTIVE, 1)
    with pytest.raises(ConsultFailed, match="three-way"):
        await asyncio.wait_for(line.consult("0491570157", answer_timeout=5), 1)


def conference_call(index: int) -> hfp.CallInfo:
    return hfp.CallInfo(
        index=index,
        direction=hfp.CallInfoDirection.MOBILE_ORIGINATED_CALL,
        status=hfp.CallInfoStatus.ACTIVE,
        mode=hfp.CallInfoMode.VOICE,
        multi_party=hfp.CallInfoMultiParty.IN_CONFERENCE,
        number=f"02555000{index:02}",
    )


async def test_wait_until_one_party_left_returns_when_conference_drops_to_one_call():
    ag, line, _ = await active_three_way_call()
    try:
        ag.calls = [conference_call(1), conference_call(2)]
        waiting = asyncio.create_task(line.wait_until_one_party_left(poll_seconds=0.05))
        await asyncio.sleep(0.2)
        assert not waiting.done()
        ag.calls = [conference_call(1)]
        await asyncio.wait_for(waiting, 2)
    finally:
        line.stop()


async def test_hangup_all_releases_held_calls_that_survive_chup():
    ag, line, commands = await active_three_way_call()
    try:
        ag.calls = [conference_call(1), conference_call(2)]
        ag.on(ag.EVENT_HANG_UP, lambda: ag.calls.pop(0))  # CHUP ends the active call only
        ag.on(ag.EVENT_CALL_HOLD, lambda operation, _index: operation is hfp.CallHoldOperation.RELEASE_ALL_HELD_CALLS and ag.calls.clear())
        await asyncio.wait_for(line.hangup_all(settle_seconds=0.1), 5)
        assert ag.calls == []
        assert commands[-1] == "CHLD=0"
    finally:
        line.stop()


async def test_hangup_all_falls_back_to_releasing_calls_when_phone_refuses_chup():
    # Seen on an iPhone: after the user left a conference, AT+CHUP could not end the remaining call.
    ag, line, commands = await active_three_way_call()
    try:
        ag.calls = [conference_call(2)]
        ag._on_chup = lambda: ag.send_error()
        ag.on(ag.EVENT_CALL_HOLD, lambda operation, _index: operation is hfp.CallHoldOperation.RELEASE_SPECIFIC_CALL and ag.calls.clear())
        await asyncio.wait_for(line.hangup_all(settle_seconds=0.1), 5)
        assert ag.calls == []
        assert commands[-1] == "CHLD=1x"
    finally:
        line.stop()


async def test_dial_fails_fast_when_bluetooth_link_drops(phone_and_line):
    # Seen live: the ACL link timed out right after ATD, and dial then waited forever for an OK.
    ag, line = phone_and_line
    ag._on_d = lambda number: None  # the phone never answers the ATD
    dialing = asyncio.create_task(line.dial("0491570110"))
    await asyncio.sleep(0.1)
    await line.protocol.dlc.multiplexer.l2cap_channel.connection.disconnect()
    with pytest.raises(LinkLost):
        await asyncio.wait_for(dialing, 2)


async def test_hangup_all_gives_the_phone_time_to_drop_the_call_before_escalating():
    # Seen on an iPhone: AT+CHUP is OK'd but the call lingers in AT+CLCC for a moment; escalating
    # meanwhile only earned ERRORs.
    ag, line, commands = await active_three_way_call()
    try:
        ag.calls = [conference_call(2)]
        ag.on(ag.EVENT_HANG_UP, lambda: asyncio.get_running_loop().call_later(0.3, ag.calls.clear))
        await asyncio.wait_for(line.hangup_all(settle_seconds=1), 5)
        assert ag.calls == []
        assert commands == []  # no CHLD escalation needed
    finally:
        line.stop()


async def test_line_still_starts_when_phone_refuses_to_turn_off_echo_cancelling():
    ec_nr = hfp.AgConfiguration(
        supported_ag_features=[*AG_CONFIGURATION.supported_ag_features, hfp.AgFeature.EC_NR],
        supported_ag_indicators=AG_CONFIGURATION.supported_ag_indicators,
        supported_hf_indicators=AG_CONFIGURATION.supported_hf_indicators,
        supported_ag_call_hold_operations=[],
        supported_audio_codecs=AG_CONFIGURATION.supported_audio_codecs,
    )
    ag, line = await connect(ec_nr)  # Bumble's AG answers AT+NREC with ERROR
    line.stop()


async def test_send_dtmf_sends_one_vts_per_digit_and_pauses_on_comma(phone_and_line):
    ag, line = phone_and_line
    sent: list[str] = []

    def on_vts(digit) -> None:
        sent.append(digit.decode() if isinstance(digit, bytes) else digit)
        ag.send_ok()

    ag._on_vts = on_vts
    loop = asyncio.get_running_loop()
    started = loop.time()
    await asyncio.wait_for(line.send_dtmf("1,2#*", pause_s=0.2), 3)
    assert sent == ["1", "2", "#", "*"]
    assert loop.time() - started >= 0.2
