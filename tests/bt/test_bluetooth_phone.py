"""BluetoothPhone surviving a dropped link: the same object is handed a new HFP line and works again."""

import asyncio

import pytest
from bumble import hfp

from telefony_app.bt.bluetooth_phone import BluetoothPhone
from telefony_app.bt.call_state import CallState
from telefony_app.phone import PhoneError
from tests.bt.test_hf_line import connect


class Events:
    def __init__(self) -> None:
        self.links: list[bool] = []
        self.call_states: list[tuple[str, str | None]] = []

    def link(self, up: bool) -> None:
        self.links.append(up)

    def ringing(self, number: str, name: str | None) -> None: ...
    def call_state(self, state: str, reason: str | None = None) -> None:
        self.call_states.append((state, reason))
    def utterance(self, text: str, start_ms: int, end_ms: int) -> None: ...
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
