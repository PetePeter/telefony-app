"""BluetoothPhone surviving a dropped link: the same object is handed a new HFP line and works again."""

import asyncio

from telefony_app.bt.bluetooth_phone import BluetoothPhone
from tests.bt.test_hf_line import connect


class Events:
    def __init__(self) -> None:
        self.links: list[bool] = []

    def link(self, up: bool) -> None:
        self.links.append(up)

    def ringing(self, number: str, name: str | None) -> None: ...
    def call_state(self, state: str, reason: str | None = None) -> None: ...
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
