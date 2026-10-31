import asyncio

from bumble import hci
from bumble.controller import Controller
from bumble.core import PhysicalTransport
from bumble.device import Device
from bumble.link import LocalLink

from telefony_app.bt.link import keep_links_alive
from tests.bt.fake_phone import make_device

US, PHONE = "F0:F1:F2:F3:F4:F5", "0C:51:01:73:71:27"
TICK = 0.01


class RssiController(Controller):
    """Bumble's controller has no Read_RSSI; count the polls and answer, as the real dongle does."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.polls = 0

    def on_hci_read_rssi_command(self, command):
        self.polls += 1
        return hci.HCI_Read_RSSI_ReturnParameters(status=hci.HCI_ErrorCode.SUCCESS, handle=command.handle, rssi=-52)


async def _linked(link: LocalLink, controller: Controller | None = None):
    if controller is None:
        us = await make_device(link, US)
    else:
        us = Device.with_hci(US, hci.Address(US), controller, controller)
        us.classic_enabled = True
        await us.power_on()
    phone = await make_device(link, PHONE)
    keep_links_alive(us, interval_s=TICK)
    return us, await us.connect(phone.public_address, transport=PhysicalTransport.BR_EDR)


async def test_idle_link_is_polled_until_it_disconnects():
    link = LocalLink()
    controller = RssiController(US, link=link, public_address=US)
    _, connection = await _linked(link, controller)

    await asyncio.sleep(TICK * 10)
    assert controller.polls >= 3

    await connection.disconnect()
    stopped_at = controller.polls
    await asyncio.sleep(TICK * 10)
    assert controller.polls == stopped_at


async def test_controller_refusing_the_poll_leaves_the_link_up():
    us, connection = await _linked(LocalLink())  # bumble's controller answers Read_RSSI with UNKNOWN_HCI_COMMAND

    await asyncio.sleep(TICK * 10)

    assert connection in us.connections.values()
