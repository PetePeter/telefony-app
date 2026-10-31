import asyncio

from bumble import hci
from bumble.controller import Controller
from bumble.core import PhysicalTransport
from bumble.device import Device
from bumble.link import LocalLink

from telefony_app.bt.link import ROLE_SWITCH_ONLY, keep_links_active
from tests.bt.fake_phone import make_device

US, PHONE = "F0:F1:F2:F3:F4:F5", "0C:51:01:73:71:27"


class PolicyRecordingController(Controller):
    """Bumble's controller has no Write_Link_Policy_Settings; record it and succeed, as a real dongle does."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.policies: list[tuple[int, int]] = []

    def on_hci_write_link_policy_settings_command(self, command):
        self.policies.append((command.connection_handle, command.link_policy_settings))
        return hci.HCI_StatusAndConnectionHandleReturnParameters(status=hci.HCI_ErrorCode.SUCCESS, connection_handle=command.connection_handle)


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


async def test_sniff_is_disallowed_on_a_link_we_open():
    link = LocalLink()
    controller = PolicyRecordingController(US, link=link, public_address=US)
    us = Device.with_hci(US, hci.Address(US), controller, controller)
    us.classic_enabled = True
    await us.power_on()
    phone = await make_device(link, PHONE)
    keep_links_active(us)

    connection = await us.connect(phone.public_address, transport=PhysicalTransport.BR_EDR)
    await _settle()

    assert controller.policies == [(connection.handle, ROLE_SWITCH_ONLY)]


async def test_link_survives_a_controller_that_refuses_the_policy():
    link = LocalLink()
    us, phone = await make_device(link, US), await make_device(link, PHONE)
    keep_links_active(us)

    connection = await us.connect(phone.public_address, transport=PhysicalTransport.BR_EDR)
    await _settle()

    assert connection in us.connections.values()
