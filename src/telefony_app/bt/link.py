"""Keep the phone's ACL link in active mode. iPhones put an idle link into sniff, and some dongles (10D7:B012) then
lose it to a supervision timeout every 30-60 s (reason 8). Role switch stays allowed; sniff, hold and park do not."""

import asyncio
import logging

from bumble import hci
from bumble.core import PhysicalTransport
from bumble.device import Connection, Device

log = logging.getLogger(__name__)
ROLE_SWITCH_ONLY = 0x0001  # Link_Policy_Settings bit 0; bits 1-3 (hold, sniff, park) cleared
_tasks: set[asyncio.Task] = set()


def keep_links_active(device: Device) -> None:
    """Apply the active-only link policy to every Classic link, whichever side opened it."""
    device.on(Device.EVENT_CONNECTION, _on_connection)


def _on_connection(connection: Connection) -> None:
    if connection.transport != PhysicalTransport.BR_EDR:
        return
    connection.on(Connection.EVENT_MODE_CHANGE, lambda: log.info("phone link mode %d, interval %d", connection.classic_mode, connection.classic_interval))
    task = asyncio.create_task(_disallow_sniff(connection))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def _disallow_sniff(connection: Connection) -> None:
    try:
        await connection.device.host.send_sync_command(
            hci.HCI_Write_Link_Policy_Settings_Command(connection_handle=connection.handle, link_policy_settings=ROLE_SWITCH_ONLY)
        )
    except (hci.HCI_Error, asyncio.TimeoutError) as refused:
        log.warning("controller refused the active-only link policy: %s", refused)
