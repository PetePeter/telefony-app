"""Keep the dongle busy while the phone is linked. With no HCI traffic for ~20 s, the 10D7:B012 dongle on Windows
stops serving the link and the phone drops it on supervision timeout (reason 8) every 30-60 s. A local Read_RSSI
every few seconds (no radio traffic) keeps it up."""

import asyncio
import logging

from bumble import hci
from bumble.core import PhysicalTransport
from bumble.device import Connection, Device

log = logging.getLogger(__name__)
KEEPALIVE_S = 5.0
_tasks: set[asyncio.Task] = set()


def keep_links_alive(device: Device, interval_s: float = KEEPALIVE_S) -> None:
    """Poll every Classic link, whichever side opened it, until it disconnects."""

    def on_connection(connection: Connection) -> None:
        if connection.transport != PhysicalTransport.BR_EDR:
            return
        task = asyncio.create_task(_keepalive(connection, interval_s))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
        connection.on(Connection.EVENT_DISCONNECTION, lambda _reason: task.cancel())

    device.on(Device.EVENT_CONNECTION, on_connection)


async def _keepalive(connection: Connection, interval_s: float) -> None:
    log.info("link keepalive every %.0f s", interval_s)
    while True:
        await asyncio.sleep(interval_s)
        try:
            response = await connection.device.host.send_sync_command(hci.HCI_Read_RSSI_Command(handle=connection.handle))
        except (hci.HCI_Error, asyncio.TimeoutError) as failed:
            log.warning("link keepalive stopped: %s", failed)
            return
        log.debug("phone link RSSI %d dBm", response.rssi)
