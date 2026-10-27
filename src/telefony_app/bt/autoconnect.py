"""Dial out to an already-bonded phone so it doesn't have to reconnect to us (phones rarely auto-reconnect)."""

import asyncio
import logging

from bumble import hfp, rfcomm
from bumble import core
from bumble.core import PhysicalTransport
from bumble.device import Connection, Device
from bumble.hci import Address
from bumble.keys import KeyStore

log = logging.getLogger("telefony")
_INBOUND_LINK_SECONDS = 5.0
# A phone ignores pages until its old link times out (~20 s after we die without disconnecting): keep trying.
_RECONNECT_ATTEMPTS = 6
_RECONNECT_PAUSE_SECONDS = 5.0


async def bonded_phones(keystore: KeyStore) -> list[Address]:
    """Peers we hold a Classic link key for; LE-only bonds can't carry HFP."""
    return [Address(name) for name, keys in await keystore.get_all() if keys.link_key is not None]


async def _acl_to(device: Device, address: Address) -> Connection:
    """ACL to the phone; it may have linked to us first (we're connectable during startup), so reuse that."""
    if existing := device.find_connection_by_bd_addr(address, PhysicalTransport.BR_EDR):
        return existing
    try:
        return await device.connect(address, transport=PhysicalTransport.BR_EDR)
    except core.ConnectionError:  # bumble's, not the builtin
        # The phone won the race while we were paging; its link completes shortly after.
        async with asyncio.timeout(_INBOUND_LINK_SECONDS):
            while not (existing := device.find_connection_by_bd_addr(address, PhysicalTransport.BR_EDR)):
                await asyncio.sleep(0.1)
        return existing


async def connect_bonded_phone(device: Device) -> rfcomm.DLC | None:
    """Open the HFP RFCOMM channel to the first bonded phone that answers, or None after ~30 s.
    Like a car kit: the hands-free side reconnects; phones don't."""
    if device.keystore is None:
        return None
    phones = await bonded_phones(device.keystore)
    for attempt in range(_RECONNECT_ATTEMPTS if phones else 0):
        if attempt:
            await asyncio.sleep(_RECONNECT_PAUSE_SECONDS)
            if device.connections:  # the phone linked to us meanwhile; its HFP channel arrives via the RFCOMM server
                return None
        if dlc := await _try_each(device, phones):
            return dlc
    return None


async def _try_each(device: Device, phones: list[Address]) -> rfcomm.DLC | None:
    for address in phones:
        try:
            connection = await _acl_to(device, address)
            await connection.authenticate()
            await connection.encrypt()
            ag = await hfp.find_ag_sdp_record(connection)
            if ag is None:
                log.warning("%s has no hands-free gateway service", address)
                await connection.disconnect()
                continue
            multiplexer = await rfcomm.Client(connection).start()
            return await multiplexer.open_dlc(ag[0])
        except asyncio.CancelledError as e:  # Bumble cancels authenticate() when a phone that forgot us drops the link
            if (task := asyncio.current_task()) is not None and task.cancelling():
                raise
            log.info("bonded phone %s dropped the link (did it forget us? pair again): %s", address, e)
        except Exception as e:  # phone off / out of range: try the next, then wait for the phone
            log.info("could not reach bonded phone %s: %s", address, e)
    return None
