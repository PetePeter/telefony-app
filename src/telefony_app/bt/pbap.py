"""PBAP client (PCE): pull the phone's main phonebook as vCard 3.0 over OBEX/RFCOMM."""

import uuid

from bumble import core, rfcomm
from bumble.device import Connection

from telefony_app.bt.call_log import CallLogEntry, parse_call_log
from telefony_app.bt.contacts import Contact, parse_vcards
from telefony_app.bt.obex import obex_session

PBAP_TARGET = uuid.UUID("796135f0-f0c5-11d8-0966-0800200c9a66").bytes
PHONEBOOK = "telecom/pb.vcf"
CALL_HISTORY = "telecom/cch.vcf"  # combined missed, received and dialed, newest first
PHONEBOOK_TYPE = "x-bt/phonebook"
# Application parameter TLV: Format (0x07) = 1, vCard 3.0. The default is 2.1.
VCARD_30 = bytes([0x07, 0x01, 0x01])


async def pull_contacts(connection: Connection, multiplexer: rfcomm.Multiplexer | None = None) -> list[Contact]:
    async with obex_session(connection, core.BT_PHONEBOOK_ACCESS_PSE_SERVICE, PBAP_TARGET, multiplexer) as client:
        vcards = await client.get(PHONEBOOK, PHONEBOOK_TYPE, VCARD_30)
    return parse_vcards(vcards.decode("utf-8", errors="replace"))


async def pull_call_log(
    connection: Connection, multiplexer: rfcomm.Multiplexer | None = None, limit: int = 20
) -> list[CallLogEntry]:
    max_list_count = bytes([0x04, 0x02]) + limit.to_bytes(2, "big")
    async with obex_session(connection, core.BT_PHONEBOOK_ACCESS_PSE_SERVICE, PBAP_TARGET, multiplexer) as client:
        vcards = await client.get(CALL_HISTORY, PHONEBOOK_TYPE, VCARD_30 + max_list_count)
    return parse_call_log(vcards.decode("utf-8", errors="replace"))[:limit]
