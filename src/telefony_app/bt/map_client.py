"""MAP client (MCE): list, read and send SMS through the phone's Message Access Server over OBEX/RFCOMM."""

import uuid
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from bumble import core, rfcomm, sdp
from bumble.device import Connection

from telefony_app.bt.bmessage import ListedMessage, Message, build_sms, parse_listing, parse_message
from telefony_app.bt.obex import NAME, ObexClient, obex_session

MAS_TARGET = uuid.UUID("bb582b40-420c-11db-b0de-0800200c9a66").bytes
MAP_VERSION = 0x0104
_MAP_SUPPORTED_FEATURES_ATTRIBUTE_ID = 0x0317
_NOTIFICATION_REGISTRATION = 0x00000001
_SERVICE_NAME_ATTRIBUTE_ID = 0x0100
# Application parameter TLVs
_CHARSET_UTF8 = bytes([0x14, 0x01, 0x01])
_NO_ATTACHMENTS = bytes([0x0A, 0x01, 0x00])


def _max_list_count(count: int) -> bytes:
    return bytes([0x01, 0x02]) + count.to_bytes(2, "big")


class MapSession:
    def __init__(self, client: ObexClient) -> None:
        self._client = client

    async def list_messages(self, folder: str = "inbox", count: int = 20) -> list[ListedMessage]:
        data = await self._client.get(folder, "x-bt/MAP-msg-listing", _max_list_count(count))
        return parse_listing(data)

    async def read_message(self, handle: str) -> Message:
        return parse_message(await self._client.get(handle, "x-bt/message", _NO_ATTACHMENTS + _CHARSET_UTF8))

    async def send_sms(self, recipient: str, text: str) -> str:
        """Push to the outbox; the phone sends it. Returns the new message handle. Callers enforce the allowlist."""
        response = await self._client.put("outbox", "x-bt/message", build_sms(recipient, text), _CHARSET_UTF8)
        handle = response.header(NAME)
        return handle if isinstance(handle, str) else ""


@asynccontextmanager
async def map_session(connection: Connection, multiplexer: rfcomm.Multiplexer | None = None) -> AsyncIterator[MapSession]:
    async with obex_session(connection, core.BT_MESSAGE_ACCESS_SERVER_SERVICE, MAS_TARGET, multiplexer) as client:
        await client.set_path("telecom")
        await client.set_path("msg")
        yield MapSession(client)


def make_mns_sdp_records(handle: int, rfcomm_channel: int) -> list[sdp.ServiceAttribute]:
    """Message Notification Server record. iOS hides "Show Notifications" (and refuses MAP) without it."""
    element = sdp.DataElement
    return [
        sdp.ServiceAttribute(sdp.SDP_SERVICE_RECORD_HANDLE_ATTRIBUTE_ID, element.unsigned_integer_32(handle)),
        sdp.ServiceAttribute(
            sdp.SDP_SERVICE_CLASS_ID_LIST_ATTRIBUTE_ID,
            element.sequence([element.uuid(core.BT_MESSAGE_NOTIFICATION_SERVER_SERVICE)]),
        ),
        sdp.ServiceAttribute(
            sdp.SDP_PROTOCOL_DESCRIPTOR_LIST_ATTRIBUTE_ID,
            element.sequence([
                element.sequence([element.uuid(core.BT_L2CAP_PROTOCOL_ID)]),
                element.sequence([element.uuid(core.BT_RFCOMM_PROTOCOL_ID), element.unsigned_integer_8(rfcomm_channel)]),
                element.sequence([element.uuid(core.BT_OBEX_PROTOCOL_ID)]),
            ]),
        ),
        sdp.ServiceAttribute(_SERVICE_NAME_ATTRIBUTE_ID, element.text_string(b"MAP MNS")),
        sdp.ServiceAttribute(
            sdp.SDP_BLUETOOTH_PROFILE_DESCRIPTOR_LIST_ATTRIBUTE_ID,
            element.sequence([
                element.sequence([element.uuid(core.BT_MESSAGE_ACCESS_PROFILE_SERVICE), element.unsigned_integer_16(MAP_VERSION)])
            ]),
        ),
        sdp.ServiceAttribute(
            _MAP_SUPPORTED_FEATURES_ATTRIBUTE_ID, element.unsigned_integer_32(_NOTIFICATION_REGISTRATION)
        ),
    ]
