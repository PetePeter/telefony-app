"""Minimal OBEX (IrOBEX 1.5) client over an RFCOMM channel: CONNECT, SETPATH, GET, PUT, DISCONNECT. Shared by PBAP and MAP."""

import asyncio
import struct
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from bumble import core, rfcomm, sdp
from bumble.device import Connection

# Opcodes (final bit set) and response codes
CONNECT, DISCONNECT, PUT, GET, SETPATH = 0x80, 0x81, 0x82, 0x83, 0x85
PUT_MORE = 0x02  # PUT without the final bit: more body follows
CONTINUE, SUCCESS = 0x90, 0xA0

# Header IDs; the top two bits give the encoding
NAME, TYPE, TARGET, BODY, END_OF_BODY, WHO, APP_PARAMETERS, CONNECTION_ID = 0x01, 0x42, 0x46, 0x48, 0x49, 0x4A, 0x4C, 0xCB

OBEX_VERSION = 0x10
MAX_PACKET = 0xFFFF
SETPATH_DONT_CREATE, SETPATH_PARENT = 0x02, 0x01
_HEADER_ROOM = 64  # bytes left for non-body headers when splitting a PUT body


class ObexError(Exception):
    pass


@dataclass
class Packet:
    code: int  # opcode for requests, response code for responses
    headers: list[tuple[int, bytes | str | int]] = field(default_factory=list)
    fields: bytes = b""  # fixed fields before the headers: CONNECT and its response (4 bytes), SETPATH (2 bytes)

    def header(self, header_id: int) -> bytes | str | int | None:
        return next((v for h, v in self.headers if h == header_id), None)

    def body(self) -> bytes:
        return b"".join(v for h, v in self.headers if h in (BODY, END_OF_BODY) and isinstance(v, bytes))


def _encode_header(header_id: int, value: bytes | str | int) -> bytes:
    kind = header_id & 0xC0
    if kind == 0x00:  # null-terminated UTF-16BE text
        assert isinstance(value, str)
        data = (value + "\0").encode("utf-16-be") if value else b""
        return struct.pack(">BH", header_id, 3 + len(data)) + data
    if kind == 0x40:
        assert isinstance(value, bytes)
        return struct.pack(">BH", header_id, 3 + len(value)) + value
    assert isinstance(value, int)
    return struct.pack(">BB", header_id, value) if kind == 0x80 else struct.pack(">BI", header_id, value)


def encode(packet: Packet) -> bytes:
    payload = packet.fields + b"".join(_encode_header(h, v) for h, v in packet.headers)
    return struct.pack(">BH", packet.code, 3 + len(payload)) + payload


def fields_length(request_code: int) -> int:
    """Length of the fixed fields in a request with this opcode, and in the response to it."""
    return {CONNECT: 4, SETPATH: 2}.get(request_code, 0)


def decode(data: bytes, fields: int = 0) -> Packet:
    code, length = struct.unpack_from(">BH", data)
    if length != len(data):
        raise ObexError(f"packet length {length} but got {len(data)} bytes")
    offset = 3 + fields
    packet = Packet(code, fields=data[3:offset])
    while offset < length:
        header_id = data[offset]
        kind = header_id & 0xC0
        if kind in (0x00, 0x40):
            (size,) = struct.unpack_from(">H", data, offset + 1)
            raw = data[offset + 3 : offset + size]
            value: bytes | str | int = raw.decode("utf-16-be").rstrip("\0") if kind == 0x00 else raw
        elif kind == 0x80:
            size, value = 2, data[offset + 1]
        else:
            size, value = 5, struct.unpack_from(">I", data, offset + 1)[0]
        packet.headers.append((header_id, value))
        offset += size
    return packet


async def find_rfcomm_channel(connection: Connection, service: core.UUID) -> int | None:
    """RFCOMM channel of one service. Asks for that service by UUID: iOS drops records from a broad RFCOMM search."""
    async with sdp.Client(connection) as client:
        records = await client.search_attributes([service], [sdp.SDP_PROTOCOL_DESCRIPTOR_LIST_ATTRIBUTE_ID])
    for record in records:
        for attribute in record:
            for protocol in attribute.value.value:  # [[L2CAP], [RFCOMM, channel], [OBEX]]
                if protocol.value[0].value == core.BT_RFCOMM_PROTOCOL_ID:
                    return protocol.value[1].value
    return None


class ObexClient:
    """One OBEX session on an open RFCOMM DLC. Requests are strictly one at a time."""

    def __init__(self, dlc: rfcomm.DLC) -> None:
        self._dlc = dlc
        self._buffer = b""
        self._packets: asyncio.Queue[bytes] = asyncio.Queue()
        self._connection_id: int | None = None
        self._peer_max_packet = 255  # the OBEX minimum until CONNECT tells us more
        dlc.sink = self._on_data

    def _on_data(self, data: bytes) -> None:
        self._buffer += data
        while len(self._buffer) >= 3:
            (length,) = struct.unpack_from(">H", self._buffer, 1)
            if len(self._buffer) < length:
                return
            self._packets.put_nowait(self._buffer[:length])
            self._buffer = self._buffer[length:]

    async def _request(self, packet: Packet) -> Packet:
        if self._connection_id is not None and packet.code != CONNECT:
            packet.headers.insert(0, (CONNECTION_ID, self._connection_id))
        self._dlc.write(encode(packet))
        # SETPATH responses carry no fixed fields; only CONNECT's does
        return decode(await self._packets.get(), fields_length(packet.code) if packet.code == CONNECT else 0)

    async def connect(self, target: bytes) -> None:
        fields = struct.pack(">BBH", OBEX_VERSION, 0, MAX_PACKET)
        response = await self._request(Packet(CONNECT, [(TARGET, target)], fields))
        if response.code != SUCCESS:
            raise ObexError(f"CONNECT refused: 0x{response.code:02X}")
        (self._peer_max_packet,) = struct.unpack_from(">H", response.fields, 2)
        connection_id = response.header(CONNECTION_ID)
        self._connection_id = connection_id if isinstance(connection_id, int) else None

    async def get(self, name: str, mime_type: str, app_parameters: bytes = b"") -> bytes:
        headers: list[tuple[int, bytes | str | int]] = [(NAME, name), (TYPE, mime_type.encode() + b"\0")]
        if app_parameters:
            headers.append((APP_PARAMETERS, app_parameters))
        body = b""
        response = await self._request(Packet(GET, headers))
        while response.code == CONTINUE:
            body += response.body()
            response = await self._request(Packet(GET))
        if response.code != SUCCESS:
            raise ObexError(f"GET {name} failed: 0x{response.code:02X}")
        return body + response.body()

    async def set_path(self, folder: str) -> None:
        """Enter a child folder, or go back to the root with an empty name."""
        response = await self._request(Packet(SETPATH, [(NAME, folder)], bytes([SETPATH_DONT_CREATE, 0])))
        if response.code != SUCCESS:
            raise ObexError(f"SETPATH {folder!r} failed: 0x{response.code:02X}")

    async def put(self, name: str, mime_type: str, body: bytes, app_parameters: bytes = b"") -> Packet:
        """Send a body, split to the peer's packet size. Returns the final response (MAP puts the new handle in NAME)."""
        headers: list[tuple[int, bytes | str | int]] = [(NAME, name), (TYPE, mime_type.encode() + b"\0")]
        if app_parameters:
            headers.append((APP_PARAMETERS, app_parameters))
        room = self._peer_max_packet - _HEADER_ROOM - len(encode(Packet(PUT, list(headers))))
        while len(body) > room:
            chunk, body = body[:room], body[room:]
            response = await self._request(Packet(PUT_MORE, headers + [(BODY, chunk)]))
            if response.code != CONTINUE:
                raise ObexError(f"PUT {name} failed: 0x{response.code:02X}")
            headers, room = [], self._peer_max_packet - _HEADER_ROOM
        response = await self._request(Packet(PUT, headers + [(END_OF_BODY, body)]))
        if response.code != SUCCESS:
            raise ObexError(f"PUT {name} failed: 0x{response.code:02X}")
        return response

    async def disconnect(self) -> None:
        await self._request(Packet(DISCONNECT))


async def _open_dlc(multiplexer: rfcomm.Multiplexer, channel: int) -> rfcomm.DLC:
    """Bumble only opens DLCs as the RFCOMM session initiator. When the phone started the session, its servers
    sit on odd DLCIs (direction bit 1), so build the parameter negotiation ourselves."""
    if multiplexer.role == rfcomm.Multiplexer.Role.INITIATOR:
        return await multiplexer.open_dlc(channel)
    if multiplexer.state != rfcomm.Multiplexer.State.CONNECTED:
        raise RuntimeError("RFCOMM multiplexer busy or closed")
    multiplexer.open_pn = rfcomm.RFCOMM_MCC_PN(
        dlci=(channel << 1) | 1,
        cl=0xF0,
        priority=7,
        ack_timer=0,
        max_frame_size=rfcomm.RFCOMM_DEFAULT_MAX_FRAME_SIZE,
        max_retransmissions=0,
        initial_credits=rfcomm.RFCOMM_DEFAULT_INITIAL_CREDITS,
    )
    mcc = rfcomm.RFCOMM_Frame.make_mcc(mcc_type=rfcomm.MccType.PN, c_r=1, data=bytes(multiplexer.open_pn))
    multiplexer.open_result = asyncio.get_running_loop().create_future()
    multiplexer.change_state(rfcomm.Multiplexer.State.OPENING)
    multiplexer.send_frame(rfcomm.RFCOMM_Frame.uih(c_r=0, dlci=0, information=mcc))
    return await multiplexer.open_result


@asynccontextmanager
async def obex_session(
    connection: Connection, service: core.UUID, target: bytes, multiplexer: rfcomm.Multiplexer | None = None
) -> AsyncIterator[ObexClient]:
    """Connected OBEX session to one phone service. Pass the HFP link's multiplexer: iOS allows one RFCOMM
    session per phone and refuses a second (L2CAP "no resources")."""
    channel = await find_rfcomm_channel(connection, service)
    if channel is None:
        raise LookupError(f"phone does not offer {service}")
    if multiplexer is None:
        multiplexer = await rfcomm.Client(connection).start()
    dlc = await _open_dlc(multiplexer, channel)
    try:
        client = ObexClient(dlc)
        await client.connect(target)
        yield client
        await client.disconnect()
    finally:
        await dlc.disconnect()
