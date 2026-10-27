"""A fake phone on Bumble's in-memory radio link, serving one OBEX service from a request handler."""

import struct
from collections.abc import Callable

from bumble import core, rfcomm, sdp
from bumble.controller import Controller
from bumble.device import Device
from bumble.hci import Address
from bumble.link import LocalLink

from telefony_app.bt import obex

Handler = Callable[[obex.Packet], obex.Packet]
PHONE_MAX_PACKET = 0x400


async def make_device(link: LocalLink, address: str) -> Device:
    controller = Controller(address, link=link, public_address=address)
    device = Device.with_hci(address, Address(address), controller, controller)
    device.classic_enabled = True
    await device.power_on()
    return device


def serve_obex(device: Device, service: core.UUID, handle: Handler) -> list[obex.Packet]:
    """Answer each OBEX request with handle(request); CONNECT is answered here. Returns the request log."""
    requests: list[obex.Packet] = []

    def on_dlc(dlc: rfcomm.DLC) -> None:
        buffer = b""

        def on_data(data: bytes) -> None:
            nonlocal buffer
            buffer += data
            (length,) = struct.unpack_from(">H", buffer, 1)
            if len(buffer) < length:
                return
            request = obex.decode(buffer[:length], obex.fields_length(buffer[0]))
            buffer = buffer[length:]
            requests.append(request)
            if request.code == obex.CONNECT:
                fields = struct.pack(">BBH", 0x10, 0, PHONE_MAX_PACKET)
                reply = obex.Packet(obex.SUCCESS, [(obex.CONNECTION_ID, 1)], fields)
            elif request.code == obex.DISCONNECT:
                reply = obex.Packet(obex.SUCCESS)
            else:
                reply = handle(request)
            dlc.write(obex.encode(reply))

        dlc.sink = on_data

    channel = rfcomm.Server(device).listen(on_dlc)
    element = sdp.DataElement
    device.sdp_service_records = {
        0x00010001: [
            sdp.ServiceAttribute(sdp.SDP_SERVICE_RECORD_HANDLE_ATTRIBUTE_ID, element.unsigned_integer_32(0x00010001)),
            sdp.ServiceAttribute(sdp.SDP_SERVICE_CLASS_ID_LIST_ATTRIBUTE_ID, element.sequence([element.uuid(service)])),
            sdp.ServiceAttribute(
                sdp.SDP_PROTOCOL_DESCRIPTOR_LIST_ATTRIBUTE_ID,
                element.sequence(
                    [
                        element.sequence([element.uuid(core.BT_L2CAP_PROTOCOL_ID)]),
                        element.sequence([element.uuid(core.BT_RFCOMM_PROTOCOL_ID), element.unsigned_integer_8(channel)]),
                        element.sequence([element.uuid(core.BT_OBEX_PROTOCOL_ID)]),
                    ]
                ),
            ),
        ]
    }
    return requests
