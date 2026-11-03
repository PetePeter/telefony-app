"""A fake phone on Bumble's in-memory radio link, serving one OBEX service from a request handler; and our side's
dongle, healthy or wedged."""

import struct
from collections.abc import Callable

from bumble import core, hci, rfcomm, sdp
from bumble.controller import Controller
from bumble.device import Device
from bumble.hci import Address
from bumble.keys import MemoryKeyStore, PairingKeys
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


class WedgedSink:
    """The host-to-controller edge of a USB dongle that has stopped taking connection commands (seen live as
    'OUT transfer not completed'): HCI_Create_Connection never reaches the controller, everything else does."""

    CREATE_CONNECTION = bytes([hci.HCI_COMMAND_PACKET]) + hci.HCI_CREATE_CONNECTION_COMMAND.to_bytes(2, "little")

    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.dropped = 0

    def on_packet(self, packet: bytes) -> None:
        if packet.startswith(self.CREATE_CONNECTION):
            self.dropped += 1
            return
        self.controller.on_packet(packet)


async def make_bonded_device(link: LocalLink, phone: str, wedged: bool = False) -> Device:
    """Our side, holding a Classic link key for phone; with wedged, its dongle swallows every page."""
    address = "F0:F1:F2:F3:F4:F5"
    controller = Controller(address, link=link, public_address=address)
    device = Device.with_hci(address, Address(address), controller, WedgedSink(controller) if wedged else controller)
    device.classic_enabled = True
    device.keystore = MemoryKeyStore()
    await device.keystore.update(phone, PairingKeys(link_key=PairingKeys.Key(value=bytes(16)), link_key_type=4))
    await device.power_on()
    return device
