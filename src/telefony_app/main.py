"""telefony-app: bring up the Bluetooth hands-free unit, then serve PhAP or MCP (one or the other, never both).

  telefony-app telefony.toml          # PhAP server for Helm or any agent host
  telefony-app telefony.toml --mcp    # MCP on stdio instead, for a plain LLM CLI that launches it
  telefony-app describe-setup --json  # Describe operator-configurable provider settings
"""

import argparse
import asyncio
import contextlib
import json
import logging
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

from bumble import hci, hfp, rfcomm
from bumble import core
from bumble.device import Connection, Device, DeviceConfiguration
from bumble.transport import open_transport
from bumble.transport.common import TransportInitError

from telefony_app import __version__
from telefony_app.bt.autoconnect import ControllerWedged, bonded_phones, connect_bonded_phone
from telefony_app.bt.bluetooth_phone import BluetoothPhone
from telefony_app.bt.hf_line import HF_CONFIGURATION, HfLine
from telefony_app.bt.link import keep_links_alive
from telefony_app.bt.map_client import make_mns_sdp_records
from telefony_app.bt.pairing import PairingController
from telefony_app.bt.sco_stream import UsbScoStreamer
from telefony_app.config import SETUP_VERSION, Config, describe_setup_fields, load_config
from telefony_app.mcp_projection import McpProjection
from telefony_app.phap.server import PhapServer
from telefony_app.service import PhoneService
from telefony_app.voice import make_stt, make_tts
from telefony_app.web.app import WebApp

HANDS_FREE_CLASS_OF_DEVICE = 0x200408
NO_DONGLE_EXIT = 3  # with the stderr line "telefony-app: dongle not found", which Helm matches
SDP_HANDLE, MNS_SDP_HANDLE = 0x00010001, 0x00010002
RECONNECT_PAUSE_S = 5.0
# A bonded phone's link with no HFP line this long after it came up is half-open (seen live after a drop).
LINE_DEADLINE_S = 20.0
log = logging.getLogger("telefony_app")
LineQueue = asyncio.Queue[tuple[HfLine, rfcomm.DLC]]


async def listen_for_phone(
    device: Device, pairing_controller: PairingController | None = None, recording_directory: Path | None = None
) -> tuple[LineQueue, Callable[[rfcomm.DLC], None]]:
    """Advertise as a hands-free unit and page the bonded phone. Every HFP line that comes up, now or after a drop,
    lands in the queue; the callback brings up a channel we opened ourselves."""
    connected: LineQueue = asyncio.Queue()
    bring_ups: set[asyncio.Task] = set()

    async def bring_up(dlc: rfcomm.DLC) -> None:
        line = HfLine(hfp.HfProtocol(dlc, HF_CONFIGURATION), recording_directory=recording_directory)
        line.attach_sco(dlc.multiplexer.l2cap_channel.connection)
        try:
            await line.start()
        except Exception:
            log.exception("service level connection failed; reconnect from the phone")
            return
        connected.put_nowait((line, dlc))

    def on_dlc(dlc: rfcomm.DLC) -> None:
        task = asyncio.create_task(bring_up(dlc))
        bring_ups.add(task)
        task.add_done_callback(bring_ups.discard)

    server = rfcomm.Server(device)
    channel = server.listen(on_dlc)
    mns_channel = server.listen(lambda dlc: log.info("phone opened MNS channel (not handled yet)"))
    device.sdp_service_records = {
        SDP_HANDLE: hfp.make_hf_sdp_records(SDP_HANDLE, channel, HF_CONFIGURATION),
        MNS_SDP_HANDLE: make_mns_sdp_records(MNS_SDP_HANDLE, mns_channel),
    }
    keep_links_alive(device)
    await device.power_on()
    await device.set_connectable(True)
    if pairing_controller is None:  # preserve the legacy first-pair path when the admin API is disabled
        await device.set_discoverable(True)
    else:
        pairing_controller.mark_ready()
    if device.keystore is not None and await bonded_phones(device.keystore):
        log.info("paging the bonded phone")
    elif pairing_controller is not None:
        log.info("waiting for a pairing window from the setup API")
    else:
        log.info("discoverable as 'telefony'; pair it from the phone's Bluetooth settings")
    return connected, on_dlc


async def keep_linked(device: Device, phone: BluetoothPhone, connected: LineQueue, on_dlc: Callable[[rfcomm.DLC], None]) -> None:
    """Whenever the link drops, wait for the next line and relink."""
    while True:
        await phone.link_down.wait()
        log.info("phone link lost; reconnecting")
        phone.relink(*await next_line(device, connected, on_dlc))
        log.info("phone reconnected")


async def next_line(device: Device, connected: LineQueue, on_dlc: Callable[[rfcomm.DLC], None]) -> tuple[HfLine, rfcomm.DLC]:
    """Page the bonded phone until a live HFP line is up (from us or the phone).
    Raises ControllerWedged (in an ExceptionGroup) once the dongle stops answering."""
    async with asyncio.TaskGroup() as paging:  # a wedged pager fails the group instead of dying unseen
        pager = paging.create_task(_page_until_back(device, on_dlc))
        line, dlc = await connected.get()
        while dlc.multiplexer.l2cap_channel.connection not in device.connections.values():  # stale, from an old link
            line, dlc = await connected.get()
        pager.cancel()
    return line, dlc


async def keep_controller(
    open_device: Callable[[], contextlib.AbstractAsyncContextManager[Device]], link_phone: Callable[[Device], Awaitable[None]]
) -> None:
    """Run link_phone on a freshly opened dongle; when the dongle wedges, close it, reopen it and start over."""
    while True:
        try:
            async with open_device() as device:
                await link_phone(device)
            return
        except* ControllerWedged as wedged:
            log.warning("Bluetooth dongle stopped answering (%s); reopening it", wedged.exceptions[0])


async def _page_until_back(device: Device, on_dlc: Callable[[rfcomm.DLC], None]) -> None:
    """Runs until cancelled: page while unlinked. A bonded phone's link that brings no HFP line within
    LINE_DEADLINE_S is half-open (the keepalive would hold it forever), so drop it and page again."""
    up_since: dict[Connection, float] = {}
    while True:
        if not device.connections and (dlc := await connect_bonded_phone(device)):
            on_dlc(dlc)
        now = asyncio.get_running_loop().time()
        bonded = await bonded_phones(device.keystore) if device.keystore is not None else []
        for connection in list(device.connections.values()):
            if connection.peer_address in bonded and now - up_since.setdefault(connection, now) >= LINE_DEADLINE_S:
                await _drop_half_open(connection)
        await asyncio.sleep(RECONNECT_PAUSE_S)


async def _drop_half_open(connection: Connection) -> None:
    log.warning("link to %s brought no hands-free line in %.0f s; dropping it", connection.peer_address, LINE_DEADLINE_S)
    try:
        await connection.disconnect()
    except core.CommandTimeoutError:
        raise ControllerWedged("no answer to a disconnect") from None
    except Exception as e:  # it went away by itself meanwhile
        log.info("dropping %s: %s", connection.peer_address, e)


def make_device_configuration(config: Config) -> DeviceConfiguration:
    return DeviceConfiguration(
        name=config.device_name,
        address=hci.Address("F0:F1:F2:F3:F4:F5"),
        classic_enabled=True,
        class_of_device=HANDS_FREE_CLASS_OF_DEVICE,
        keystore="JsonKeyStore",  # keep the phone paired across runs
    )


async def run(config: Config, config_path: Path, mcp: bool) -> None:
    tts, stt = make_tts(config.tts, config.piper_model, config.sapi_voice), make_stt(config.stt_model, config.whisper_cpp_server, config.whisper_cpp_model)
    log.info("text-to-speech: %s; speech-to-text: %s", type(tts).__name__ if tts else "none", type(stt).__name__ if stt else "none")
    device: Device | None = None
    phone = BluetoothPhone(None, None, tts, stt)  # PhAP serves it from the start; it reports link down until linked
    async with contextlib.AsyncExitStack() as stack:
        web = None
        if config.web_token:  # up before the phone, so pairing can be watched from the page
            web = WebApp(config_path, config, keystore=lambda: device.keystore if device else None)
            port = await stack.enter_async_context(web.serve())
            log.info("web UI on http://%s:%d/", config.web_host, port)

        @contextlib.asynccontextmanager
        async def open_device() -> AsyncIterator[Device]:
            nonlocal device
            async with await open_transport(config.transport) as transport:
                if getattr(transport.sink, "isochronous_out", None) is not None:
                    UsbScoStreamer(transport.sink)
                device = Device.from_config_with_hci(make_device_configuration(config), transport.source, transport.sink)
                yield device

        async def link_phone(device: Device) -> None:
            async with contextlib.AsyncExitStack() as session:
                pairing_controller = PairingController(device, config.device_name) if config.web_token else None
                if pairing_controller is not None:
                    session.push_async_callback(pairing_controller.close)
                if web:
                    web.pairing = pairing_controller
                connected, on_dlc = await listen_for_phone(device, pairing_controller, config.recordings_directory)
                phone.relink(*await next_line(device, connected, on_dlc))  # a reopened dongle: same phone, service and clients
                log.info("phone connected")
                await keep_linked(device, phone, connected, on_dlc)

        bluetooth = asyncio.create_task(keep_controller(open_device, link_phone))
        stack.callback(bluetooth.cancel)
        service = PhoneService(phone, config.policy, config.event_buffer)
        if web:
            web.service = service
        if mcp:
            await McpProjection(service).build().run_stdio_async()
            return
        async with PhapServer(service, config.token, config.heartbeat_s).serve(config.host, config.port) as port:
            log.info("PhAP listening on ws://%s:%d/phap", config.host, port)
            await bluetooth


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments[:1] == ["describe-setup"]:
        setup_parser = argparse.ArgumentParser(prog="telefony-app describe-setup")
        setup_parser.add_argument("--json", action="store_true", required=True, help="print the setup manifest as JSON")
        setup_parser.parse_args(arguments[1:])
        manifest = {
            "provider": "telefony-app",
            "setupVersion": SETUP_VERSION,
            "providerVersion": __version__,
            "fields": describe_setup_fields(),
        }
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("config", type=Path, help="TOML config (see telefony.example.toml)")
    parser.add_argument("--mcp", action="store_true", help="serve MCP on stdio instead of PhAP")
    args = parser.parse_args(arguments)
    # Logs go to stderr: with --mcp, stdout carries the MCP protocol.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(message)s")
    try:
        asyncio.run(run(load_config(args.config), args.config, args.mcp))
    except TransportInitError as failed:
        print("telefony-app: dongle not found", file=sys.stderr)
        print(f"({failed}; check it is plugged in, bound to libusbK, and matches [device] transport)", file=sys.stderr)
        sys.exit(NO_DONGLE_EXIT)


if __name__ == "__main__":
    main()
