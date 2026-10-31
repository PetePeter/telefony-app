"""telefony-app: bring up the Bluetooth hands-free unit, then serve PhAP or MCP (one or the other, never both).

  telefony-app telefony.toml          # PhAP server for Helm or any agent host
  telefony-app telefony.toml --mcp    # MCP on stdio instead, for a plain LLM CLI that launches it
"""

import argparse
import asyncio
import contextlib
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from bumble import hci, hfp, rfcomm
from bumble.device import Device, DeviceConfiguration
from bumble.transport import open_transport
from bumble.transport.common import TransportInitError

from telefony_app import __version__
from telefony_app.bt.autoconnect import connect_bonded_phone
from telefony_app.bt.bluetooth_phone import BluetoothPhone
from telefony_app.bt.hf_line import HF_CONFIGURATION, HfLine
from telefony_app.bt.map_client import make_mns_sdp_records
from telefony_app.bt.pairing import PairingController
from telefony_app.bt.sco_stream import UsbScoStreamer
from telefony_app.config import Config, load_config
from telefony_app.mcp_projection import McpProjection
from telefony_app.phap.server import PhapServer
from telefony_app.service import PhoneService
from telefony_app.voice import make_stt, make_tts
from telefony_app.web.app import WebApp

HANDS_FREE_CLASS_OF_DEVICE = 0x200408
NO_DONGLE_EXIT = 3  # with the stderr line "telefony-app: dongle not found", which Helm matches
SDP_HANDLE, MNS_SDP_HANDLE = 0x00010001, 0x00010002
RECONNECT_PAUSE_S = 5.0
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
    await device.power_on()
    await device.set_connectable(True)
    if pairing_controller is None:  # preserve the legacy first-pair path when the admin API is disabled
        await device.set_discoverable(True)
    else:
        pairing_controller.mark_ready()
    if dlc := await connect_bonded_phone(device):
        on_dlc(dlc)
    elif pairing_controller is not None:
        log.info("waiting for a pairing window from the setup API")
    else:
        log.info("discoverable as 'telefony'; pair it from the phone's Bluetooth settings")
    return connected, on_dlc


async def keep_linked(device: Device, phone: BluetoothPhone, connected: LineQueue, on_dlc: Callable[[rfcomm.DLC], None]) -> None:
    """Whenever the link drops, page the bonded phone until a new line is up (from us or the phone), then relink."""
    while True:
        await phone.link_down.wait()
        log.info("phone link lost; reconnecting")
        pager = asyncio.create_task(_page_until_back(device, on_dlc))
        try:
            line, dlc = await connected.get()
            while dlc.multiplexer.l2cap_channel.connection not in device.connections.values():  # stale, from an old link
                line, dlc = await connected.get()
        finally:
            pager.cancel()
        phone.relink(line, dlc)
        log.info("phone reconnected")


async def _page_until_back(device: Device, on_dlc: Callable[[rfcomm.DLC], None]) -> None:
    """Runs until cancelled: if bringing up a channel fails, the link drops and we page again."""
    while True:
        if not device.connections and (dlc := await connect_bonded_phone(device)):
            on_dlc(dlc)
        await asyncio.sleep(RECONNECT_PAUSE_S)


def make_device_configuration(config: Config) -> DeviceConfiguration:
    return DeviceConfiguration(
        name=config.device_name,
        address=hci.Address("F0:F1:F2:F3:F4:F5"),
        classic_enabled=True,
        class_of_device=HANDS_FREE_CLASS_OF_DEVICE,
        keystore="JsonKeyStore",  # keep the phone paired across runs
    )


async def run(config: Config, config_path: Path, mcp: bool) -> None:
    tts, stt = make_tts(config.tts, config.piper_model, config.sapi_voice), make_stt(config.stt_model)
    log.info("text-to-speech: %s; speech-to-text: %s", type(tts).__name__ if tts else "none", "whisper" if stt else "none")
    async with await open_transport(config.transport) as transport:
        device_config = make_device_configuration(config)
        if getattr(transport.sink, "isochronous_out", None) is not None:
            UsbScoStreamer(transport.sink)
        device = Device.from_config_with_hci(device_config, transport.source, transport.sink)
        async with contextlib.AsyncExitStack() as stack:
            pairing_controller = PairingController(device, config.device_name) if config.web_token else None
            if pairing_controller is not None:
                stack.push_async_callback(pairing_controller.close)
            web = None
            if config.web_token:  # up before the phone, so pairing can be watched from the page
                web = WebApp(config_path, config, keystore=lambda: device.keystore)
                web.pairing = pairing_controller
                port = await stack.enter_async_context(web.serve())
                log.info("web UI on http://%s:%d/", config.web_host, port)
            connected, on_dlc = await listen_for_phone(device, pairing_controller, config.recordings_directory)
            line, dlc = await connected.get()
            log.info("phone connected")
            phone = BluetoothPhone(line, dlc, tts, stt)
            service = PhoneService(phone, config.policy, config.event_buffer)
            relinker = asyncio.create_task(keep_linked(device, phone, connected, on_dlc))
            stack.callback(relinker.cancel)
            if web:
                web.service = service
            if mcp:
                await McpProjection(service).build().run_stdio_async()
                return
            async with PhapServer(service, config.token, config.heartbeat_s).serve(config.host, config.port) as port:
                log.info("PhAP listening on ws://%s:%d/phap", config.host, port)
                await asyncio.get_running_loop().create_future()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("config", type=Path, help="TOML config (see telefony.example.toml)")
    parser.add_argument("--mcp", action="store_true", help="serve MCP on stdio instead of PhAP")
    args = parser.parse_args()
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
