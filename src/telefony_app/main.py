"""telefony-app: bring up the Bluetooth hands-free unit, then serve PhAP or MCP (one or the other, never both).

  telefony-app telefony.toml          # PhAP server for Helm or any agent host
  telefony-app telefony.toml --mcp    # MCP on stdio instead, for a plain LLM CLI that launches it
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from bumble import hci, hfp, rfcomm
from bumble.device import Device, DeviceConfiguration
from bumble.transport import open_transport

from telefony_app.bt.autoconnect import connect_bonded_phone
from telefony_app.bt.bluetooth_phone import BluetoothPhone
from telefony_app.bt.hf_line import HF_CONFIGURATION, HfLine
from telefony_app.bt.map_client import make_mns_sdp_records
from telefony_app.bt.sco_stream import UsbScoStreamer
from telefony_app.config import Config, load_config
from telefony_app.mcp_projection import McpProjection
from telefony_app.phap.server import PhapServer
from telefony_app.service import PhoneService
from telefony_app.voice import make_stt, make_tts

HANDS_FREE_CLASS_OF_DEVICE = 0x200408
SDP_HANDLE, MNS_SDP_HANDLE = 0x00010001, 0x00010002
log = logging.getLogger("telefony_app")


async def connect_phone(device: Device) -> tuple[HfLine, rfcomm.DLC]:
    """Advertise as a hands-free unit and wait for the bonded phone (or a new pairing) to bring up HFP."""
    connected: asyncio.Queue[tuple[HfLine, rfcomm.DLC]] = asyncio.Queue()
    bring_ups: set[asyncio.Task] = set()

    async def bring_up(dlc: rfcomm.DLC) -> None:
        line = HfLine(hfp.HfProtocol(dlc, HF_CONFIGURATION))
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
    await device.set_discoverable(True)
    await device.set_connectable(True)
    if dlc := await connect_bonded_phone(device):
        on_dlc(dlc)
    else:
        log.info("discoverable as 'telefony'; pair it from the phone's Bluetooth settings")
    return await connected.get()


async def run(config: Config, mcp: bool) -> None:
    tts, stt = make_tts(config.tts, config.piper_model), make_stt(config.stt_model)
    log.info("text-to-speech: %s; speech-to-text: %s", type(tts).__name__ if tts else "none", "whisper" if stt else "none")
    async with await open_transport(config.transport) as transport:
        device_config = DeviceConfiguration(
            name="telefony",
            address=hci.Address("F0:F1:F2:F3:F4:F5"),
            classic_enabled=True,
            class_of_device=HANDS_FREE_CLASS_OF_DEVICE,
            keystore="JsonKeyStore",  # keep the phone paired across runs
        )
        if getattr(transport.sink, "isochronous_out", None) is not None:
            UsbScoStreamer(transport.sink)
        device = Device.from_config_with_hci(device_config, transport.source, transport.sink)
        line, dlc = await connect_phone(device)
        log.info("phone connected")
        service = PhoneService(BluetoothPhone(line, dlc, tts, stt), config.policy, config.event_buffer)
        if mcp:
            await McpProjection(service).build().run_stdio_async()
            return
        async with PhapServer(service, config.token, config.heartbeat_s).serve(config.host, config.port) as port:
            log.info("PhAP listening on ws://%s:%d/phap", config.host, port)
            await asyncio.get_running_loop().create_future()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="TOML config (see telefony.example.toml)")
    parser.add_argument("--mcp", action="store_true", help="serve MCP on stdio instead of PhAP")
    args = parser.parse_args()
    # Logs go to stderr: with --mcp, stdout carries the MCP protocol.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(run(load_config(args.config), args.mcp))


if __name__ == "__main__":
    main()
