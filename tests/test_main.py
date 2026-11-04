import asyncio
import contextlib
import json
import sys

from bumble import hci
from bumble.core import PhysicalTransport
from bumble.device import Device
from bumble.link import LocalLink

from telefony_app import __version__
from telefony_app.config import parse_config
from telefony_app.bt import autoconnect
from telefony_app.bt.autoconnect import connect_bonded_phone
from telefony_app import main as telefony_main
from telefony_app.main import keep_controller, main, make_device_configuration
from tests.bt.fake_phone import make_bonded_device, make_device

PHONE = "C8:58:95:E0:2A:F5/P"


def test_describe_setup_prints_deterministic_versioned_json_without_a_config(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "argv", ["telefony-app", "describe-setup", "--json"])
    monkeypatch.chdir(tmp_path)

    main()
    first = capsys.readouterr()
    main()
    second = capsys.readouterr()

    manifest = json.loads(first.out)
    assert manifest["provider"] == "telefony-app"
    assert type(manifest["setupVersion"]) is int
    assert manifest["setupVersion"] == 1
    assert manifest["providerVersion"] == __version__
    assert first.out == second.out
    assert first.err == second.err == ""


def test_device_configuration_changes_name_without_changing_bluetooth_address():
    config = parse_config({
        "device": {"transport": "usb:0", "name": "Timbre - Work"},
        "phap": {"token": "t"},
    })

    device = make_device_configuration(config)

    assert device.name == "Timbre - Work"
    assert device.address == hci.Address("F0:F1:F2:F3:F4:F5")


async def test_a_wedged_dongle_is_closed_and_reopened_until_one_answers(monkeypatch):
    # Seen live: the dongle stopped taking USB transfers and telefony paged the phone into the void for 11 hours.
    monkeypatch.setattr(autoconnect, "_RECONNECT_PAUSE_SECONDS", 0)
    opened: list[Device] = []
    closed: list[Device] = []

    @contextlib.asynccontextmanager
    async def open_device():
        device = await make_bonded_device(LocalLink(), PHONE, wedged=len(opened) < 2)
        device.command_timeout = 0.05
        opened.append(device)
        try:
            yield device
        finally:
            closed.append(device)

    async def link_phone(device: Device) -> None:
        await connect_bonded_phone(device)  # the healthy third dongle pages; nobody answers; done

    await asyncio.wait_for(keep_controller(open_device, link_phone), 5)
    assert len(opened) == 3 and closed == opened


async def test_a_bonded_phone_link_that_brings_no_hands_free_line_is_dropped(monkeypatch):
    # Seen live: after a drop the phone relinked without HFP; the keepalive held that link and nothing paged for 14 h.
    monkeypatch.setattr(autoconnect, "_RECONNECT_PAUSE_SECONDS", 0)
    monkeypatch.setattr(telefony_main, "RECONNECT_PAUSE_S", 0.05)
    monkeypatch.setattr(telefony_main, "LINE_DEADLINE_S", 0.2)
    link = LocalLink()
    device = await make_bonded_device(link, PHONE)
    phone = await make_device(link, PHONE.removesuffix("/P"))  # bonded, but serves no hands-free gateway
    half_open = await phone.connect(device.public_address, transport=PhysicalTransport.BR_EDR)
    dropped = asyncio.Event()
    half_open.on(half_open.EVENT_DISCONNECTION, lambda _reason: dropped.set())

    async with asyncio.TaskGroup() as tasks:
        pager = tasks.create_task(telefony_main._page_until_back(device, lambda _dlc: None))
        await asyncio.wait_for(dropped.wait(), 5)
        pager.cancel()


async def test_a_link_from_an_unbonded_phone_is_left_to_finish_pairing(monkeypatch):
    monkeypatch.setattr(telefony_main, "RECONNECT_PAUSE_S", 0.05)
    monkeypatch.setattr(telefony_main, "LINE_DEADLINE_S", 0.2)
    link = LocalLink()
    device = await make_bonded_device(link, PHONE)
    stranger = await make_device(link, "D0:D1:D2:D3:D4:D5")
    pairing = await stranger.connect(device.public_address, transport=PhysicalTransport.BR_EDR)

    pager = asyncio.create_task(telefony_main._page_until_back(device, lambda _dlc: None))
    await asyncio.sleep(0.6)
    pager.cancel()
    assert pairing in stranger.connections.values()
