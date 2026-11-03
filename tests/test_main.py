import asyncio
import contextlib
import json
import sys

from bumble import hci
from bumble.device import Device
from bumble.link import LocalLink

from telefony_app import __version__
from telefony_app.config import parse_config
from telefony_app.bt import autoconnect
from telefony_app.bt.autoconnect import connect_bonded_phone
from telefony_app.main import keep_controller, main, make_device_configuration
from tests.bt.fake_phone import make_bonded_device

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
