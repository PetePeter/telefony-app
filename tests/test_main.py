import json
import sys

from bumble import hci

from telefony_app import __version__
from telefony_app.config import parse_config
from telefony_app.main import main, make_device_configuration


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
