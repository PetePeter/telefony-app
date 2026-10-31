from bumble import hci

from telefony_app.config import parse_config
from telefony_app.main import make_device_configuration


def test_device_configuration_changes_name_without_changing_bluetooth_address():
    config = parse_config({
        "device": {"transport": "usb:0", "name": "Timbre - Work"},
        "phap": {"token": "t"},
    })

    device = make_device_configuration(config)

    assert device.name == "Timbre - Work"
    assert device.address == hci.Address("F0:F1:F2:F3:F4:F5")
