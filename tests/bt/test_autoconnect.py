import asyncio
from typing import cast

import pytest
from bumble.link import LocalLink

from bumble.hci import Address
from bumble.keys import JsonKeyStore, PairingKeys

from telefony_app.bt import autoconnect
from telefony_app.bt.autoconnect import ControllerWedged, bonded_phones, connect_bonded_phone
from tests.bt.fake_phone import WedgedSink, make_bonded_device

PHONE = "C8:58:95:E0:2A:F5/P"


def store(tmp_path, keys: dict[str, PairingKeys]) -> JsonKeyStore:
    ks = JsonKeyStore("F4:4E:FC:25:CA:B1/P", str(tmp_path / "keys.json"))
    for name, k in keys.items():
        asyncio.run(ks.update(name, k))
    return ks


def classic_key() -> PairingKeys:
    return PairingKeys(link_key=PairingKeys.Key(value=bytes(16)), link_key_type=4)


def test_no_bonds_means_nobody_to_dial(tmp_path):
    assert asyncio.run(bonded_phones(store(tmp_path, {}))) == []


def test_bonded_phone_is_returned_as_address(tmp_path):
    assert asyncio.run(bonded_phones(store(tmp_path, {PHONE: classic_key()}))) == [Address(PHONE)]


def test_le_only_bond_is_ignored(tmp_path):
    le_only = PairingKeys(ltk=PairingKeys.Key(value=bytes(16)))
    assert asyncio.run(bonded_phones(store(tmp_path, {PHONE: le_only}))) == []


async def test_a_dongle_that_swallows_every_page_is_reported_wedged(monkeypatch):
    monkeypatch.setattr(autoconnect, "_RECONNECT_PAUSE_SECONDS", 0)
    device = await make_bonded_device(LocalLink(), PHONE, wedged=True)
    device.command_timeout = 0.05
    with pytest.raises(ControllerWedged):
        await connect_bonded_phone(device)
    assert cast(WedgedSink, device.host.hci_sink).dropped == autoconnect.WEDGED_AFTER_TIMEOUTS


async def test_a_phone_out_of_range_is_not_mistaken_for_a_wedged_dongle(monkeypatch):
    monkeypatch.setattr(autoconnect, "_RECONNECT_PAUSE_SECONDS", 0)
    device = await make_bonded_device(LocalLink(), PHONE)  # nobody at PHONE answers the page
    device.command_timeout = 0.05
    assert await connect_bonded_phone(device) is None
