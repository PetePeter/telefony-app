import asyncio

from bumble.hci import Address
from bumble.keys import JsonKeyStore, PairingKeys

from telefony_app.bt.autoconnect import bonded_phones

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
