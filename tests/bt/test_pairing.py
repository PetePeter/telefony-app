import asyncio
from collections.abc import Callable
from typing import cast

import pytest
from bumble import hci, pairing
from bumble.device import Connection, Device
from bumble.keys import JsonKeyStore, PairingKeys

from telefony_app.bt import autoconnect
from telefony_app.bt.pairing import PairingController, PairingError

PHONE = "AA:BB:CC:DD:EE:01/P"


class FakeConnection:
    peer_address = hci.Address("AA:BB:CC:DD:EE:01/P")
    peer_name = "Bonded Phone"

    def __init__(self) -> None:
        self.listeners: dict[str, list] = {}
        self.authenticated = False
        self.encrypted = False
        self.disconnected = False

    def on(self, event: str, callback) -> None:
        self.listeners.setdefault(event, []).append(callback)

    def emit(self, event: str, *args) -> None:
        for callback in self.listeners.get(event, []):
            callback(*args)

    async def authenticate(self) -> None:
        self.authenticated = True

    async def encrypt(self) -> None:
        self.encrypted = True

    async def disconnect(self) -> None:
        self.disconnected = True


class FakeDevice:
    def __init__(self, keystore=None, connection: FakeConnection | None = None) -> None:
        self.keystore = keystore
        self.connection = connection
        self.connections = [connection] if connection else []
        self.pairing_config_factory: Callable[[Connection], pairing.PairingConfig] | None = None
        self.discoverability: list[bool] = []
        self.connectable = True

    async def set_discoverable(self, enabled: bool) -> None:
        self.discoverability.append(enabled)

    def find_connection_by_bd_addr(self, address, transport):
        if self.connection and str(self.connection.peer_address) == str(address):
            return self.connection
        return None

    async def connect(self, address, transport):
        raise AssertionError("a bonded reconnect should reuse the existing link")


def make_pairing_config(device: FakeDevice, connection: Connection) -> pairing.PairingConfig:
    assert device.pairing_config_factory is not None
    return device.pairing_config_factory(connection)


def pairing_id(value: dict[str, object]) -> str:
    result = value["pairingId"]
    assert isinstance(result, str)
    return result


async def test_bumble_delegate_waits_for_matching_numeric_comparison():
    device = FakeDevice()
    pairing_controller = PairingController(cast(Device, device))
    pairing_controller.mark_ready()
    started = await pairing_controller.start(30)
    connection = FakeConnection()
    config = make_pairing_config(device, cast(Connection, connection))
    assert config.mitm and config.bonding
    assert config.delegate.io_capability == pairing.PairingDelegate.DISPLAY_OUTPUT_AND_YES_NO_INPUT

    compare = asyncio.create_task(config.delegate.compare_numbers(987, digits=6))
    await asyncio.sleep(0)
    with pytest.raises(PairingError) as mismatch:
        await pairing_controller.confirm(pairing_id(started), "00:00:00:00:00:00", "000987", True)
    assert mismatch.value.status == 409
    assert pairing_controller.status()["state"] == "awaiting-confirmation"

    confirmation = await asyncio.wait_for(pairing_controller.confirm(
        pairing_id(started), connection.peer_address.to_string(with_type_qualifier=False), "000987", True,
    ), timeout=1)
    assert confirmation["state"] == "pairing"
    assert await asyncio.wait_for(compare, timeout=1) is True
    assert pairing_controller.status()["state"] == "pairing"
    connection.emit(Connection.EVENT_CLASSIC_PAIRING)
    for _ in range(5):
        if pairing_controller.status()["state"] == "paired":
            break
        await asyncio.sleep(0)
    assert pairing_controller.status()["state"] == "paired"
    await pairing_controller.close()


async def test_cancel_rejects_a_pending_pairing_and_stops_discoverability():
    device = FakeDevice()
    pairing_controller = PairingController(cast(Device, device))
    pairing_controller.mark_ready()
    started = await pairing_controller.start(30)
    connection = FakeConnection()
    delegate = make_pairing_config(device, cast(Connection, connection)).delegate
    compare = asyncio.create_task(delegate.compare_numbers(42, digits=6))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert pairing_controller.status()["state"] == "awaiting-confirmation"

    cancelled = await pairing_controller.cancel(pairing_id(started))
    assert cancelled["state"] == "cancelled"
    assert await compare is False
    assert device.discoverability == [True, False]
    await pairing_controller.close()


async def test_explicit_rejection_denies_numeric_comparison():
    device = FakeDevice()
    pairing_controller = PairingController(cast(Device, device))
    pairing_controller.mark_ready()
    started = await pairing_controller.start(30)
    connection = FakeConnection()
    delegate = make_pairing_config(device, cast(Connection, connection)).delegate
    compare = asyncio.create_task(delegate.compare_numbers(424242, digits=6))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    result = await pairing_controller.confirm(
        pairing_id(started), connection.peer_address.to_string(with_type_qualifier=False), "424242", False,
    )
    assert result == {"pairingId": pairing_id(started), "state": "rejected"}
    assert await compare is False
    assert device.discoverability == [True, False]
    await pairing_controller.close()


async def test_bumble_pairing_failure_is_terminal():
    device = FakeDevice()
    pairing_controller = PairingController(cast(Device, device))
    pairing_controller.mark_ready()
    started = await pairing_controller.start(30)
    connection = FakeConnection()
    delegate = make_pairing_config(device, cast(Connection, connection)).delegate
    compare = asyncio.create_task(delegate.compare_numbers(333333, digits=6))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    confirmation = await asyncio.wait_for(pairing_controller.confirm(
        pairing_id(started), connection.peer_address.to_string(with_type_qualifier=False), "333333", True,
    ), timeout=1)
    assert confirmation["state"] == "pairing"
    assert await asyncio.wait_for(compare, timeout=1) is True
    assert pairing_controller.status()["state"] == "pairing"

    connection.emit(Connection.EVENT_CLASSIC_PAIRING_FAILURE, 5)
    for _ in range(5):
        if pairing_controller.status()["state"] == "failed":
            break
        await asyncio.sleep(0)
    assert pairing_controller.status()["state"] == "failed"
    assert device.discoverability == [True, False]
    await pairing_controller.close()


async def test_pairing_window_preserves_a_bonded_reconnect(tmp_path, monkeypatch):
    keystore = JsonKeyStore("F0:F1:F2:F3:F4:F5", str(tmp_path / "keys.json"))
    await keystore.update(PHONE, PairingKeys(link_key=PairingKeys.Key(bytes(16))))
    connection = FakeConnection()
    device = FakeDevice(keystore, connection)
    pairing_controller = PairingController(cast(Device, device))
    pairing_controller.mark_ready()
    started = await pairing_controller.start(30)
    assert device.connections == [connection]
    assert device.connectable is True

    class Multiplexer:
        async def open_dlc(self, channel):
            assert channel == 7
            return "reconnected-dlc"

    class Client:
        def __init__(self, connection) -> None:
            assert connection is device.connection

        async def start(self):
            return Multiplexer()

    async def find_ag_sdp_record(_connection):
        return (7, "hands-free")

    monkeypatch.setattr(autoconnect.hfp, "find_ag_sdp_record", find_ag_sdp_record)
    monkeypatch.setattr(autoconnect.rfcomm, "Client", Client)
    assert await autoconnect.connect_bonded_phone(cast(Device, device)) == "reconnected-dlc"
    assert pairing_controller.status()["state"] == "discoverable"
    assert connection.authenticated and connection.encrypted
    assert not connection.disconnected
    assert device.connections == [connection]
    await pairing_controller.cancel(pairing_id(started))
    await pairing_controller.close()


async def test_expiry_rejects_a_pending_comparison(monkeypatch):
    from telefony_app.bt import pairing as pairing_module

    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(pairing_module.asyncio, "sleep", immediate_sleep)
    device = FakeDevice()
    pairing_controller = PairingController(cast(Device, device))
    pairing_controller.mark_ready()
    await pairing_controller.start(30)
    connection = FakeConnection()
    delegate = make_pairing_config(device, cast(Connection, connection)).delegate
    compare = asyncio.create_task(delegate.compare_numbers(111111, digits=6))
    await real_sleep(0)
    await real_sleep(0)
    assert pairing_controller.status()["state"] == "expired"
    assert await compare is False
    await pairing_controller.close()
