import asyncio
import contextlib
from collections.abc import Callable
from datetime import datetime
from typing import cast

import pytest
import websockets
from bumble import pairing as bumble_pairing
from bumble.device import Connection, Device
from websockets.typing import Subprotocol

from telefony_app.bt.pairing import PairingController
from telefony_app.phap.server import PhapServer
from tests.conftest import TOKEN, Client
from tests.web.conftest import Web


class FakeConnection:
    peer_address = "AA:BB:CC:DD:EE:FF"
    peer_name = "Test Phone"

    def __init__(self) -> None:
        self.listeners: dict[str, list] = {}

    def on(self, event: str, callback) -> None:
        self.listeners.setdefault(event, []).append(callback)

    def emit(self, event: str, *args) -> None:
        for callback in self.listeners.get(event, []):
            callback(*args)


class FakeDevice:
    def __init__(self) -> None:
        self.pairing_config_factory: Callable[[Connection], bumble_pairing.PairingConfig] | None = None
        self.discoverability: list[bool] = []
        self.fail_enable = False

    async def set_discoverable(self, enabled: bool) -> None:
        if enabled and self.fail_enable:
            raise RuntimeError("adapter not ready")
        self.discoverability.append(enabled)


def make_pairing_config(device: FakeDevice, connection: Connection) -> bumble_pairing.PairingConfig:
    assert device.pairing_config_factory is not None
    return device.pairing_config_factory(connection)


def pairing_id(value: dict[str, object]) -> str:
    result = value["pairingId"]
    assert isinstance(result, str)
    return result


def setup_pairing(web: Web) -> tuple[FakeDevice, PairingController]:
    device = FakeDevice()
    pairing = PairingController(cast(Device, device))
    pairing.mark_ready()
    web.app.pairing = pairing
    return device, pairing


async def test_pairing_window_contract_and_live_phap_session(web: Web):
    device, _ = setup_pairing(web)
    service = web.connect_phone()
    phap = PhapServer(service, token=TOKEN, heartbeat_s=30)

    async with phap.serve("127.0.0.1", 0) as port:
        async with websockets.connect(f"ws://127.0.0.1:{port}/phap", subprotocols=[Subprotocol("phap")]) as ws:
            client = Client(ws)
            assert (await client.hello())["result"]["state"] == service.state()

            idle = await web.http.request("GET", "/api/setup/pairing")
            assert idle.json() == {"state": "idle", "deviceName": "telefony"}

            started = await web.http.request("POST", "/api/setup/pairing", {"timeoutSeconds": 30})
            assert started.status == 200
            assert started.headers["content-type"].startswith("application/json")
            data = started.json()
            assert data["state"] == "discoverable" and data["deviceName"] == "telefony"
            assert datetime.fromisoformat(data["expiresAt"].replace("Z", "+00:00"))
            assert device.discoverability == [True]
            assert (await web.http.request("GET", "/api/setup/pairing")).json() == data
            assert (await web.http.request("POST", "/api/setup/pairing", {})).status == 409

            status = await client.request("status")
            assert status["result"]["state"] == service.state()
            cancelled = await web.http.request("POST", "/api/setup/pairing/cancel", {"pairingId": data["pairingId"]})
            assert cancelled.json() == {"pairingId": data["pairingId"], "state": "cancelled"}
            assert device.discoverability[-1] is False
            assert (await client.request("status"))["result"]["state"] == service.state()


async def test_numeric_comparison_is_bound_to_address_and_code(web: Web):
    device, pairing = setup_pairing(web)
    started = (await web.http.request("POST", "/api/setup/pairing", {"timeoutSeconds": 30})).json()
    connection = FakeConnection()
    config = make_pairing_config(device, cast(Connection, connection))
    comparison = asyncio.create_task(config.delegate.compare_numbers(123, digits=6))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    pending = (await web.http.request("GET", "/api/setup/pairing")).json()
    assert pending["state"] == "awaiting-confirmation"
    assert pending["phone"] == {"address": connection.peer_address, "name": connection.peer_name}
    assert pending["numericCode"] == "000123"

    stale = await web.http.request("POST", "/api/setup/pairing/confirm", {
        "pairingId": pairing_id(started), "address": "00:00:00:00:00:00", "numericCode": "000123", "accept": True,
    })
    assert stale.status == 409
    assert (await web.http.request("GET", "/api/setup/pairing")).json()["state"] == "awaiting-confirmation"

    confirmed = await asyncio.wait_for(web.http.request("POST", "/api/setup/pairing/confirm", {
        "pairingId": pairing_id(started), "address": connection.peer_address, "numericCode": "000123", "accept": True,
    }), timeout=0.5)
    assert confirmed.json() == {"pairingId": pairing_id(started), "state": "pairing"}
    assert (await web.http.request("GET", "/api/setup/pairing")).json()["state"] == "pairing"
    assert await asyncio.wait_for(comparison, timeout=1) is True
    connection.emit(Connection.EVENT_CLASSIC_PAIRING)
    for _ in range(5):
        if pairing.status()["state"] == "paired":
            break
        await asyncio.sleep(0)
    assert (await web.http.request("GET", "/api/setup/pairing")).json()["state"] == "paired"
    await pairing.close()


@pytest.mark.parametrize("body", [
    {"timeoutSeconds": 29}, {"timeoutSeconds": 301}, {"timeoutSeconds": True}, {"timeoutSeconds": 1.5},
])
async def test_pairing_rejects_invalid_timeouts(web: Web, body: dict):
    setup_pairing(web)
    reply = await web.http.request("POST", "/api/setup/pairing", body)
    assert reply.status == 400
    assert set(reply.json()) == {"error"}


async def test_missing_controller_returns_json_and_server_keeps_serving(web: Web, caplog):
    web.app.pairing = None
    reply = await web.http.request("GET", "/api/setup/pairing")
    assert reply.status == 409
    assert reply.json() == {"error": "pairing controller unavailable"}
    assert "pairing controller unavailable" in caplog.text
    assert (await web.http.request("GET", "/api/status")).status == 200


async def test_confirmation_without_a_connected_phone_returns_json(web: Web, caplog):
    setup_pairing(web)
    reply = await web.http.request("POST", "/api/setup/pairing/confirm", {
        "pairingId": "stale", "address": "AA:BB:CC:DD:EE:FF", "numericCode": "123456", "accept": True,
    })
    assert reply.status == 404
    assert reply.json() == {"error": "pairing session not found"}
    assert "pairing API request failed" in caplog.text
    assert (await web.http.request("GET", "/api/status")).status == 200


async def test_unready_controller_returns_logged_json_error(web: Web, caplog):
    pairing = PairingController(cast(Device, FakeDevice()))
    web.app.pairing = pairing
    status = await web.http.request("GET", "/api/setup/pairing")
    assert status.status == 409
    assert status.json() == {"error": "Bluetooth adapter unavailable"}
    reply = await web.http.request("POST", "/api/setup/pairing", {})
    assert reply.status == 409
    assert reply.json() == {"error": "Bluetooth adapter unavailable"}
    assert "pairing API request failed" in caplog.text
    assert (await web.http.request("GET", "/api/status")).status == 200
    await pairing.close()


async def test_adapter_start_failure_returns_logged_json_error(web: Web, caplog):
    device = FakeDevice()
    device.fail_enable = True
    pairing = PairingController(cast(Device, device))
    pairing.mark_ready()
    web.app.pairing = pairing
    reply = await web.http.request("POST", "/api/setup/pairing", {})
    assert reply.status == 409
    assert reply.json() == {"error": "Bluetooth adapter unavailable"}
    assert (await web.http.request("GET", "/api/status")).status == 200
    await pairing.close()
