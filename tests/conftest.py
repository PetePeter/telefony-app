import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import jsonschema
import pytest
import websockets
from websockets.typing import Subprotocol

from telefony_app.phap.schema import SCHEMA
from telefony_app.phap.server import PhapServer
from telefony_app.service import Policy, PhoneService
from tests.fakes import FakePhone

TOKEN = "secret"
ALLOWED = "+61491570156"
BLOCKED = "+61491570159"
VERSION = "2026-10-27"

_validator = jsonschema.Draft202012Validator(SCHEMA)


def assert_compliant(frame: dict[str, Any], definition: str) -> None:
    """Every frame the server sends must match the PhAP schema."""
    _validator.evolve(schema={"$ref": f"#/$defs/{definition}", "$defs": SCHEMA["$defs"]}).validate(frame)


class Client:
    """A PhAP client over a real WebSocket that checks each server frame against the schema."""

    def __init__(self, ws: websockets.ClientConnection) -> None:
        self.ws = ws
        self.events: list[dict] = []
        self._next_id = 0

    async def recv(self) -> dict:
        frame = json.loads(await self.ws.recv())
        if "id" in frame:
            assert_compliant(frame, "JSONRPCResponse")
        else:
            assert_compliant(frame, "JSONRPCNotification")
            self.events.append(frame["params"])
        return frame

    async def request(self, method: str, params: dict | None = None, id: int | None = None) -> dict:
        if id is None:
            self._next_id += 1
            id = self._next_id
        await self.ws.send(json.dumps({"jsonrpc": "2.0", "id": id, "method": method, "params": params or {}}))
        while True:
            frame = await self.recv()
            if frame.get("id") == id:
                return frame

    async def hello(self, role: str = "controller", token: str = TOKEN, version: str = VERSION, **extra: Any) -> dict:
        params = {"protocolVersion": version, "token": token, "role": role, "clientInfo": {"name": "test", "version": "1"}}
        return await self.request("hello", params | extra)

    async def event(self, type: str, timeout: float = 2.0) -> dict:
        """The next event of this type, including ones already received."""
        async with asyncio.timeout(timeout):
            while True:
                for e in self.events:
                    if e["type"] == type:
                        self.events.remove(e)
                        return e
                await self.recv()


@pytest.fixture
def phone() -> FakePhone:  # the phone shared by `connect` clients
    return FakePhone()


@pytest.fixture
def policy() -> Policy:
    return Policy(allowlist=[ALLOWED], fallback_policy="reject", ring_timeout_s=0.2)


@pytest.fixture
async def start_server(policy: Policy) -> AsyncIterator[Callable[..., Awaitable[str]]]:
    """Start a real PhAP server around a phone; returns its URI."""
    async with contextlib.AsyncExitStack() as stack:

        async def _start(phone: FakePhone, event_buffer: int = 1000) -> str:
            phap = PhapServer(PhoneService(phone, policy, event_buffer=event_buffer), token=TOKEN, heartbeat_s=30)
            port = await stack.enter_async_context(phap.serve("127.0.0.1", 0))
            return f"ws://127.0.0.1:{port}/phap"

        yield _start


@pytest.fixture
async def open_client() -> AsyncIterator[Callable[[str], Awaitable[Client]]]:
    async with contextlib.AsyncExitStack() as stack:

        async def _open(uri: str) -> Client:
            ws = await stack.enter_async_context(websockets.connect(uri, subprotocols=[Subprotocol("phap")]))
            return Client(ws)

        yield _open


@pytest.fixture
async def connect(start_server, open_client, phone: FakePhone) -> Callable[[], Awaitable[Client]]:
    """Clients of one server around the shared fake phone."""
    uri = await start_server(phone)
    return lambda: open_client(uri)


@pytest.fixture
def connect_with(start_server, open_client) -> Callable[[FakePhone], Awaitable[Client]]:
    """A client of a fresh server around the given phone."""

    async def _connect(phone: FakePhone) -> Client:
        return await open_client(await start_server(phone))

    return _connect


@pytest.fixture
def connect_with_buffer(start_server, open_client, phone: FakePhone) -> Callable[[int], Awaitable[Client]]:
    """A client of a fresh server, around the shared phone, that keeps only the last n events."""

    async def _connect(n: int) -> Client:
        return await open_client(await start_server(phone, event_buffer=n))

    return _connect
