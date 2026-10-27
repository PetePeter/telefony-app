import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from bumble.keys import JsonKeyStore

from telefony_app.config import load_config
from telefony_app.service import PhoneService
from telefony_app.web.app import WebApp
from tests.conftest import ALLOWED
from tests.fakes import FakePhone

ADMIN = "admin-s3cret"
CONFIG = f"""# my telefony box
[device]
transport = "usb:10D7:B012+sco=2"   # the dongle

[policy]
allowlist = ["{ALLOWED}"]        # who may be called
country_code = "+61"
fallback_policy = "ring"
ring_timeout_s = 15

[phap]
bind = "127.0.0.1:7781"
token = "phap-s3cret"

[voice]
tts = "none"

[web]
bind = "127.0.0.1:0"
admin_token = "{ADMIN}"
"""


@dataclass
class Reply:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self):
        return json.loads(self.body)


class Http:
    """A raw HTTP/1.1 client, so tests can also send malformed requests."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.cookie: str | None = None

    async def raw(self, data: bytes) -> Reply:
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        writer.write(data)
        await writer.drain()
        async with asyncio.timeout(3):
            response = await reader.read()
        writer.close()
        head, _, body = response.partition(b"\r\n\r\n")
        status_line, *lines = head.decode().split("\r\n")
        headers = {k.lower(): v.strip() for k, _, v in (line.partition(":") for line in lines)}
        return Reply(int(status_line.split()[1]), headers, body)

    async def request(self, method: str, path: str, body=None, auth: bool = True, headers: dict | None = None) -> Reply:
        payload = b"" if body is None else json.dumps(body).encode()
        sent = {"Host": f"127.0.0.1:{self.port}", "Content-Length": str(len(payload)), "Connection": "close"}
        if method == "POST":
            sent["X-Telefony"] = "1"
        if auth:
            sent["Authorization"] = f"Bearer {ADMIN}"
        if self.cookie:
            sent["Cookie"] = self.cookie
        sent |= headers or {}
        head = "".join(f"{k}: {v}\r\n" for k, v in sent.items() if v is not None)
        return await self.raw(f"{method} {path} HTTP/1.1\r\n{head}\r\n".encode() + payload)

    async def events(self) -> AsyncGenerator[dict]:
        """Server-sent events from /api/events."""
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        writer.write(f"GET /api/events HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {ADMIN}\r\n\r\n".encode())
        await writer.drain()
        try:
            await reader.readuntil(b"\r\n\r\n")
            while True:
                line = (await reader.readline()).decode().strip()
                if line.startswith("data: "):
                    yield json.loads(line[6:])
        finally:
            writer.close()


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "telefony.toml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


@pytest.fixture
def keystore(tmp_path: Path) -> JsonKeyStore:
    return JsonKeyStore("F0:F1:F2:F3:F4:F5", str(tmp_path / "keys.json"))


@dataclass
class Web:
    app: WebApp
    http: Http
    config_path: Path
    phone: FakePhone
    service: PhoneService | None = None

    def connect_phone(self) -> PhoneService:
        self.service = PhoneService(self.phone, self.app.config.policy)
        self.app.service = self.service
        return self.service


@pytest.fixture
async def web(config_path: Path, keystore: JsonKeyStore) -> AsyncIterator[Web]:
    app = WebApp(config_path, load_config(config_path), keystore=lambda: keystore)
    async with app.serve() as port:
        yield Web(app, Http(port), config_path, FakePhone())

