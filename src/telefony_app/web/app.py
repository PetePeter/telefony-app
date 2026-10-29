"""The operator's web UI: edit telefony.toml, watch the service, manage bonded phones. Agents never see this port."""

import contextlib
import hmac
import logging
import re
import secrets
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import tomlkit
from bumble.keys import KeyStore

from telefony_app.bt.autoconnect import bonded_phones
from telefony_app.bt.pairing import PairingController, PairingError
from telefony_app.config import Config, ConfigError, parse_config
from telefony_app.service import PhoneService
from telefony_app.web import http
from telefony_app.web.http import HttpError, Request, Response

MASK = "********"
SECRETS = {("phap", "token"), ("web", "admin_token")}
EDITABLE = {
    "device": {"transport"},
    "policy": {"allowlist", "country_code", "fallback_policy", "ring_timeout_s", "outbound"},
    "phap": {"bind", "token", "heartbeat_s", "event_buffer"},
    "voice": {"tts", "piper_model", "stt_model"},
    "web": {"bind", "admin_token"},
}
LIVE = {"policy"}  # sections applied to the running service; the rest wait for a restart
_PAGE = Path(__file__).with_name("index.html")
_COOKIE = "telefony_session"
_PAIRING_ADDRESS = re.compile(r"^(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")
log = logging.getLogger(__name__)


class WebApp:
    def __init__(self, config_path: Path, config: Config, keystore: Callable[[], KeyStore | None]) -> None:
        self.config_path = config_path
        self.config = config
        self.keystore = keystore  # the device's, once it is powered on
        self.service: PhoneService | None = None  # set once the phone is connected
        self.pairing: PairingController | None = None  # set once the Bumble device is created
        self._sessions: set[str] = set()
        self._port = 0

    @contextlib.asynccontextmanager
    async def serve(self) -> AsyncIterator[int]:
        async with http.serve(self.handle, self.config.web_host, self.config.web_port) as port:
            self._port = port
            yield port

    async def handle(self, request: Request) -> Response:
        route = (request.method, request.path)
        if route == ("GET", "/"):
            return Response(body=_PAGE.read_bytes(), content_type="text/html; charset=utf-8",
                            headers={"Content-Security-Policy": "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'"})  # fmt: skip
        if request.method == "POST":
            self._check_csrf(request)
        if route in _PAIRING_ROUTES:
            self._check_pairing_origin(request)
        if route == ("POST", "/api/login"):
            return self._login(request)
        if route not in _ROUTES:
            raise HttpError(404)
        self._check_auth(request, bearer_only=route in _PAIRING_ROUTES)
        try:
            return await _ROUTES[route](self, request)
        except PairingError as error:
            log.warning("pairing API request failed: %s", error)
            raise HttpError(error.status, str(error)) from None

    # --- security ---

    def _login(self, request: Request) -> Response:
        if not self._is_admin(str(request.json().get("token", ""))):
            raise HttpError(401)
        session = secrets.token_urlsafe(32)
        self._sessions.add(session)
        return Response.of({}, headers={"Set-Cookie": f"{_COOKIE}={session}; HttpOnly; SameSite=Strict; Path=/"})

    def _check_auth(self, request: Request, *, bearer_only: bool = False) -> None:
        bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
        cookies = {}
        for cookie in request.headers.get("cookie", "").split(";"):
            name, _, value = cookie.strip().partition("=")
            cookies[name] = value
        authorized = self._is_admin(bearer) or (not bearer_only and cookies.get(_COOKIE) in self._sessions)
        if not authorized:
            raise HttpError(401)

    def _is_admin(self, token: str) -> bool:
        expected = self.config.web_token or ""
        return bool(expected) and hmac.compare_digest(token.encode(), expected.encode())

    def _check_csrf(self, request: Request) -> None:
        """A custom header forces a CORS preflight, which this server never grants; a foreign Origin is refused too."""
        origin = request.headers.get("origin")
        ours = {f"http://{self.config.web_host}:{self._port}", f"http://localhost:{self._port}"}
        if request.headers.get("x-telefony") != "1" or (origin is not None and origin not in ours):
            raise HttpError(403)

    def _check_pairing_origin(self, request: Request) -> None:
        origin = request.headers.get("origin")
        expected = f"http://{self.config.web_host}:{self._port}"
        if origin is not None and origin != expected:
            raise HttpError(403)

    def _pairing(self) -> PairingController:
        if self.pairing is None:
            log.error("pairing controller unavailable")
            raise HttpError(409, "pairing controller unavailable")
        return self.pairing

    @staticmethod
    def _json_object(request: Request) -> dict[str, object]:
        value = request.json()
        if not isinstance(value, dict):
            raise HttpError(400, "body must be a JSON object")
        return value

    # --- config ---

    async def get_config(self, request: Request) -> Response:
        document = tomlkit.parse(self.config_path.read_text(encoding="utf-8")).unwrap()
        shown = {section: {k: v for k, v in document.get(section, {}).items() if k in keys} for section, keys in EDITABLE.items()}
        for section, key in SECRETS:
            if key in shown[section]:
                shown[section][key] = MASK
        return Response.of(shown)

    async def save_config(self, request: Request) -> Response:
        changes = request.json()
        document = tomlkit.parse(self.config_path.read_text(encoding="utf-8"))
        changed: set[str] = set()
        for section, values in changes.items():
            if section not in EDITABLE or not isinstance(values, dict) or not set(values) <= EDITABLE[section]:
                raise HttpError(400, f"{section}: not an editable setting")
            table = document.setdefault(section, tomlkit.table())
            for key, value in values.items():
                if (section, key) in SECRETS and value == MASK:
                    continue
                if table.get(key) != value:
                    table[key] = value
                    changed.add(section)
        try:
            config = parse_config(document.unwrap())
        except ConfigError as invalid:
            raise HttpError(400, str(invalid)) from None
        self.config_path.write_text(tomlkit.dumps(document), encoding="utf-8")
        self._apply_policy(config)
        return Response.of({"restartRequired": bool(changed - LIVE)})

    def _apply_policy(self, config: Config) -> None:
        """Update the live Policy in place: the running service holds this same object."""
        live = self.config.policy
        live.allowlist, live.country_code = config.policy.allowlist, config.policy.country_code
        live.fallback_policy, live.ring_timeout_s = config.policy.fallback_policy, config.policy.ring_timeout_s
        live.outbound = config.policy.outbound

    # --- status ---

    async def status(self, request: Request) -> Response:
        if self.service is None:
            return Response.of({"phone": "waiting"})
        return Response.of({"phone": "connected", "state": self.service.state()})

    async def events(self, request: Request) -> Response:
        if self.service is None:
            raise HttpError(404, "phone not connected yet")
        return Response(stream=_subscribe(self.service))

    # --- bonded phones ---

    async def phones(self, request: Request) -> Response:
        keystore = self.keystore()
        found = await bonded_phones(keystore) if keystore else []
        return Response.of({"phones": [str(address) for address in found]})

    async def forget(self, request: Request) -> Response:
        keystore, address = self.keystore(), str(request.json().get("address", ""))
        if keystore is None or await keystore.get(address) is None:
            raise HttpError(404, f"no bonded phone {address!r}")
        await keystore.delete(address)
        return Response.of({})

    # --- pairing setup ---

    async def pairing_status(self, request: Request) -> Response:
        del request
        return Response.of(self._pairing().status())

    async def pairing_start(self, request: Request) -> Response:
        body = self._json_object(request)
        timeout = body.get("timeoutSeconds", 120)
        if type(timeout) is not int or not 30 <= timeout <= 300:
            raise HttpError(400, "timeoutSeconds must be an integer from 30 to 300")
        return Response.of(await self._pairing().start(timeout))

    async def pairing_confirm(self, request: Request) -> Response:
        body = self._json_object(request)
        pairing_id = body.get("pairingId")
        address = body.get("address")
        code = body.get("numericCode")
        accept = body.get("accept")
        if not isinstance(pairing_id, str) or not pairing_id:
            raise HttpError(400, "pairingId is required")
        if not isinstance(address, str) or not _PAIRING_ADDRESS.fullmatch(address):
            raise HttpError(400, "address must be a Bluetooth address")
        if not isinstance(code, str) or not re.fullmatch(r"[0-9]{6}", code):
            raise HttpError(400, "numericCode must be six digits")
        if type(accept) is not bool:
            raise HttpError(400, "accept must be a boolean")
        return Response.of(await self._pairing().confirm(pairing_id, address.upper(), code, accept))

    async def pairing_cancel(self, request: Request) -> Response:
        body = self._json_object(request)
        pairing_id = body.get("pairingId")
        if not isinstance(pairing_id, str) or not pairing_id:
            raise HttpError(400, "pairingId is required")
        return Response.of(await self._pairing().cancel(pairing_id))


async def _subscribe(service: PhoneService) -> AsyncIterator[dict]:
    queue = service.events.subscribe()
    try:
        yield {"type": "snapshot", "state": service.state()}  # where things stand; live events follow
        while True:
            yield await queue.get()
    finally:
        service.events.unsubscribe(queue)


_ROUTES = {
    ("GET", "/api/config"): WebApp.get_config,
    ("POST", "/api/config"): WebApp.save_config,
    ("GET", "/api/status"): WebApp.status,
    ("GET", "/api/events"): WebApp.events,
    ("GET", "/api/phones"): WebApp.phones,
    ("POST", "/api/phones/forget"): WebApp.forget,
    ("GET", "/api/setup/pairing"): WebApp.pairing_status,
    ("POST", "/api/setup/pairing"): WebApp.pairing_start,
    ("POST", "/api/setup/pairing/confirm"): WebApp.pairing_confirm,
    ("POST", "/api/setup/pairing/cancel"): WebApp.pairing_cancel,
}
_PAIRING_ROUTES = {
    ("GET", "/api/setup/pairing"),
    ("POST", "/api/setup/pairing"),
    ("POST", "/api/setup/pairing/confirm"),
    ("POST", "/api/setup/pairing/cancel"),
}
