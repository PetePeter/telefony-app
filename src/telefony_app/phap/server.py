"""PhAP 2026-10-27 over WebSocket: hello/welcome, roles, events with resume, idempotent methods, heartbeat."""

import asyncio
import contextlib
import hmac
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.typing import Subprotocol

from telefony_app import __version__
from telefony_app.phap.schema import PROTOCOL_VERSION
from telefony_app.service import BUSY, INVALID_PARAMS, PhapError, PhoneService

log = logging.getLogger(__name__)

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INTERNAL_ERROR = -32700, -32600, -32601, -32603
UNAUTHORIZED, NOT_CONTROLLER, VERSION_MISMATCH = -32001, 1005, 1008
CLOSE_UNSUPPORTED_DATA, CLOSE_NO_HELLO, CLOSE_UNAUTHORIZED, CLOSE_CONFLICT, CLOSE_VERSION = 1003, 4400, 4401, 4409, 4426
IDEMPOTENCY_S = 60
HELLO_TIMEOUT_S = 10

Handler = Callable[[PhoneService, dict], Awaitable[Any]]


def _method(state_changing: bool, run: Callable[[PhoneService, dict], Awaitable[Any]]) -> tuple[bool, Handler]:
    return state_changing, run


async def _sync(value: Any) -> Any:
    return value


METHODS: dict[str, tuple[bool, Handler]] = {
    "call.dial": _method(True, lambda s, p: s.dial(p.get("number"), p.get("contact"), p.get("recordingId"))),
    "call.answer": _method(True, lambda s, p: s.answer(p["callId"], p.get("recordingId"))),
    "call.reject": _method(True, lambda s, p: s.reject(p["callId"])),
    "call.hangup": _method(True, lambda s, p: s.hangup(p["callId"])),
    "call.hold": _method(True, lambda s, p: s.hold(p["callId"])),
    "call.resume": _method(True, lambda s, p: s.resume(p["callId"])),
    "call.say": _method(True, lambda s, p: s.say(p["callId"], p["text"])),
    "call.sendDtmf": _method(True, lambda s, p: s.send_dtmf(p["callId"], p["digits"])),
    "conference.create": _method(True, lambda s, p: s.conference_create(p["callId"], p.get("number"), p.get("contact"))),
    "conference.join": _method(True, lambda s, p: s.conference_join(p["callId"])),
    "sms.send": _method(True, lambda s, p: s.sms_send(p["number"], p["text"])),
    "sms.list": _method(False, lambda s, p: s.sms_list(p.get("limit", 20))),
    "sms.read": _method(False, lambda s, p: s.sms_read(p["handle"])),
    "contacts.search": _method(False, lambda s, p: s.contacts_search(p["query"])),
    "call.history": _method(False, lambda s, p: s.call_history(p.get("limit", 20))),
    "transcript.get": _method(False, lambda s, p: _sync(s.transcript(p["callId"]))),
    "status": _method(False, lambda s, p: _sync(s.state())),
    "engine.start": _method(True, lambda s, p: _sync(s.engine_start())),
    "engine.stop": _method(True, lambda s, p: _sync(s.engine_stop())),
}

# How each method's return value becomes its PhAP result object.
RESULTS: dict[str, Callable[[Any, PhoneService], dict]] = {
    "call.dial": lambda r, s: {"callId": r},
    "call.say": lambda r, s: {"sayId": r},
    "conference.create": lambda r, s: {"callId": r},
    "sms.send": lambda r, s: {"handle": r},
    "sms.list": lambda r, s: {"messages": r},
    "sms.read": lambda r, s: r,
    "contacts.search": lambda r, s: {"matches": r},
    "call.history": lambda r, s: {"calls": r},
    "transcript.get": lambda r, s: {"utterances": r},
    "status": lambda r, s: {"state": r},
    "engine.start": lambda r, s: {"state": s.state()},
    "engine.stop": lambda r, s: {"state": s.state()},
}


def _response(id: Any, result: Any = None, error: PhapError | None = None) -> dict:
    if error:
        body = {"code": error.code, "message": error.message} | ({"data": error.data} if error.data is not None else {})
        return {"jsonrpc": "2.0", "id": id, "error": body}
    return {"jsonrpc": "2.0", "id": id, "result": result}


class PhapServer:
    def __init__(self, service: PhoneService, token: str, heartbeat_s: float = 10) -> None:
        self.service = service
        self.token = token
        self.heartbeat_s = heartbeat_s
        self.instance_id = f"{time.time_ns():x}"
        self._controller: ServerConnection | None = None
        self._done: dict[tuple, tuple[float, asyncio.Future]] = {}  # (identity, request id) -> (completed, response)

    @contextlib.asynccontextmanager
    async def serve(self, host: str, port: int) -> AsyncIterator[int]:
        """Listen until the context exits; yields the bound port (useful when port is 0)."""
        async with serve(self._session, host, port, subprotocols=[Subprotocol("phap")]) as server:
            heartbeat = asyncio.create_task(self._heartbeat())
            try:
                yield next(iter(server.sockets)).getsockname()[1]
            finally:
                heartbeat.cancel()

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_s)
            self.service.events.emit("heartbeat")

    async def _session(self, ws: ServerConnection) -> None:
        queue: asyncio.Queue | None = None
        try:
            accepted = await self._handshake(ws)
            if accepted is None:
                return
            identity, queue = accepted
            pump = asyncio.create_task(self._forward(ws, queue))
            try:
                await self._serve_requests(ws, identity)
            finally:
                pump.cancel()
        except ConnectionClosed:
            pass
        finally:
            if queue is not None:
                self.service.events.unsubscribe(queue)
            if self._controller is ws:
                self._controller = None

    async def _handshake(self, ws: ServerConnection) -> tuple[tuple, asyncio.Queue] | None:
        """Answer hello with welcome (plus any replay). Returns the client identity and its event queue, or None."""
        try:
            async with asyncio.timeout(HELLO_TIMEOUT_S):
                hello = json.loads(await ws.recv())
        except (json.JSONDecodeError, TypeError, TimeoutError):
            await ws.close(CLOSE_NO_HELLO, "hello expected")
            return None
        if not isinstance(hello, dict) or hello.get("method") != "hello" or "id" not in hello:
            await ws.close(CLOSE_NO_HELLO, "hello expected")
            return None
        id, params = hello["id"], hello.get("params") or {}
        if not isinstance(params, dict):
            await self._refuse(ws, id, PhapError(INVALID_PARAMS, "invalid params"), CLOSE_NO_HELLO)
            return None
        token = params.get("token")
        if not isinstance(token, str) or not hmac.compare_digest(token.encode(), self.token.encode()):
            await self._refuse(ws, id, PhapError(UNAUTHORIZED, "unauthorized"), CLOSE_UNAUTHORIZED)
            return None
        if params.get("protocolVersion") != PROTOCOL_VERSION:
            error = PhapError(VERSION_MISMATCH, "version_mismatch", {"supported": [PROTOCOL_VERSION]})
            await self._refuse(ws, id, error, CLOSE_VERSION)
            return None
        role = params.get("role")
        if role not in ("controller", "observer") or not isinstance(params.get("clientInfo"), dict):
            await self._refuse(ws, id, PhapError(INVALID_PARAMS, "invalid params"), CLOSE_NO_HELLO)
            return None
        if role == "controller":
            if self._controller is not None:
                await self._refuse(ws, id, PhapError(BUSY, "busy", {"detail": "another controller is connected"}), CLOSE_CONFLICT)
                return None
            self._controller = ws
        resume_from = params.get("resumeFrom")
        # Subscribe, read the replay and build welcome with no await between: nothing is lost or sent twice.
        queue = self.service.events.subscribe()
        replay, gap = self.service.events.since(resume_from) if isinstance(resume_from, int) else ([], False)
        welcome = {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {"name": "telefony-app", "version": __version__, "instanceId": self.instance_id},
            "capabilities": self.service.capabilities() | {"heartbeat": {"interval": self.heartbeat_s}},
            "role": role,
            "state": self.service.state(),
            "lastSeq": self.service.events.last_seq,
            "gap": gap,
        }
        try:
            await self._send(ws, _response(id, welcome))
            for event in replay:
                if event["type"] != "heartbeat":
                    await self._send(ws, {"jsonrpc": "2.0", "method": "event", "params": event})
        except BaseException:
            self.service.events.unsubscribe(queue)
            raise
        return (params["clientInfo"].get("name"), self.token), queue

    async def _refuse(self, ws: ServerConnection, id: Any, error: PhapError, close_code: int) -> None:
        await self._send(ws, _response(id, error=error))
        await ws.close(close_code, error.message)

    async def _forward(self, ws: ServerConnection, queue: asyncio.Queue) -> None:
        with contextlib.suppress(ConnectionClosed):
            while True:
                await self._send(ws, {"jsonrpc": "2.0", "method": "event", "params": await queue.get()})

    async def _serve_requests(self, ws: ServerConnection, identity: tuple) -> None:
        running: set[asyncio.Task] = set()
        async for message in ws:
            if isinstance(message, bytes):
                await ws.close(CLOSE_UNSUPPORTED_DATA, "text frames only")
                return
            task = asyncio.create_task(self._answer(ws, identity, message))
            running.add(task)
            task.add_done_callback(running.discard)

    async def _answer(self, ws: ServerConnection, identity: tuple, message: str) -> None:
        try:
            request = json.loads(message)
        except json.JSONDecodeError:
            await self._send(ws, _response(None, error=PhapError(PARSE_ERROR, "parse error")))
            return
        id = request.get("id") if isinstance(request, dict) else None
        valid_id = id is None or (isinstance(id, (str, int)) and not isinstance(id, bool))
        if not isinstance(request, dict) or request.get("jsonrpc") != "2.0" or "id" not in request or "method" not in request or not valid_id:
            await self._send(ws, _response(id if valid_id else None, error=PhapError(INVALID_REQUEST, "invalid request")))
            return
        await self._send(ws, await self._dispatch(ws, identity, request))

    async def _dispatch(self, ws: ServerConnection, identity: tuple, request: dict) -> dict:
        id, method, params = request["id"], request["method"], request.get("params") or {}
        if method not in METHODS:
            return _response(id, error=PhapError(METHOD_NOT_FOUND, "method not found"))
        state_changing = METHODS[method][0]
        if state_changing and self._controller is not ws:
            return _response(id, error=PhapError(NOT_CONTROLLER, "not_controller"))
        if not state_changing:
            return await self._run(method, id, params)
        key = (identity, id)
        self._expire()
        if key in self._done:  # a retry gets the same response, even while the first attempt is still running
            return await asyncio.shield(self._done[key][1])
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._done[key] = (float("inf"), future)
        try:
            response = await self._run(method, id, params)
        except BaseException:
            del self._done[key]
            future.cancel()
            raise
        future.set_result(response)
        self._done[key] = (time.monotonic(), future)
        return response

    async def _run(self, method: str, id: Any, params: dict) -> dict:
        try:
            value = await METHODS[method][1](self.service, params)
            return _response(id, RESULTS.get(method, lambda r, s: {})(value, self.service))
        except PhapError as error:
            return _response(id, error=error)
        except (KeyError, TypeError, AttributeError):
            return _response(id, error=PhapError(INVALID_PARAMS, "invalid params"))
        except Exception:
            log.exception("%s failed", method)
            return _response(id, error=PhapError(INTERNAL_ERROR, "internal error"))

    def _expire(self) -> None:
        now = time.monotonic()
        for stale in [k for k, (when, _) in self._done.items() if now - when > IDEMPOTENCY_S]:
            del self._done[stale]

    async def _send(self, ws: ServerConnection, frame: dict) -> None:
        await ws.send(json.dumps(frame))
