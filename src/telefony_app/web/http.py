"""Just enough HTTP/1.1 for a loopback admin page: one request per connection, JSON bodies, server-sent events."""

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

MAX_HEAD = 16 * 1024
MAX_BODY = 64 * 1024
_REASONS = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 403: "Forbidden", 404: "Not Found",
            413: "Payload Too Large", 500: "Internal Server Error"}  # fmt: skip


class HttpError(Exception):
    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(message or _REASONS.get(status, ""))
        self.status = status


@dataclass
class Request:
    method: str
    path: str
    headers: dict[str, str]  # lower-case names
    body: bytes = b""

    def json(self) -> Any:
        try:
            return json.loads(self.body or b"{}")
        except json.JSONDecodeError:
            raise HttpError(400, "body is not JSON") from None


@dataclass
class Response:
    status: int = 200
    body: bytes = b""
    content_type: str = "application/json"
    headers: dict[str, str] = field(default_factory=dict)
    stream: AsyncIterator[dict] | None = None  # server-sent events, one JSON object each

    @classmethod
    def of(cls, value: Any, status: int = 200, headers: dict[str, str] | None = None) -> "Response":
        return cls(status, json.dumps(value).encode(), headers=headers or {})


Handler = Callable[[Request], Awaitable[Response]]


@contextlib.asynccontextmanager
async def serve(handler: Handler, host: str, port: int) -> AsyncIterator[int]:
    """Serve until the context exits; yields the bound port. Open event streams are cut on exit."""
    connections: set[asyncio.Task] = set()

    async def on_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task
        connections.add(task)
        try:
            await _exchange(handler, reader, writer)
        finally:
            connections.discard(task)
            writer.close()

    server = await asyncio.start_server(on_connection, host, port, limit=MAX_HEAD)
    try:
        yield server.sockets[0].getsockname()[1]
    finally:
        server.close()
        for task in list(connections):
            task.cancel()
        await server.wait_closed()


async def _exchange(handler: Handler, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        request = await _read(reader)
        response = await handler(request)
    except HttpError as error:
        response = Response.of({"error": str(error)}, error.status)
    except Exception:
        log.exception("web request failed")
        response = Response.of({"error": "internal error"}, 500)
    with contextlib.suppress(ConnectionError):
        await _write(writer, response, reader)


async def _read(reader: asyncio.StreamReader) -> Request:
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.LimitOverrunError, asyncio.IncompleteReadError, ValueError):
        raise HttpError(400) from None
    request_line, *lines = head.decode("latin-1").split("\r\n")
    parts = request_line.split(" ")
    if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
        raise HttpError(400)
    headers = {k.strip().lower(): v.strip() for k, sep, v in (line.partition(":") for line in lines if line) if sep}
    try:
        length = int(headers.get("content-length", "0"))
    except ValueError:
        raise HttpError(400) from None
    if not 0 <= length <= MAX_BODY:
        raise HttpError(413)
    body = await reader.readexactly(length) if length else b""
    return Request(parts[0], parts[1].split("?")[0], headers, body)


async def _write(writer: asyncio.StreamWriter, response: Response, reader: asyncio.StreamReader) -> None:
    content_type = "text/event-stream" if response.stream else response.content_type
    headers = {"Content-Type": content_type, "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
    headers |= {"Connection": "close"} if response.stream else {"Content-Length": str(len(response.body)), "Connection": "close"}
    headers |= response.headers
    head = f"HTTP/1.1 {response.status} {_REASONS.get(response.status, '')}\r\n"
    writer.write((head + "".join(f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n").encode())
    if response.stream is None:
        writer.write(response.body)
        await writer.drain()
        return
    await writer.drain()
    hung_up = asyncio.ensure_future(reader.read())  # the client closing is the only way a stream ends
    try:
        while True:
            event = asyncio.ensure_future(anext(response.stream))
            await asyncio.wait([event, hung_up], return_when=asyncio.FIRST_COMPLETED)
            if not event.done():
                event.cancel()
                return
            writer.write(f"data: {json.dumps(event.result())}\n\n".encode())
            await writer.drain()
    finally:
        hung_up.cancel()
