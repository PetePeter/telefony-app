"""A demo PhAP client: a closed-loop phone agent that answers allowlisted callers and talks through an LLM.

It has only three tools (the time, a weather forecast and hanging up), so a caller cannot make it do
anything else. The LLM is any OpenAI-compatible chat API: OpenRouter, or a local llama-server.
"""

import asyncio
import contextlib
import json
import logging
import urllib.request
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from typing import Any, Protocol

import websockets
from websockets.typing import Subprotocol

log = logging.getLogger("call_agent")

PROTOCOL_VERSION = "2026-10-27"
SYSTEM_PROMPT = (
    "You are a friendly phone assistant on a live voice call. Reply in one or two short spoken sentences, "
    "no lists or symbols. Use your tools for the time, the weather, texts, contacts and calls; "
    "only text or call people the caller asks for. "
    "Keep the conversation going until the caller hangs up. Use hang_up only when the caller asks you to end the call."
)
GREETING = "Hello, this is the telefony demo assistant. I can tell you the time and weather, read and send texts, or call someone in."
SORRY = "Sorry, I couldn't manage that. Could you say it another way?"
MAX_TOOL_ROUNDS = 4

# Phone tools: name -> (PhAP method, description, parameters, required). "callId" is filled in with the current call.
# The app's allowlist still guards every dial and SMS, whatever the LLM asks for.
PHONE_TOOLS: dict[str, tuple[str, str, dict, list[str]]] = {
    "add_to_call": ("conference.create", "Call someone and add them to this call (the caller stays on).",
                    {"number": {"type": "string"}, "contact": {"type": "string", "description": "a contact name"}}, []),
    "merge_calls": ("conference.join", "Merge the held call into a conference.", {}, []),
    "hold": ("call.hold", "Put this call on hold.", {}, []),
    "resume": ("call.resume", "Take this call off hold.", {}, []),
    "send_sms": ("sms.send", "Send a text message.", {"number": {"type": "string"}, "text": {"type": "string"}}, ["number", "text"]),
    "list_sms": ("sms.list", "The latest text messages.", {"limit": {"type": "integer"}}, []),
    "read_sms": ("sms.read", "One text message in full.", {"handle": {"type": "string"}}, ["handle"]),
    "find_contact": ("contacts.search", "Look up a contact's numbers by name.", {"query": {"type": "string"}}, ["query"]),
    "phone_status": ("status", "The phone's link and the calls in progress.", {}, []),
    "call_transcript": ("transcript.get", "What has been said on this call so far.", {}, []),
}
CALL_SCOPED = {"conference.create", "conference.join", "call.hold", "call.resume", "transcript.get"}
LOCAL_TOOLS = [
    ("time_now", "The current local date and time."),
    ("weather", "The weather now, and the forecast for today and tomorrow."),
    ("hang_up", "End the call, only when the caller asks to; say goodbye in the same reply."),
]


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    parameters = {"type": "object", "properties": properties, "required": required}
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


TOOLS = [_tool(name, description, {}, []) for name, description in LOCAL_TOOLS] + [
    _tool(name, description, properties, required) for name, (_, description, properties, required) in PHONE_TOOLS.items()
]


class LlmError(Exception):
    pass


class Llm(Protocol):
    async def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        """The assistant message replying to `messages`."""
        ...


class OpenAiLlm:
    """An OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str | None) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key = model, api_key

    async def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        body = json.dumps({"model": self.model, "messages": messages, "tools": tools}).encode()
        headers = {"Content-Type": "application/json"} | ({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})
        request = urllib.request.Request(self.url, body, headers)
        try:
            response = await asyncio.to_thread(lambda: json.load(urllib.request.urlopen(request, timeout=30)))
            return response["choices"][0]["message"]
        except Exception as e:
            raise LlmError(str(e)) from e


def open_meteo(latitude: float, longitude: float) -> Callable[[], Awaitable[dict]]:
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max&timezone=auto&forecast_days=2"
    )

    async def fetch() -> dict:
        return await asyncio.to_thread(lambda: json.load(urllib.request.urlopen(url, timeout=10)))

    return fetch


def summarise_forecast(data: dict) -> str:
    daily = data["daily"]
    days = [
        f"{label} {round(daily['temperature_2m_min'][i])} to {round(daily['temperature_2m_max'][i])}°C, "
        f"{daily['precipitation_probability_max'][i]}% chance of rain."
        for i, label in enumerate(["Today", "Tomorrow"])
    ]
    return " ".join([f"Now {round(data['current']['temperature_2m'])}°C.", *days])


class CallEnded(Exception):
    pass


class PhapClient:
    """A minimal PhAP controller: requests, plus a buffer of pushed events."""

    def __init__(self, ws: websockets.ClientConnection) -> None:
        self.ws = ws
        self._pending: dict[int, asyncio.Future] = {}
        self._events: list[dict] = []
        self._arrived = asyncio.Event()
        self._next_id = 0

    @classmethod
    @contextlib.asynccontextmanager
    async def connect(cls, uri: str, token: str) -> AsyncIterator["PhapClient"]:
        async with websockets.connect(uri, subprotocols=[Subprotocol("phap")]) as ws:
            client = cls(ws)
            reader = asyncio.create_task(client._read())
            try:
                hello = {"protocolVersion": PROTOCOL_VERSION, "token": token, "role": "controller",
                         "clientInfo": {"name": "call-agent", "version": "0.1"}}
                await client.request("hello", hello)
                yield client
            finally:
                reader.cancel()

    async def request(self, method: str, params: dict) -> Any:
        self._next_id += 1
        future = self._pending[self._next_id] = asyncio.get_running_loop().create_future()
        await self.ws.send(json.dumps({"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params}))
        response = await future
        if "error" in response:
            raise RuntimeError(f"{method}: {response['error']['message']}")
        return response["result"]

    async def next_event(self, accept: Callable[[dict], bool], timeout: float | None = None) -> dict:
        """The oldest buffered event that `accept` takes; other events stay buffered for later."""
        async with asyncio.timeout(timeout):
            while True:
                for event in self._events:
                    if accept(event):
                        self._events.remove(event)
                        return event
                self._arrived.clear()
                await self._arrived.wait()

    def drop_events(self, reject: Callable[[dict], bool]) -> None:
        self._events = [e for e in self._events if not reject(e)]

    async def _read(self) -> None:
        async for raw in self.ws:
            frame = json.loads(raw)
            if "id" in frame and (future := self._pending.pop(frame["id"], None)):
                future.set_result(frame)
            elif frame.get("method") == "event" and frame["params"]["type"] != "heartbeat":
                self._events.append(frame["params"])
                self._arrived.set()


class CallAgent:
    def __init__(self, client: PhapClient, llm: Llm, *, fetch_weather: Callable[[], Awaitable[dict]]) -> None:
        self.client, self.llm, self.fetch_weather = client, llm, fetch_weather

    async def run_forever(self) -> None:
        while True:
            ringing = await self.client.next_event(lambda e: e["type"] == "call.ringing" and e.get("allowed", False))
            call_id = ringing["callId"]
            log.info("answering %s", ringing.get("number"))
            try:
                await self.client.request("call.answer", {"callId": call_id})
                await self._converse(call_id)
            except CallEnded:
                log.info("caller hung up")
            except Exception:
                log.exception("call failed")
                with contextlib.suppress(Exception):
                    await self.client.request("call.hangup", {"callId": call_id})
            self.client.drop_events(lambda e: e.get("callId") == call_id)

    async def _converse(self, call_id: str) -> None:
        await self._wait(call_id, lambda e: e["type"] == "call.state" and e["state"] == "active")
        await self._say(call_id, GREETING)
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        while True:
            heard = await self._wait(call_id, lambda e: e["type"] == "call.utterance")  # silence never ends the call
            log.info("heard: %s", heard["text"])
            messages.append({"role": "user", "content": heard["text"]})
            try:
                reply, hang_up = await self._think(call_id, messages)
            except LlmError as e:
                log.warning("LLM failed: %s", e)
                await self._say(call_id, SORRY)  # only the caller ends the call
                continue
            if hang_up:
                return await self._finish(call_id, reply)
            await self._say(call_id, reply)

    async def _think(self, call_id: str, messages: list[dict]) -> tuple[str, bool]:
        """Ask the LLM, running its tool calls, until it answers; also whether it asked to hang up."""
        hang_up = False
        for _ in range(MAX_TOOL_ROUNDS):
            message = await self.llm.chat(messages, TOOLS)
            messages.append(message)
            calls = message.get("tool_calls") or []
            if not calls:
                return message.get("content") or "", hang_up
            for call in calls:
                name = call["function"]["name"]
                hang_up |= name == "hang_up"
                args = json.loads(call["function"].get("arguments") or "{}")
                result = await self._tool(call_id, name, args)
                log.info("tool %s %s -> %s", name, args, result)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        raise LlmError("too many tool rounds")

    async def _tool(self, call_id: str, name: str, args: dict) -> str:
        if name in PHONE_TOOLS:
            method = PHONE_TOOLS[name][0]
            params = args | ({"callId": call_id} if method in CALL_SCOPED else {})
            try:
                return json.dumps(await self.client.request(method, params))
            except Exception as e:
                if "not_allowed" in str(e):
                    return "Not allowed: that number is not on the allowlist. Tell the caller; do not retry."
                return f"Failed: {e}"
        if name == "time_now":
            return datetime.now().astimezone().strftime("%A %d %B %Y, %H:%M")
        if name == "weather":
            try:
                return summarise_forecast(await self.fetch_weather())
            except Exception as e:
                return f"The forecast is unavailable ({e})."
        if name == "hang_up":
            return "The call will end after your reply."
        return f"Unknown tool {name}."

    async def _say(self, call_id: str, text: str) -> None:
        if not text:
            return
        log.info("saying: %s", text)
        try:
            say_id = (await self.client.request("call.say", {"callId": call_id, "text": text}))["sayId"]
        except RuntimeError as e:  # e.g. the call is on hold while someone is added
            log.warning("could not say it: %s", e)
            return
        await self._wait(call_id, lambda e: e["type"] == "call.spoken" and e["sayId"] == say_id)

    async def _finish(self, call_id: str, goodbye: str) -> None:
        with contextlib.suppress(CallEnded):
            await self._say(call_id, goodbye)
            await self.client.request("call.hangup", {"callId": call_id})

    async def _wait(self, call_id: str, accept: Callable[[dict], bool], timeout: float | None = None) -> dict:
        """The next event on this call that `accept` takes; raises CallEnded if the call ends first."""

        def ours(e: dict) -> bool:
            return e.get("callId") == call_id and (accept(e) or (e["type"] == "call.state" and e["state"] == "ended"))

        event = await self.client.next_event(ours, timeout)
        if event["type"] == "call.state" and event["state"] == "ended" and not accept(event):
            raise CallEnded
        return event
