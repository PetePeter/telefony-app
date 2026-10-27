"""The MCP projection: turn-based listen/say over the real service, plus the allowlist."""

import asyncio

import pytest

from telefony_app.mcp_projection import McpProjection
from telefony_app.phone import PhoneError
from telefony_app.service import PhapError, PhoneService
from tests.conftest import ALLOWED, BLOCKED
from tests.fakes import FakePhone


@pytest.fixture
def mcp(phone: FakePhone, policy) -> McpProjection:
    return McpProjection(PhoneService(phone, policy))


async def test_listen_returns_speech_buffered_while_saying(mcp: McpProjection, phone: FakePhone):
    await mcp.dial(ALLOWED)
    phone.remote_answers()
    phone.speak_gate = asyncio.Event()
    say = asyncio.create_task(mcp.say("tell me more"))
    await asyncio.sleep(0.05)
    phone.caller_says("well")  # spoken over our reply
    phone.caller_says("it is like this")
    phone.speak_gate.set()
    assert await say == "spoken"
    assert await mcp.listen(timeout_s=1) == {"heard": ["well", "it is like this"], "callEnded": False}


async def test_say_blocks_until_spoken(mcp: McpProjection, phone: FakePhone):
    await mcp.dial(ALLOWED)
    phone.remote_answers()
    phone.speak_gate = asyncio.Event()
    say = asyncio.create_task(mcp.say("hi"))
    await asyncio.sleep(0.05)
    assert not say.done()
    phone.speak_gate.set()
    await say
    assert phone.spoken == ["hi"]


async def test_listen_returns_when_the_call_ends(mcp: McpProjection, phone: FakePhone):
    await mcp.dial(ALLOWED)
    phone.remote_answers()
    listen = asyncio.create_task(mcp.listen(timeout_s=5))
    await asyncio.sleep(0.05)
    phone.remote_hangs_up()
    assert await asyncio.wait_for(listen, 1) == {"heard": [], "callEnded": True}


async def test_listen_times_out_with_nothing_heard(mcp: McpProjection, phone: FakePhone):
    await mcp.dial(ALLOWED)
    phone.remote_answers()
    assert await mcp.listen(timeout_s=0.1) == {"heard": [], "callEnded": False}


async def test_listen_without_a_call_is_an_error(mcp: McpProjection):
    with pytest.raises(PhapError) as error:
        await mcp.listen(timeout_s=0.1)
    assert error.value.code == 1003


async def test_dial_to_blocked_number_is_not_allowed(mcp: McpProjection, phone: FakePhone):
    with pytest.raises(PhapError) as error:
        await mcp.dial(BLOCKED)
    assert error.value.code == 1001
    assert phone.commands == []


async def test_say_reports_a_tts_failure_instead_of_hanging(mcp: McpProjection, phone: FakePhone):
    await mcp.dial(ALLOWED)
    phone.remote_answers()
    phone.failing["speak"] = PhoneError("synthesis failed")
    assert await asyncio.wait_for(mcp.say("hi"), 1) == "failed: synthesis failed"


async def test_listen_after_the_call_ended_returns_at_once(mcp: McpProjection, phone: FakePhone):
    await mcp.dial(ALLOWED)
    phone.remote_answers()
    phone.remote_hangs_up()
    assert await mcp.listen(timeout_s=5) == {"heard": [], "callEnded": True}
    assert await asyncio.wait_for(mcp.listen(timeout_s=5), 0.5) == {"heard": [], "callEnded": True}
