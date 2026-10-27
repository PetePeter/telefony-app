"""The demo call agent against the real PhAP server and service, with a fake phone and a scripted LLM."""

import asyncio
import contextlib
import json
from pathlib import Path

import pytest

from examples.call_agent.agent import CallAgent, LlmError, PhapClient, summarise_forecast
from tests.conftest import ALLOWED, BLOCKED, TOKEN
from telefony_app.phone import Contact, SmsMessage
from tests.fakes import FakePhone

SAMPLE = json.loads((Path(__file__).parent / "open_meteo_sample.json").read_text(encoding="utf-8"))


class ScriptedLlm:
    """Replies from a script; records every conversation it was shown."""

    def __init__(self, *replies: dict | Exception) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    async def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        self.seen.append([dict(m) for m in messages])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def say(text: str) -> dict:
    return {"role": "assistant", "content": text}


def use(name: str, **args) -> dict:
    call = {"id": f"t-{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
    return {"role": "assistant", "content": None, "tool_calls": [call]}


async def no_weather() -> dict:
    raise AssertionError("weather was not expected")


@pytest.fixture
async def run_agent(start_server, phone: FakePhone):
    """Start an agent on a real server around the fake phone; returns it once it is waiting for calls."""
    async with contextlib.AsyncExitStack() as stack:

        async def _run(llm: ScriptedLlm, fetch_weather=no_weather) -> CallAgent:
            client = await stack.enter_async_context(PhapClient.connect(await start_server(phone), TOKEN))
            agent = CallAgent(client, llm, fetch_weather=fetch_weather)
            task = asyncio.create_task(agent.run_forever())
            stack.callback(task.cancel)
            return agent

        yield _run


async def until(condition, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


def hung_up(phone: FakePhone) -> bool:
    return ("hangup",) in phone.commands


async def test_allowed_caller_is_answered_and_greeted(run_agent, phone: FakePhone):
    await run_agent(ScriptedLlm())
    phone.ring(ALLOWED)
    await until(lambda: ("answer",) in phone.commands)
    phone.remote_answers()
    await until(lambda: len(phone.spoken) == 1)
    assert "hello" in phone.spoken[0].lower()


async def test_blocked_caller_is_never_answered(run_agent, phone: FakePhone):
    await run_agent(ScriptedLlm())
    phone.ring(BLOCKED)
    await asyncio.sleep(0.3)
    assert ("answer",) not in phone.commands


async def answered(run_agent, phone: FakePhone, llm: ScriptedLlm, **kwargs) -> CallAgent:
    agent = await run_agent(llm, **kwargs)
    phone.ring(ALLOWED)
    await until(lambda: ("answer",) in phone.commands)
    phone.remote_answers()
    await until(lambda: len(phone.spoken) == 1)  # the greeting
    return agent


async def test_what_the_caller_says_goes_to_the_llm_and_its_reply_is_spoken(run_agent, phone: FakePhone):
    llm = ScriptedLlm(say("I'm fine, thanks."))
    await answered(run_agent, phone, llm)
    phone.caller_says("how are you")
    await until(lambda: len(phone.spoken) == 2)
    assert phone.spoken[1] == "I'm fine, thanks."
    assert llm.seen[0][-1] == {"role": "user", "content": "how are you"}


async def test_time_tool_result_is_fed_back_and_the_answer_spoken(run_agent, phone: FakePhone):
    llm = ScriptedLlm(use("time_now"), say("It's late."))
    await answered(run_agent, phone, llm)
    phone.caller_says("what time is it")
    await until(lambda: len(phone.spoken) == 2)
    assert phone.spoken[1] == "It's late."
    tool_result = llm.seen[1][-1]
    assert tool_result["role"] == "tool" and tool_result["tool_call_id"] == "t-time_now"
    assert ":" in tool_result["content"]  # a clock time


async def test_weather_tool_uses_the_forecast(run_agent, phone: FakePhone):
    async def sample() -> dict:
        return SAMPLE

    llm = ScriptedLlm(use("weather"), say("Rain likely."))
    await answered(run_agent, phone, llm, fetch_weather=sample)
    phone.caller_says("will it rain")
    await until(lambda: len(phone.spoken) == 2)
    assert llm.seen[1][-1]["content"] == summarise_forecast(SAMPLE)


def test_forecast_summary_reads_a_real_open_meteo_response():
    assert summarise_forecast(SAMPLE) == (
        "Now 18°C. Today 14 to 21°C, 92% chance of rain. Tomorrow 14 to 23°C, 32% chance of rain."
    )


async def test_hang_up_tool_says_goodbye_and_ends_the_call(run_agent, phone: FakePhone):
    llm = ScriptedLlm(use("hang_up"), say("Bye now."))
    await answered(run_agent, phone, llm)
    phone.caller_says("that's all, bye")
    await until(lambda: hung_up(phone))
    assert phone.spoken[-1] == "Bye now."


async def test_only_the_caller_or_hang_up_ends_the_call(run_agent, phone: FakePhone):
    llm = ScriptedLlm(say("Sure, anything else?"))
    await answered(run_agent, phone, llm)
    phone.caller_says("thanks")
    await until(lambda: len(phone.spoken) == 2)
    await asyncio.sleep(0.5)  # a long silence
    assert not hung_up(phone)


async def test_after_the_caller_hangs_up_the_agent_takes_the_next_call(run_agent, phone: FakePhone):
    await answered(run_agent, phone, ScriptedLlm())
    phone.remote_hangs_up()
    await asyncio.sleep(0.1)
    phone.commands.clear()
    phone.ring(ALLOWED)
    await until(lambda: ("answer",) in phone.commands)


async def test_llm_failure_apologises_and_the_call_carries_on(run_agent, phone: FakePhone):
    await answered(run_agent, phone, ScriptedLlm(LlmError("503"), say("Back again.")))
    phone.caller_says("hello?")
    await until(lambda: len(phone.spoken) == 2)
    assert "sorry" in phone.spoken[-1].lower()
    phone.caller_says("hello again")
    await until(lambda: phone.spoken[-1] == "Back again.")
    assert not hung_up(phone)


# --- phone tools: the agent's PhAP access, still guarded by the app's allowlist ---


async def tool_result(run_agent, phone: FakePhone, name: str, **args) -> str:
    """Have the LLM use one tool during a call; returns what the tool told it."""
    llm = ScriptedLlm(use(name, **args), say("Done."))
    await answered(run_agent, phone, llm)
    phone.caller_says("please do it")
    await until(lambda: len(llm.seen) == 2)
    return llm.seen[1][-1]["content"]


async def test_add_to_call_conferences_an_allowed_number(run_agent, phone: FakePhone):
    await tool_result(run_agent, phone, "add_to_call", number=ALLOWED)
    assert ("consult", ALLOWED) in phone.commands


async def test_a_blocked_number_is_refused_and_the_llm_is_told(run_agent, phone: FakePhone):
    result = await tool_result(run_agent, phone, "add_to_call", number=BLOCKED)
    assert "not allowed" in result.lower()
    assert not any(c[0] == "consult" for c in phone.commands)
    await until(lambda: phone.spoken[-1] == "Done.")  # the call carries on


async def test_send_sms_texts_an_allowed_number(run_agent, phone: FakePhone):
    await tool_result(run_agent, phone, "send_sms", number=ALLOWED, text="see you at 6")
    assert ("sms_send", ALLOWED, "see you at 6") in phone.commands


async def test_list_sms_shows_the_inbox(run_agent, phone: FakePhone):
    phone.sms["h1"] = SmsMessage("h1", ALLOWED, "Sam", "running late", "2026-10-27T09:00:00Z")
    result = await tool_result(run_agent, phone, "list_sms")
    assert "running late" in result and "Sam" in result


async def test_find_contact_returns_numbers(run_agent, phone: FakePhone):
    phone.contacts_list = [Contact("Sam Smith", [("CELL", "0491570156")])]
    result = await tool_result(run_agent, phone, "find_contact", query="sam")
    assert "0491570156" in result


async def test_hold_puts_the_call_on_hold(run_agent, phone: FakePhone):
    await tool_result(run_agent, phone, "hold")
    assert ("hold",) in phone.commands
