"""The demo call agent against the real PhAP server and service, with a fake phone and a scripted LLM."""

import asyncio
import contextlib
import json
from pathlib import Path

import pytest

from examples.call_agent.agent import UNREACHABLE, CallAgent, LlmError, PhapClient, Phonebook, summarise_forecast
from examples.call_agent.home import init_home
from telefony_app.service import Policy
from tests.conftest import ALLOWED, BLOCKED, TOKEN
from tests.fakes import FakePhone

PHONEBOOK = Phonebook([("Oliver van Kent", ALLOWED), ("Olivia Smith", ALLOWED), ("Mai Tran", BLOCKED)])
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


def use(tool: str, **args) -> dict:
    call = {"id": f"t-{tool}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
    return {"role": "assistant", "content": None, "tool_calls": [call]}


async def no_weather() -> dict:
    raise AssertionError("weather was not expected")


@pytest.fixture
def policy() -> Policy:  # unanswered calls keep ringing, so only the agent can reject one
    return Policy(allowlist=[ALLOWED], fallback_policy="ring", ring_timeout_s=0.2)


@pytest.fixture
async def run_agent(start_server, phone: FakePhone):
    """Start an agent on a real server around the fake phone; returns it once it is waiting for calls."""
    async with contextlib.AsyncExitStack() as stack:

        async def _run(llm: ScriptedLlm, fetch_weather=no_weather) -> CallAgent:
            client = await stack.enter_async_context(PhapClient.connect(await start_server(phone), TOKEN))
            agent = CallAgent(client, llm, fetch_weather=fetch_weather, phonebook=PHONEBOOK)
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


async def test_private_number_is_rejected(run_agent, phone: FakePhone):
    await run_agent(ScriptedLlm())
    phone.ring("")  # caller ID withheld
    await until(lambda: ("reject",) in phone.commands)
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


# --- phone tools: the LLM names people from the harness phonebook; it never sees or gives a number ---


async def tool_result(run_agent, phone: FakePhone, tool: str, **args) -> str:
    """Have the LLM use one tool during a call; returns what the tool told it."""
    llm = ScriptedLlm(use(tool, **args), say("Done."))
    await answered(run_agent, phone, llm)
    phone.caller_says("please do it")
    await until(lambda: len(llm.seen) == 2)
    return llm.seen[1][-1]["content"]


def dialled(phone: FakePhone) -> bool:
    return any(c[0] in ("consult", "dial", "sms_send") for c in phone.commands)


async def test_add_to_call_dials_the_phonebook_number(run_agent, phone: FakePhone):
    result = await tool_result(run_agent, phone, "add_to_call", name="oliver van kent")
    assert ("consult", ALLOWED) in phone.commands
    assert "Oliver van Kent" in result


@pytest.mark.parametrize("name", ["Nobody Known", "Oli", ""])  # unknown, ambiguous (Oliver/Olivia), empty
async def test_unknown_or_ambiguous_names_reach_nobody(run_agent, phone: FakePhone, name: str):
    assert await tool_result(run_agent, phone, "add_to_call", name=name) == UNREACHABLE
    assert not dialled(phone)


async def test_a_number_instead_of_a_name_reaches_nobody(run_agent, phone: FakePhone):
    assert await tool_result(run_agent, phone, "send_sms", name=ALLOWED, text="hi") == UNREACHABLE
    assert not dialled(phone)


async def test_phonebook_entry_the_app_blocks_gets_the_same_answer(run_agent, phone: FakePhone):
    assert await tool_result(run_agent, phone, "add_to_call", name="Mai Tran") == UNREACHABLE


async def test_send_sms_texts_the_phonebook_number(run_agent, phone: FakePhone):
    result = await tool_result(run_agent, phone, "send_sms", name="Olivia Smith", text="see you at 6")
    assert ("sms_send", ALLOWED, "see you at 6") in phone.commands
    assert not any(ch.isdigit() for ch in result.replace("6", ""))  # no number leaks back


async def test_address_book_tools_do_not_exist(run_agent, phone: FakePhone):
    phone.contacts_list = []
    result = await tool_result(run_agent, phone, "find_contact", query="sam")
    assert "unknown tool" in result.lower()


def test_phonebook_reads_a_hand_written_csv(tmp_path: Path):
    path = tmp_path / "phonebook.csv"
    path.write_text('name,number\n  Oliver van Kent , 0491 570 156\n"Smith, Olivia",+61491570157\n\n', encoding="utf-8")
    book = Phonebook.load(path)
    assert book.resolve("oliver van kent") == ("Oliver van Kent", "0491570156")
    assert book.resolve("Smith, Olivia") == ("Smith, Olivia", "+61491570157")


def test_init_creates_the_home_folder_and_never_overwrites(tmp_path: Path):
    home = tmp_path / "telefony-call-agent"
    init_home(home)
    assert (home / "call_agent.toml").exists() and (home / "phonebook.csv").read_text(encoding="utf-8").startswith("name,number")
    (home / "phonebook.csv").write_text("name,number\nMe,0491570156\n", encoding="utf-8")
    init_home(home)
    assert "Me,0491570156" in (home / "phonebook.csv").read_text(encoding="utf-8")
