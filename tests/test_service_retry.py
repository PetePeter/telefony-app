"""Phone commands ride out a Bluetooth drop: retried while nothing reached the far end, never once it may have."""

import asyncio

import pytest

from telefony_app import service as service_module
from telefony_app.phone import CallMaybePlaced, CallNotPlaced, LinkDown, PhoneError, SmsMaybeSent
from telefony_app.service import PhapError, PhoneService, Policy
from tests.conftest import ALLOWED
from tests.fakes import FakePhone


@pytest.fixture
def phone() -> FakePhone:
    return FakePhone()


@pytest.fixture
def service(phone: FakePhone) -> PhoneService:
    return PhoneService(phone, Policy(allowlist=[ALLOWED]))


@pytest.fixture
def pauses(monkeypatch) -> list[float]:
    """Backoff pauses the service took, without waiting them out."""
    taken: list[float] = []
    real_sleep = asyncio.sleep

    async def sleep(seconds: float, *_) -> None:
        if seconds:  # zero-second sleeps are the tests' own yields
            taken.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(service_module.asyncio, "sleep", sleep)
    return taken


async def test_sms_send_waits_for_the_link_and_sends_once_it_is_back(service, phone, pauses):
    phone.link_up = False

    async def back_after_two_pauses() -> None:
        while len(pauses) < 2:
            await asyncio.sleep(0)
        phone.link_up = True

    returning = asyncio.create_task(back_after_two_pauses())
    handle = await service.sms_send(ALLOWED, "hi")
    await returning
    assert handle
    assert pauses == [1, 2]
    assert phone.commands == [("sms_send", ALLOWED, "hi")]


async def test_a_timeout_before_the_message_left_is_retried(service, phone, pauses):
    phone.failing["sms_send"] = LinkDown("phone did not answer OBEX 0x80")

    async def recover() -> None:
        while not pauses:
            await asyncio.sleep(0)
        del phone.failing["sms_send"]

    recovering = asyncio.create_task(recover())
    assert await service.sms_send(ALLOWED, "hi")
    await recovering
    assert pauses == [1]
    assert len([c for c in phone.commands if c[0] == "sms_send"]) == 2


async def test_gives_up_with_no_link_after_five_retries(service, phone, pauses):
    phone.failing["sms_send"] = LinkDown("down")
    with pytest.raises(PhapError) as failed:
        await service.sms_send(ALLOWED, "hi")
    assert failed.value.message == "no_link"
    assert pauses == [1, 2, 4, 8, 16]
    assert len(phone.commands) == 6


async def test_an_sms_that_may_have_left_is_never_sent_again(service, phone, pauses):
    phone.failing["sms_send"] = SmsMaybeSent("sms may have been sent")
    with pytest.raises(PhapError) as failed:
        await service.sms_send(ALLOWED, "hi")
    assert failed.value.message == "phone_error"
    assert "may have been sent" in failed.value.data["detail"]
    assert pauses == []
    assert phone.commands == [("sms_send", ALLOWED, "hi")]


async def test_dial_waits_for_the_link_when_the_call_could_not_be_placed(service, phone, pauses):
    phone.link_up = False

    async def back() -> None:
        while not pauses:
            await asyncio.sleep(0)
        phone.link_up = True

    returning = asyncio.create_task(back())
    call_id = await service.dial(number=ALLOWED)
    await returning
    assert service.calls[call_id].state == "dialing"
    assert [c for c in phone.commands if c[0] == "dial"] == [("dial", ALLOWED)]


async def test_dial_refused_at_once_is_retried(service, phone, pauses):
    phone.failing["dial"] = CallNotPlaced("ERROR")

    async def accept() -> None:
        while not pauses:
            await asyncio.sleep(0)
        del phone.failing["dial"]

    accepting = asyncio.create_task(accept())
    await service.dial(number=ALLOWED)
    await accepting
    assert len([c for c in phone.commands if c[0] == "dial"]) == 2


@pytest.mark.parametrize("failure", [CallMaybePlaced("call may have been placed"), PhoneError("BUSY"), PhoneError("NO ANSWER")])
async def test_dial_that_may_have_reached_the_network_is_never_repeated(service, phone, pauses, failure):
    phone.failing["dial"] = failure
    with pytest.raises(PhapError) as failed:
        await service.dial(number=ALLOWED)
    assert failed.value.message == "phone_error"
    assert pauses == []
    assert len([c for c in phone.commands if c[0] == "dial"]) == 1
