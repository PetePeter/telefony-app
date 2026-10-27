import pytest

from telefony_app.config import load_config
from telefony_app.service import PhapError
from telefony_app.web.app import MASK
from tests.conftest import ALLOWED, BLOCKED
from tests.web.conftest import Web


async def test_saving_rewrites_the_file_keeping_comments(web: Web):
    reply = await web.http.request("POST", "/api/config", {"policy": {"allowlist": [ALLOWED, BLOCKED], "ring_timeout_s": 25}})
    assert reply.status == 200
    text = web.config_path.read_text(encoding="utf-8")
    assert "# my telefony box" in text and "# who may be called" in text
    saved = load_config(web.config_path)
    assert saved.policy.allowlist == [ALLOWED, BLOCKED]
    assert saved.policy.ring_timeout_s == 25


@pytest.mark.parametrize(
    "change,field",
    [
        ({"phap": {"bind": "0.0.0.0:7781"}}, "phap.bind"),
        ({"phap": {"token": ""}}, "phap.token"),
        ({"policy": {"fallback_policy": "answer"}}, "policy.fallback_policy"),
        ({"policy": {"ring_timeout_s": "soon"}}, "policy.ring_timeout_s"),
        ({"policy": {"allowlist": ["+61491570156;ATD"]}}, "policy.allowlist"),
        ({"web": {"bind": "192.168.1.5:7782"}}, "web.bind"),
        ({"nope": {"x": 1}}, "nope"),
    ],
)
async def test_invalid_changes_are_refused_and_the_file_is_untouched(web: Web, change: dict, field: str):
    before = web.config_path.read_text(encoding="utf-8")
    reply = await web.http.request("POST", "/api/config", change)
    assert reply.status == 400
    assert field in reply.json()["error"]
    assert web.config_path.read_text(encoding="utf-8") == before


async def test_policy_changes_apply_to_the_running_service(web: Web):
    service = web.connect_phone()
    await web.http.request("POST", "/api/config", {"policy": {"allowlist": [BLOCKED]}})
    await service.dial(BLOCKED)
    service.calls.clear()
    with pytest.raises(PhapError) as refused:
        await service.dial(ALLOWED)
    assert refused.value.code == 1001


@pytest.mark.parametrize(
    "change,restart",
    [
        ({"policy": {"country_code": "+44"}}, False),
        ({"device": {"transport": "usb:0"}}, True),
        ({"phap": {"token": "new"}}, True),
        ({"voice": {"tts": "sapi"}}, True),
    ],
)
async def test_only_non_policy_changes_need_a_restart(web: Web, change: dict, restart: bool):
    assert (await web.http.request("POST", "/api/config", change)).json() == {"restartRequired": restart}


async def test_secrets_are_masked_and_a_masked_secret_is_kept(web: Web):
    config = (await web.http.request("GET", "/api/config")).json()
    assert config["phap"]["token"] == MASK and config["web"]["admin_token"] == MASK
    assert config["policy"]["allowlist"] == [ALLOWED]
    reply = await web.http.request("POST", "/api/config", {"phap": {"token": MASK}, "web": {"admin_token": MASK}})
    assert reply.json() == {"restartRequired": False}
    saved = load_config(web.config_path)
    assert (saved.token, saved.web_token) == ("phap-s3cret", "admin-s3cret")
