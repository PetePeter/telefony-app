from pathlib import Path

import pytest

from telefony_app.config import ConfigError, load_config


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "telefony.toml"
    path.write_text(text, encoding="utf-8")
    return path


FULL = """
[device]
transport = "usb:10D7:B012+sco=2"

[policy]
allowlist = ["+447700900123"]
fallback_policy = "reject"
ring_timeout_s = 20

[phap]
bind = "127.0.0.1:7781"
token = "s3cret"

[voice]
tts = "none"
"""


def test_loads_a_full_config(tmp_path: Path):
    config = load_config(write(tmp_path, FULL))
    assert config.transport == "usb:10D7:B012+sco=2"
    assert config.policy.allowlist == ["+447700900123"]
    assert config.policy.fallback_policy == "reject"
    assert config.policy.ring_timeout_s == 20
    assert (config.host, config.port, config.token) == ("127.0.0.1", 7781, "s3cret")
    assert config.tts == "none"


def test_defaults_apply_to_optional_settings(tmp_path: Path):
    config = load_config(write(tmp_path, '[device]\ntransport = "usb:0"\n[phap]\ntoken = "t"\n'))
    assert config.policy.fallback_policy == "ring"
    assert config.policy.ring_timeout_s == 15
    assert config.policy.allowlist == []
    assert (config.host, config.port) == ("127.0.0.1", 7781)
    assert config.tts == "auto"


@pytest.mark.parametrize(
    "text,problem",
    [
        ('[phap]\ntoken = "t"\n', "device.transport"),
        ('[device]\ntransport = "usb:0"\n', "phap.token"),
        ('[device]\ntransport = "usb:0"\n[phap]\ntoken = "t"\n[policy]\nfallback_policy = "answer"\n', "fallback_policy"),
        ('[device]\ntransport = "usb:0"\n[phap]\ntoken = "t"\n[voice]\ntts = "espeak"\n', "voice.tts"),
    ],
)
def test_rejects_invalid_configs(tmp_path: Path, text: str, problem: str):
    with pytest.raises(ConfigError, match=problem):
        load_config(write(tmp_path, text))


def test_remote_bind_without_tls_is_refused(tmp_path: Path):
    text = '[device]\ntransport = "usb:0"\n[phap]\ntoken = "t"\nbind = "0.0.0.0:7781"\n'
    with pytest.raises(ConfigError, match="TLS"):
        load_config(write(tmp_path, text))


def test_bind_host_that_is_not_an_ip_is_a_config_error(tmp_path: Path):
    text = '[device]\ntransport = "usb:0"\n[phap]\ntoken = "t"\nbind = "localhost:7781"\n'
    with pytest.raises(ConfigError, match="not an IP address"):
        load_config(write(tmp_path, text))
