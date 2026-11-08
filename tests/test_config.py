from pathlib import Path

import pytest

from telefony_app.config import ConfigError, describe_setup_fields, load_config, parse_config
from telefony_app.voice import make_stt


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
    assert config.policy.outbound == "allowlist"
    assert (config.host, config.port) == ("127.0.0.1", 7781)
    assert config.tts == "auto"
    assert config.sapi_voice is None
    assert config.recordings_directory is None
    assert config.device_name == "telefony"


def test_setup_field_defaults_match_parsed_config_defaults():
    config = parse_config({"device": {"transport": "usb:0"}, "phap": {"token": "t"}})
    actual_defaults = {
        "voice.tts": config.tts,
        "voice.piper_model": config.piper_model,
        "voice.sapi_voice": config.sapi_voice,
        "voice.stt_model": config.stt_model,
        "voice.whisper_cpp_server": config.whisper_cpp_server,
        "voice.whisper_cpp_model": config.whisper_cpp_model,
    }

    fields = describe_setup_fields()

    assert {item["key"]: item["default"] for item in fields} == actual_defaults
    stt = next(item for item in fields if item["key"] == "voice.stt_model")
    assert "none" in stt["choices"]


def test_stt_none_disables_runtime_and_empty_keeps_legacy_off_behavior():
    base = {"device": {"transport": "usb:0"}, "phap": {"token": "t"}}
    disabled = parse_config(base | {"voice": {"stt_model": "none"}})
    legacy_empty = parse_config(base | {"voice": {"stt_model": ""}})

    assert disabled.stt_model is None
    assert make_stt(disabled.stt_model) is None
    assert legacy_empty.stt_model == ""
    assert make_stt(legacy_empty.stt_model) is None


def test_device_name_is_trimmed_and_capped_at_248_utf8_bytes():
    configured = parse_config({
        "device": {"transport": "usb:0", "name": f"  {'🧭' * 63}  "},
        "phap": {"token": "t"},
    })
    assert configured.device_name == "🧭" * 62
    assert len(configured.device_name.encode("utf-8")) == 248


def test_loads_a_configured_sapi_voice(tmp_path: Path):
    text = FULL.replace('tts = "none"', 'tts = "none"\nsapi_voice = "Microsoft Zira Desktop"')
    config = load_config(write(tmp_path, text))
    assert config.sapi_voice == "Microsoft Zira Desktop"


@pytest.mark.parametrize("value", ["42", "true", "[]"])
def test_sapi_voice_must_be_a_string(tmp_path: Path, value: str):
    text = FULL.replace('tts = "none"', f'tts = "none"\nsapi_voice = {value}')
    with pytest.raises(ConfigError, match="voice.sapi_voice must be a string"):
        load_config(write(tmp_path, text))


def test_present_sapi_voice_cannot_be_null():
    with pytest.raises(ConfigError, match="voice.sapi_voice must be a string"):
        parse_config({"device": {"transport": "usb:0"}, "phap": {"token": "t"}, "voice": {"sapi_voice": None}})


def test_loads_absolute_recordings_directory(tmp_path: Path):
    directory = tmp_path / "recordings"
    config = load_config(write(tmp_path, FULL + f'\n[recordings]\ndirectory = "{directory.as_posix()}"\n'))
    assert config.recordings_directory == directory


def test_recordings_directory_must_be_absolute(tmp_path: Path):
    with pytest.raises(ConfigError, match="absolute"):
        load_config(write(tmp_path, FULL + '\n[recordings]\ndirectory = "recordings"\n'))


@pytest.mark.parametrize(
    "text,problem",
    [
        ('[phap]\ntoken = "t"\n', "device.transport"),
        ('[device]\ntransport = "usb:0"\n', "phap.token"),
        ('[device]\ntransport = "usb:0"\n[phap]\ntoken = "t"\n[policy]\nfallback_policy = "answer"\n', "fallback_policy"),
        ('[device]\ntransport = "usb:0"\n[phap]\ntoken = "t"\n[voice]\ntts = "espeak"\n', "voice.tts"),
        ('[device]\ntransport = "usb:0"\n[phap]\ntoken = "t"\n[policy]\noutbound = "some"\n', "policy.outbound"),
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


def test_whisper_cpp_server_and_model_select_the_gpu_engine(tmp_path: Path):
    from telefony_app.voice import WhisperCppStt

    config = load_config(write(tmp_path, FULL + 'stt_model = "none"\nwhisper_cpp_server = "C:/w/whisper-server.exe"\nwhisper_cpp_model = "C:/w/ggml-base.en.bin"\n'))

    assert (config.whisper_cpp_server, config.whisper_cpp_model) == ("C:/w/whisper-server.exe", "C:/w/ggml-base.en.bin")
    stt = make_stt(config.stt_model, config.whisper_cpp_server, config.whisper_cpp_model)
    assert isinstance(stt, WhisperCppStt)
    stt.close()


def test_whisper_cpp_server_without_a_model_is_a_config_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="whisper_cpp_model"):
        load_config(write(tmp_path, FULL + 'whisper_cpp_server = "C:/w/whisper-server.exe"\n'))


def test_whisper_cpp_paths_must_be_strings(tmp_path: Path):
    with pytest.raises(ConfigError, match="whisper_cpp_server"):
        load_config(write(tmp_path, FULL + 'whisper_cpp_server = 42\nwhisper_cpp_model = "m.bin"\n'))
