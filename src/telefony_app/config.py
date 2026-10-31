"""Startup configuration. Nothing here can be changed over PhAP or MCP, so a client cannot widen its own allowlist.

Only the operator's web UI (admin token, its own port) edits it; policy changes apply live, the rest on restart.
"""

import ipaddress
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from telefony_app.bt.dialing import normalize_number
from telefony_app.service import Policy

TTS_BACKENDS = ("auto", "piper", "sapi", "none")
TTS_DEFAULT = "auto"
PIPER_MODEL_DEFAULT = None
SAPI_VOICE_DEFAULT = None
STT_MODEL_DEFAULT = "small.en"
STT_DISABLED = "none"
SETUP_VERSION = 1

VOICE_SETUP_FIELDS: tuple[dict[str, Any], ...] = (
    {
        "key": "voice.tts",
        "label": "Text-to-speech mode",
        "type": "enum",
        "choices": TTS_BACKENDS,
        "default": TTS_DEFAULT,
        "restartRequired": True,
        "help": "Empty uses the default auto mode; none disables speech output.",
    },
    {
        "key": "voice.piper_model",
        "label": "Piper model path",
        "type": "path",
        "default": PIPER_MODEL_DEFAULT,
        "restartRequired": True,
        "help": "Path to the Piper ONNX model; leave empty to use the provider default.",
    },
    {
        "key": "voice.sapi_voice",
        "label": "SAPI voice",
        "type": "string",
        "default": SAPI_VOICE_DEFAULT,
        "restartRequired": True,
        "help": "Installed Windows SAPI voice name; leave empty to use the Windows default voice.",
    },
    {
        "key": "voice.stt_model",
        "label": "Speech-to-text model (Whisper)",
        "type": "string",
        "choices": (STT_DISABLED,),
        "default": STT_MODEL_DEFAULT,
        "restartRequired": True,
        "help": "Whisper model name or path. Use none to disable; empty retains the legacy off behavior.",
    },
)


class ConfigError(ValueError):
    pass


@dataclass
class Config:
    transport: str
    token: str
    policy: Policy
    host: str = "127.0.0.1"
    port: int = 7781
    heartbeat_s: float = 10
    event_buffer: int = 1000
    tts: str = TTS_DEFAULT
    sapi_voice: str | None = SAPI_VOICE_DEFAULT
    piper_model: str | None = PIPER_MODEL_DEFAULT
    stt_model: str | None = STT_MODEL_DEFAULT
    web_host: str = "127.0.0.1"
    web_port: int = 7782
    web_token: str | None = None  # no [web] admin_token: no web UI
    recordings_directory: Path | None = None
    device_name: str = "telefony"


def load_config(path: Path) -> Config:
    with path.open("rb") as f:
        return parse_config(tomllib.load(f))


def describe_setup_fields() -> list[dict[str, Any]]:
    fields = []
    for setting in VOICE_SETUP_FIELDS:
        descriptor = setting.copy()
        if "choices" in descriptor:
            descriptor["choices"] = list(descriptor["choices"])
        fields.append(descriptor)
    return fields


def parse_config(raw: dict) -> Config:
    device, policy, phap, voice, web, recordings = (raw.get(k, {}) for k in ("device", "policy", "phap", "voice", "web", "recordings"))
    transport = _required(device, "transport", "device.transport")
    device_name = _device_name(device.get("name", "telefony"))
    token = _required(phap, "token", "phap.token")
    fallback = policy.get("fallback_policy", "ring")
    if fallback not in ("ring", "reject"):
        raise ConfigError(f"policy.fallback_policy must be ring or reject, not {fallback!r}")
    outbound = policy.get("outbound", "allowlist")
    if outbound not in ("allowlist", "all"):
        raise ConfigError(f"policy.outbound must be allowlist or all, not {outbound!r}")
    tts = voice.get("tts", TTS_DEFAULT)
    if tts not in TTS_BACKENDS:
        raise ConfigError(f"voice.tts must be one of {', '.join(TTS_BACKENDS)}, not {tts!r}")
    if "sapi_voice" in voice and not isinstance(voice["sapi_voice"], str):
        raise ConfigError("voice.sapi_voice must be a string")
    sapi_voice = voice.get("sapi_voice", SAPI_VOICE_DEFAULT)
    allowlist = list(policy.get("allowlist", []))
    for number in allowlist:
        try:
            normalize_number(str(number))
        except ValueError:
            raise ConfigError(f"policy.allowlist: {number!r} is not a phone number") from None
    host, port = _loopback(phap.get("bind", "127.0.0.1:7781"), "phap.bind")
    web_host, web_port = _loopback(web.get("bind", "127.0.0.1:7782"), "web.bind")
    recording_directory = recordings.get("directory")
    if recording_directory is not None:
        if not isinstance(recording_directory, str) or not Path(recording_directory).is_absolute():
            raise ConfigError("recordings.directory must be an absolute path")
        recording_directory = Path(recording_directory)
    return Config(
        transport=transport,
        token=token,
        policy=Policy(
            allowlist=allowlist,
            fallback_policy=fallback,
            ring_timeout_s=_number(policy, "ring_timeout_s", 15, float, "policy"),
            country_code=policy.get("country_code", "+61"),
            outbound=outbound,
        ),
        device_name=device_name,
        host=host,
        port=port,
        heartbeat_s=_number(phap, "heartbeat_s", 10, float, "phap"),
        event_buffer=_number(phap, "event_buffer", 1000, int, "phap"),
        tts=tts,
        sapi_voice=sapi_voice,
        piper_model=voice.get("piper_model", PIPER_MODEL_DEFAULT),
        stt_model=_stt_model(voice.get("stt_model", STT_MODEL_DEFAULT)),
        web_host=web_host,
        web_port=web_port,
        web_token=web.get("admin_token") or None,
        recordings_directory=recording_directory,
    )


def _stt_model(value: Any) -> str | None:
    return None if value == STT_DISABLED else value


def _device_name(value: object) -> str:
    if not isinstance(value, str):
        raise ConfigError("device.name must be a string")
    name = value.strip() or "telefony"
    encoded = name.encode("utf-8")
    if len(encoded) > 248:
        name = encoded[:248].decode("utf-8", errors="ignore")
    return name


def _loopback(bind: str, name: str) -> tuple[str, int]:
    host, _, port = bind.rpartition(":")
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        raise ConfigError(f"{name} {host!r} is not an IP address") from None
    if not loopback:
        raise ConfigError(f"{name} {host} is not loopback; remote binds need TLS, which v0.1 does not provide")
    return host, _number({"port": port}, "port", 0, int, name)


def _number(section: dict, key: str, default, kind: type, name: str):
    try:
        return kind(section.get(key, default))
    except (TypeError, ValueError):
        raise ConfigError(f"{name}.{key} must be a number, not {section[key]!r}") from None


def _required(section: dict, key: str, name: str) -> str:
    if not section.get(key):
        raise ConfigError(f"{name} is required")
    return section[key]
