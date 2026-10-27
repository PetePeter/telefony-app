"""Startup configuration. Nothing here can be changed over PhAP or MCP, so a client cannot widen its own allowlist."""

import ipaddress
import tomllib
from dataclasses import dataclass
from pathlib import Path

from telefony_app.service import Policy

TTS_BACKENDS = ("auto", "piper", "sapi", "none")


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
    tts: str = "auto"
    piper_model: str | None = None
    stt_model: str | None = "small.en"


def load_config(path: Path) -> Config:
    with path.open("rb") as f:
        raw = tomllib.load(f)
    device, policy, phap, voice = (raw.get(k, {}) for k in ("device", "policy", "phap", "voice"))
    transport = _required(device, "transport", "device.transport")
    token = _required(phap, "token", "phap.token")
    fallback = policy.get("fallback_policy", "ring")
    if fallback not in ("ring", "reject"):
        raise ConfigError(f"policy.fallback_policy must be ring or reject, not {fallback!r}")
    tts = voice.get("tts", "auto")
    if tts not in TTS_BACKENDS:
        raise ConfigError(f"voice.tts must be one of {', '.join(TTS_BACKENDS)}, not {tts!r}")
    host, _, port = phap.get("bind", "127.0.0.1:7781").rpartition(":")
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        raise ConfigError(f"phap.bind {host!r} is not an IP address") from None
    if not loopback:
        raise ConfigError(f"phap.bind {host} is not loopback; remote binds need TLS, which v0.1 does not provide")
    return Config(
        transport=transport,
        token=token,
        policy=Policy(
            allowlist=list(policy.get("allowlist", [])),
            fallback_policy=fallback,
            ring_timeout_s=float(policy.get("ring_timeout_s", 15)),
            country_code=policy.get("country_code", "+61"),
        ),
        host=host,
        port=int(port),
        heartbeat_s=float(phap.get("heartbeat_s", 10)),
        event_buffer=int(phap.get("event_buffer", 1000)),
        tts=tts,
        piper_model=voice.get("piper_model"),
        stt_model=voice.get("stt_model", "small.en"),
    )


def _required(section: dict, key: str, name: str) -> str:
    if not section.get(key):
        raise ConfigError(f"{name} is required")
    return section[key]
