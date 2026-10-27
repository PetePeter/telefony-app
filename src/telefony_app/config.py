"""Startup configuration. Nothing here can be changed over PhAP or MCP, so a client cannot widen its own allowlist.

Only the operator's web UI (admin token, its own port) edits it; policy changes apply live, the rest on restart.
"""

import ipaddress
import tomllib
from dataclasses import dataclass
from pathlib import Path

from telefony_app.bt.dialing import normalize_number
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
    web_host: str = "127.0.0.1"
    web_port: int = 7782
    web_token: str | None = None  # no [web] admin_token: no web UI


def load_config(path: Path) -> Config:
    with path.open("rb") as f:
        return parse_config(tomllib.load(f))


def parse_config(raw: dict) -> Config:
    device, policy, phap, voice, web = (raw.get(k, {}) for k in ("device", "policy", "phap", "voice", "web"))
    transport = _required(device, "transport", "device.transport")
    token = _required(phap, "token", "phap.token")
    fallback = policy.get("fallback_policy", "ring")
    if fallback not in ("ring", "reject"):
        raise ConfigError(f"policy.fallback_policy must be ring or reject, not {fallback!r}")
    outbound = policy.get("outbound", "allowlist")
    if outbound not in ("allowlist", "all"):
        raise ConfigError(f"policy.outbound must be allowlist or all, not {outbound!r}")
    tts = voice.get("tts", "auto")
    if tts not in TTS_BACKENDS:
        raise ConfigError(f"voice.tts must be one of {', '.join(TTS_BACKENDS)}, not {tts!r}")
    allowlist = list(policy.get("allowlist", []))
    for number in allowlist:
        try:
            normalize_number(str(number))
        except ValueError:
            raise ConfigError(f"policy.allowlist: {number!r} is not a phone number") from None
    host, port = _loopback(phap.get("bind", "127.0.0.1:7781"), "phap.bind")
    web_host, web_port = _loopback(web.get("bind", "127.0.0.1:7782"), "web.bind")
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
        host=host,
        port=port,
        heartbeat_s=_number(phap, "heartbeat_s", 10, float, "phap"),
        event_buffer=_number(phap, "event_buffer", 1000, int, "phap"),
        tts=tts,
        piper_model=voice.get("piper_model"),
        stt_model=voice.get("stt_model", "small.en"),
        web_host=web_host,
        web_port=web_port,
        web_token=web.get("admin_token") or None,
    )


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
