"""Parse the phone's PBAP call history (cch.vcf): one vCard per call, X-IRMC-CALL-DATETIME giving direction and time."""

from dataclasses import dataclass
from datetime import UTC, datetime

from telefony_app.bt.contacts import unfold, value
from telefony_app.bt.dialing import normalize_number

_DIRECTIONS = {"MISSED": "missed", "RECEIVED": "received", "DIALED": "dialed"}


@dataclass
class CallLogEntry:
    direction: str  # missed | received | dialed
    number: str  # "withheld" when the phone has none
    name: str | None  # untrusted
    ts: str  # ISO-8601


def _iso(stamp: str) -> str | None:
    try:
        moment = datetime.strptime(stamp[:15], "%Y%m%dT%H%M%S")
    except ValueError:
        return None
    return moment.replace(tzinfo=UTC).isoformat() if stamp.endswith("Z") else moment.astimezone().isoformat()


def parse_call_log(text: str) -> list[CallLogEntry]:
    entries: list[CallLogEntry] = []
    name = number = direction = ts = None
    for line in unfold(text).splitlines():
        key, sep, raw = line.partition(":")
        if not sep:
            continue
        prop, *params = key.split(";")
        prop = prop.split(".")[-1].upper()
        if prop == "BEGIN":
            name = number = direction = ts = None
        elif prop == "FN":
            name = value(params, raw).strip() or name
        elif prop == "TEL" and raw.strip():
            try:
                number = normalize_number(value(params, raw))
            except ValueError:
                pass
        elif prop == "X-IRMC-CALL-DATETIME":
            kinds = {p.split("=", 1)[-1].upper() for p in params}
            direction = next((_DIRECTIONS[k] for k in kinds if k in _DIRECTIONS), None)
            ts = _iso(raw.strip())
        elif prop == "END" and direction and ts:
            entries.append(CallLogEntry(direction, number or "withheld", name, ts))
    return entries
