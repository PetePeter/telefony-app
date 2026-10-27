"""Parse the phone's vCard phonebook and resolve a spoken name to one number, never guessing between candidates."""

import quopri
import re
import unicodedata
from dataclasses import dataclass, field

from telefony_app.bt.dialing import normalize_number

_FOLD = re.compile(r"\r?\n[ \t]")
_QP_SOFT_BREAK = re.compile(r"=\r?\n")


@dataclass
class Contact:
    name: str
    numbers: list[tuple[str, str]] = field(default_factory=list)  # (type, number), e.g. ("CELL", "0491570157")


class Ambiguous(Exception):
    def __init__(self, message: str, candidates: list[Contact]) -> None:
        super().__init__(message)
        self.candidates = candidates


def _value(params: list[str], raw: str) -> str:
    upper = [p.upper() for p in params]
    if "ENCODING=QUOTED-PRINTABLE" in upper or "QUOTED-PRINTABLE" in upper:
        charset = next((p.split("=", 1)[1] for p in params if p.upper().startswith("CHARSET=")), "utf-8")
        return quopri.decodestring(raw.encode("latin-1")).decode(charset, errors="replace")
    return raw.replace("\\,", ",").replace("\\;", ";")


def _tel_type(params: list[str]) -> str:
    for p in params:
        for t in p.split("=", 1)[-1].split(","):
            if t.upper() in ("CELL", "HOME", "WORK", "MAIN", "IPHONE"):
                return t.upper()
    return "OTHER"


def parse_vcards(text: str) -> list[Contact]:
    text = _QP_SOFT_BREAK.sub("", _FOLD.sub("", text))
    contacts: list[Contact] = []
    current: Contact | None = None
    structured = ""
    for line in text.splitlines():
        key, sep, raw = line.partition(":")
        if not sep:
            continue
        prop, *params = key.split(";")
        prop = prop.split(".")[-1].upper()  # drop iOS "item1." groups
        if prop == "BEGIN":
            current, structured = Contact(""), ""
        elif current is None:
            continue
        elif prop == "FN":
            current.name = _value(params, raw).strip()
        elif prop == "N":
            last, first, *_ = _value(params, raw).split(";") + ["", ""]
            structured = f"{first} {last}".strip()
        elif prop == "TEL":
            try:
                current.numbers.append((_tel_type(params), normalize_number(_value(params, raw))))
            except ValueError:
                pass  # not dialable (e.g. a URI); skip it
        elif prop == "END":
            current.name = current.name or structured
            if current.name and current.numbers:
                contacts.append(current)
            current = None
    return contacts


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).casefold().strip()


def search(contacts: list[Contact], query: str) -> list[Contact]:
    """Exact name matches if any; otherwise contacts whose name words start with every query word."""
    q = _fold(query)
    if exact := [c for c in contacts if _fold(c.name) == q]:
        return exact
    words = q.split()
    return [c for c in contacts if words and all(any(n.startswith(w) for n in _fold(c.name).split()) for w in words)]


def resolve_number(contacts: list[Contact], query: str) -> str:
    """The single number to dial for this name. Raises LookupError if none, Ambiguous if more than one."""
    matches = search(contacts, query)
    if not matches:
        raise LookupError(f"no contact matches {query!r}")
    if len(matches) > 1:
        raise Ambiguous(f"{query!r} matches {', '.join(c.name for c in matches)}", matches)
    contact = matches[0]
    numbers = {n for _, n in contact.numbers}
    if len(numbers) == 1:
        return numbers.pop()
    mobiles = {n for t, n in contact.numbers if t in ("CELL", "IPHONE")}
    if len(mobiles) == 1:
        return mobiles.pop()
    raise Ambiguous(f"{contact.name} has several numbers: {', '.join(sorted(numbers))}", matches)
