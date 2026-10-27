import re

_SEPARATORS = re.compile(r"[\s().\-]")
_VALID = re.compile(r"\+?[0-9*#]{1,32}")


def normalize_number(raw: str) -> str:
    """Strip formatting and validate a number for ATD. Rejects anything that could inject AT syntax."""
    number = _SEPARATORS.sub("", raw)
    if not _VALID.fullmatch(number):
        raise ValueError(f"invalid phone number: {raw!r}")
    return number
