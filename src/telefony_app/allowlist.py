"""Who may be called, texted or conferenced. Caller ID is untrusted: only exact allowlist matches count."""

from telefony_app.bt.dialing import normalize_number


def _canonical(number: str, country_code: str) -> str | None:
    try:
        number = normalize_number(number)
    except ValueError:
        return None
    return country_code + number[1:] if number.startswith("0") else number


def is_allowed(number: str | None, allow: list[str], country_code: str) -> bool:
    """Exact match after normalising national (0..) and international (+CC..) forms. Malformed numbers never match."""
    canonical = _canonical(number, country_code) if number else None
    return canonical is not None and canonical in {_canonical(n, country_code) for n in allow}
