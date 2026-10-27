from telefony_app.allowlist import is_allowed

OCTAVIO = "0491 570 157"


def test_matches_national_and_international_forms():
    assert is_allowed("+61491570157", [OCTAVIO], "+61")
    assert is_allowed("0491570157", ["+61 491 570 157"], "+61")


def test_national_form_uses_the_configured_country():
    assert not is_allowed("+44491570157", [OCTAVIO], "+61")
    assert is_allowed("+44491570157", [OCTAVIO], "+44")


def test_withheld_or_malformed_numbers_never_match():
    assert not is_allowed(None, [OCTAVIO], "+61")
    assert not is_allowed("", [OCTAVIO], "+61")
    assert not is_allowed("12;ATH", [OCTAVIO], "+61")


def test_stranger_does_not_match():
    assert not is_allowed("0491571491", [OCTAVIO], "+61")
