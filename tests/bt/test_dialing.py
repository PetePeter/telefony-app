import pytest

from telefony_app.bt.dialing import normalize_number


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("+44 7700 900123", "+447700900123"),
        ("(020) 7946-0018", "02079460018"),
        ("*#06#", "*#06#"),
        ("123", "123"),
    ],
)
def test_normalizes(raw, expected):
    assert normalize_number(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "abc", "12;ATH", "+", "44+7700", "1" * 33])
def test_rejects(raw):
    with pytest.raises(ValueError):
        normalize_number(raw)
