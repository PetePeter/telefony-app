import pytest

from telefony_app.bt.contacts import Ambiguous, Contact, parse_vcards, resolve_number, search

IOS_VCARDS = (
    "BEGIN:VCARD\r\nVERSION:3.0\r\nFN:Oliver van Kent\r\nN:van Kent;Oliver;;;\r\n"
    "TEL;TYPE=CELL:0491 570 157\r\nEND:VCARD\r\n"
    "BEGIN:VCARD\r\nVERSION:3.0\r\nFN:Zoë\r\n  Hart\r\nitem1.TEL;TYPE=HOME;TYPE=VOICE:(02) 5550-1234\r\n"
    "TEL;TYPE=CELL:+61 491 570 158\r\nEND:VCARD\r\n"
    "BEGIN:VCARD\r\nVERSION:3.0\r\nN:Tran;Mai;;;\r\nTEL:0491570110\r\nEND:VCARD\r\n"
    "BEGIN:VCARD\r\nVERSION:3.0\r\nFN:No Number\r\nEMAIL:x@y.z\r\nEND:VCARD\r\n"
)


def test_parses_names_numbers_and_types():
    oliver, zoe, mai = parse_vcards(IOS_VCARDS)
    assert oliver == Contact("Oliver van Kent", [("CELL", "0491570157")])
    assert zoe == Contact("Zoë Hart", [("HOME", "0255501234"), ("CELL", "+61491570158")])
    assert mai.name == "Mai Tran"  # no FN: built from N


def test_contacts_without_numbers_are_dropped():
    assert "No Number" not in [c.name for c in parse_vcards(IOS_VCARDS)]


def test_quoted_printable_vcard21_name():
    card = "BEGIN:VCARD\r\nVERSION:2.1\r\nFN;CHARSET=UTF-8;ENCODING=QUOTED-PRINTABLE:Jos=C3=A9 =\r\nPerez\r\nTEL;CELL:0491571266\r\nEND:VCARD\r\n"
    assert parse_vcards(card)[0].name == "José Perez"


def test_undialable_tel_is_skipped():
    card = "BEGIN:VCARD\r\nFN:A\r\nTEL:tel:;AT+CHUP\r\nTEL:0491571266\r\nEND:VCARD\r\n"
    assert parse_vcards(card)[0].numbers == [("OTHER", "0491571266")]


BOOK = parse_vcards(IOS_VCARDS) + [Contact("Olivia Smith", [("CELL", "0491571804")])]


def test_search_ignores_case_and_accents():
    assert [c.name for c in search(BOOK, "zoe")] == ["Zoë Hart"]


def test_exact_match_beats_prefix_match():
    assert [c.name for c in search(BOOK, "oliver van kent")] == ["Oliver van Kent"]


def test_partial_words_match():
    assert [c.name for c in search(BOOK, "oli ken")] == ["Oliver van Kent"]


def test_ambiguous_name_is_never_resolved():
    with pytest.raises(Ambiguous) as e:
        resolve_number(BOOK, "Oliv")
    assert len(e.value.candidates) == 2


def test_unique_mobile_wins_over_landline():
    assert resolve_number(BOOK, "Zoe Hart") == "+61491570158"


def test_several_mobiles_is_ambiguous():
    book = [Contact("Al", [("CELL", "0491571804"), ("CELL", "0491572549")])]
    with pytest.raises(Ambiguous):
        resolve_number(book, "Al")


def test_unknown_name():
    with pytest.raises(LookupError):
        resolve_number(BOOK, "Nobody")
