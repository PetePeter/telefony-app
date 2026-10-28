from telefony_app.bt.call_log import CallLogEntry, parse_call_log

# Shaped like iOS/Android PBAP cch.vcf: newest first, X-IRMC-CALL-DATETIME carries the direction.
CCH = (
    "BEGIN:VCARD\r\nVERSION:3.0\r\nFN:Oliver van Kent\r\nN:van Kent;Oliver;;;\r\nTEL;TYPE=CELL:0491 570 157\r\n"
    "X-IRMC-CALL-DATETIME;MISSED:20261028T194500\r\nEND:VCARD\r\n"
    "BEGIN:VCARD\r\nVERSION:3.0\r\nFN:\r\nN:\r\nTEL:\r\nX-IRMC-CALL-DATETIME;TYPE=RECEIVED:20261028T120000\r\nEND:VCARD\r\n"
    "BEGIN:VCARD\r\nVERSION:3.0\r\nN:;;;;\r\nTEL:+61 491 570 158\r\nX-IRMC-CALL-DATETIME;DIALED:20261027T090102Z\r\nEND:VCARD\r\n"
)


def test_parses_direction_number_name_and_time():
    missed, withheld, dialed = parse_call_log(CCH)
    assert missed == CallLogEntry("missed", "0491570157", "Oliver van Kent", missed.ts)
    assert missed.ts.startswith("2026-10-28T19:45:00")
    assert withheld.direction == "received" and withheld.number == "withheld" and withheld.name is None
    assert dialed == CallLogEntry("dialed", "+61491570158", None, "2026-10-27T09:01:02+00:00")


def test_entries_without_a_call_time_are_skipped():
    assert parse_call_log("BEGIN:VCARD\r\nVERSION:3.0\r\nFN:X\r\nTEL:0491570157\r\nEND:VCARD\r\n") == []
