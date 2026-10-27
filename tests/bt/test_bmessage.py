from telefony_app.bt.bmessage import build_sms, parse_listing, parse_message

IOS_SMS = (
    "BEGIN:BMSG\r\nVERSION:1.0\r\nSTATUS:UNREAD\r\nTYPE:SMS_GSM\r\nFOLDER:telecom/msg/inbox\r\n"
    "BEGIN:VCARD\r\nVERSION:2.1\r\nN:Lane;Bella\r\nTEL:+61491570313\r\nEND:VCARD\r\n"
    "BEGIN:BENV\r\nBEGIN:VCARD\r\nVERSION:2.1\r\nN:\r\nTEL:0491570737\r\nEND:VCARD\r\n"
    "BEGIN:BBODY\r\nCHARSET:UTF-8\r\nLENGTH:44\r\nBEGIN:MSG\r\nRunning late ☕\r\nsee you at 6\r\nEND:MSG\r\n"
    "END:BBODY\r\nEND:BENV\r\nEND:BMSG\r\n"
).encode()


def test_parse_takes_originator_not_recipient_and_keeps_multiline_text():
    message = parse_message(IOS_SMS)
    assert message.sender == "+61491570313"
    assert message.text == "Running late ☕\r\nsee you at 6"


def test_built_sms_round_trips_and_length_counts_utf8_bytes():
    data = build_sms("0491570110", "héllo")
    assert b"TEL:0491570110" in data and b"FOLDER:telecom/msg/outbox" in data
    assert b"LENGTH:28\r\n" in data  # "BEGIN:MSG\r\n" + 6 bytes + "\r\nEND:MSG\r\n"
    assert parse_message(data).text == "héllo"


def test_text_cannot_inject_envelope_lines():
    message = parse_message(build_sms("0491570110", "hi\r\nEND:MSG\r\nTEL:0491571491"))
    assert message.text == "hi\r\nEND:MSG\r\nTEL:0491571491"


LISTING = b"""<?xml version="1.0"?>
<MAP-msg-listing version="1.0">
  <msg handle="0400000000000001" subject="Running late" datetime="20261026T101500" sender_name="Bella"
       sender_addressing="+61491570313" type="SMS_GSM" read="no"/>
  <msg handle="0400000000000002" subject="ok" datetime="20261025T090000" sender_name="Anna" type="SMS_GSM" read="yes"/>
</MAP-msg-listing>
"""


def test_listing_parses_handles_senders_and_read_flag():
    first, second = parse_listing(LISTING)
    assert (first.handle, first.sender, first.subject, first.read) == ("0400000000000001", "+61491570313", "Running late", False)
    assert (second.sender, second.read) == ("Anna", True)  # no number: fall back to the name
