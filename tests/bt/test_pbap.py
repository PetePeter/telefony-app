"""PBAP pull end-to-end over Bumble's in-memory radio link, against a fake phonebook server (PSE)."""

from bumble import core
from bumble.core import PhysicalTransport
from bumble.link import LocalLink

from tests.bt.fake_phone import make_device, serve_obex
from telefony_app.bt import obex
from telefony_app.bt.pbap import PBAP_TARGET, VCARD_30, pull_call_log, pull_contacts

VCARDS = "".join(
    f"BEGIN:VCARD\r\nVERSION:3.0\r\nFN:Contact {i}\r\nTEL;TYPE=CELL:02555000{i:02d}\r\nEND:VCARD\r\n" for i in range(60)
).encode()
CHUNK = 500  # forces several CONTINUE round trips


async def test_pulls_whole_phonebook_across_continue_packets():
    remaining = VCARDS

    def phonebook(request: obex.Packet) -> obex.Packet:
        nonlocal remaining
        chunk, remaining = remaining[:CHUNK], remaining[CHUNK:]
        return obex.Packet(obex.CONTINUE if remaining else obex.SUCCESS, [(obex.BODY if remaining else obex.END_OF_BODY, chunk)])

    link = LocalLink()
    us, phone = await make_device(link, "F0:F1:F2:F3:F4:F5"), await make_device(link, "0C:51:01:73:71:27")
    requests = serve_obex(phone, core.BT_PHONEBOOK_ACCESS_PSE_SERVICE, phonebook)

    connection = await us.connect(phone.public_address, transport=PhysicalTransport.BR_EDR)
    contacts = await pull_contacts(connection)

    assert [c.name for c in contacts] == [f"Contact {i}" for i in range(60)]
    assert contacts[59].numbers == [("CELL", "0255500059")]
    connect, first_get, *more_gets, disconnect = requests
    assert connect.header(obex.TARGET) == PBAP_TARGET
    assert first_get.header(obex.NAME) == "telecom/pb.vcf" and first_get.header(obex.APP_PARAMETERS) == VCARD_30
    assert len(more_gets) >= 2 and all(g.header(obex.CONNECTION_ID) == 1 for g in more_gets)
    assert disconnect.code == obex.DISCONNECT


async def test_pulls_combined_call_history_newest_first_up_to_limit():
    def history(request: obex.Packet) -> obex.Packet:
        card = "BEGIN:VCARD\r\nVERSION:3.0\r\nTEL:0491570157\r\nX-IRMC-CALL-DATETIME;MISSED:20261028T194500\r\nEND:VCARD\r\n"
        return obex.Packet(obex.SUCCESS, [(obex.END_OF_BODY, card.encode())])

    link = LocalLink()
    us, phone = await make_device(link, "F0:F1:F2:F3:F4:F5"), await make_device(link, "0C:51:01:73:71:27")
    requests = serve_obex(phone, core.BT_PHONEBOOK_ACCESS_PSE_SERVICE, history)

    connection = await us.connect(phone.public_address, transport=PhysicalTransport.BR_EDR)
    calls = await pull_call_log(connection, limit=5)

    assert [(c.direction, c.number) for c in calls] == [("missed", "0491570157")]
    get = requests[1]
    assert get.header(obex.NAME) == "telecom/cch.vcf"
    assert get.header(obex.APP_PARAMETERS) == VCARD_30 + bytes([0x04, 0x02, 0x00, 0x05])  # MaxListCount = 5
