"""PBAP pull end-to-end over Bumble's in-memory radio link, against a fake phonebook server (PSE)."""

from bumble import core
from bumble.core import PhysicalTransport
from bumble.link import LocalLink

from tests.bt.fake_phone import make_device, serve_obex
from telefony_app.bt import obex
from telefony_app.bt.pbap import PBAP_TARGET, VCARD_30, pull_contacts

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
