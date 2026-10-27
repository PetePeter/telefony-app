"""MAP list/read/send end-to-end over Bumble's in-memory radio link, against a fake Message Access Server."""

from bumble import core
from bumble.core import PhysicalTransport
from bumble.link import LocalLink

from tests.bt.fake_phone import make_device, serve_obex
from tests.bt.test_bmessage import IOS_SMS, LISTING
from telefony_app.bt import obex
from telefony_app.bt.bmessage import parse_message
from telefony_app.bt.map_client import MAS_TARGET, make_mns_sdp_records, map_session


async def phone_and_link(handler):
    link = LocalLink()
    us, phone = await make_device(link, "F0:F1:F2:F3:F4:F5"), await make_device(link, "0C:51:01:73:71:27")
    requests = serve_obex(phone, core.BT_MESSAGE_ACCESS_SERVER_SERVICE, handler)
    return await us.connect(phone.public_address, transport=PhysicalTransport.BR_EDR), requests


def mas(request: obex.Packet) -> obex.Packet:
    if request.code == obex.SETPATH:
        return obex.Packet(obex.SUCCESS)
    if request.code == obex.GET:
        body = LISTING if request.header(obex.TYPE) == b"x-bt/MAP-msg-listing\0" else IOS_SMS
        return obex.Packet(obex.SUCCESS, [(obex.END_OF_BODY, body)])
    if request.code == obex.PUT_MORE:
        return obex.Packet(obex.CONTINUE)
    return obex.Packet(obex.SUCCESS, [(obex.NAME, "0400000000000099")])  # final PUT


async def test_lists_and_reads_inbox_from_msg_folder():
    connection, requests = await phone_and_link(mas)
    async with map_session(connection) as session:
        listing = await session.list_messages()
        message = await session.read_message(listing[0].handle)

    assert requests[0].header(obex.TARGET) == MAS_TARGET
    assert [r.header(obex.NAME) for r in requests if r.code == obex.SETPATH] == ["telecom", "msg"]
    get_listing, get_message = [r for r in requests if r.code == obex.GET]
    assert get_listing.header(obex.NAME) == "inbox"
    assert get_message.header(obex.NAME) == "0400000000000001"
    assert message.sender == "+61491570313" and message.text.startswith("Running late")


async def test_long_sms_is_split_across_put_packets_and_returns_handle():
    connection, requests = await phone_and_link(mas)
    text = "word " * 600  # 3000 bytes > the phone's 1 KiB packets
    async with map_session(connection) as session:
        handle = await session.send_sms("0491570110", text)

    puts = [r for r in requests if r.code in (obex.PUT, obex.PUT_MORE)]
    assert handle == "0400000000000099"
    assert len(puts) >= 3 and puts[-1].code == obex.PUT and puts[0].header(obex.NAME) == "outbox"
    assert all(len(obex.encode(p)) <= 0x400 for p in puts)
    sent = parse_message(b"".join(p.body() for p in puts))
    assert sent.text == text


async def test_phone_finds_our_notification_server_by_sdp():
    """iOS only offers "Show Notifications" (and allows MAP) when the car kit advertises an MNS record."""
    link = LocalLink()
    us, phone = await make_device(link, "F0:F1:F2:F3:F4:F5"), await make_device(link, "0C:51:01:73:71:27")
    us.sdp_service_records = {0x00010002: make_mns_sdp_records(0x00010002, 7)}
    connection = await phone.connect(us.public_address, transport=PhysicalTransport.BR_EDR)
    assert await obex.find_rfcomm_channel(connection, core.BT_MESSAGE_NOTIFICATION_SERVER_SERVICE) == 7
