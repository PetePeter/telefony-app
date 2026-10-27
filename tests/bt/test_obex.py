import struct

from telefony_app.bt.obex import BODY, CONNECT, CONNECTION_ID, END_OF_BODY, GET, NAME, SUCCESS, TARGET, TYPE, Packet, decode, encode

PBAP_TARGET = bytes.fromhex("796135f0f0c511d809660800200c9a66")


def test_connect_request_matches_spec_layout():
    data = encode(Packet(CONNECT, [(TARGET, PBAP_TARGET)], struct.pack(">BBH", 0x10, 0, 0xFFFF)))
    assert data == bytes.fromhex("80001a" "1000ffff" "460013") + PBAP_TARGET


def test_connect_response_yields_connection_id():
    raw = bytes.fromhex("a0000c" "1000fc00" "cb00000001")
    packet = decode(raw, fields=4)
    assert packet.code == SUCCESS and packet.header(CONNECTION_ID) == 1


def test_get_headers_round_trip_with_unicode_name():
    packet = Packet(GET, [(CONNECTION_ID, 7), (NAME, "telecom/pb.vcf"), (TYPE, b"x-bt/phonebook\0")])
    data = encode(packet)
    assert data[data.index(NAME) : data.index(NAME) + 3] == bytes([NAME, 0, 3 + 2 * 15])
    assert decode(data) == packet


def test_empty_name_header_is_bare():
    assert encode(Packet(GET, [(NAME, "")])) == bytes.fromhex("830006" "010003")


def test_body_joins_body_and_end_of_body():
    assert Packet(SUCCESS, [(BODY, b"BEGIN"), (END_OF_BODY, b":VCARD")]).body() == b"BEGIN:VCARD"


def test_truncated_packet_is_rejected():
    import pytest

    from telefony_app.bt.obex import ObexError

    with pytest.raises(ObexError):
        decode(bytes.fromhex("a00010" "cb00000001"))
