from telefony_app.bt.sco_stream import ScoOutQueue, iso_lengths

HANDLE = bytes([0x01, 0x09])  # connection handle 0x901, little-endian, as in the HCI SCO header


def sco(fill: int) -> bytes:
    return HANDLE + bytes([48]) + bytes([fill]) * 48


def silence() -> bytes:
    return HANDLE + bytes([48]) + bytes(48)


def test_take_returns_queued_packets_in_order():
    q = ScoOutQueue(max_packets=10)
    q.push(sco(1))
    q.push(sco(2))
    assert q.take(2) == sco(1) + sco(2)


def test_underrun_pads_with_silence_for_the_same_link():
    # The USB stream must never pause, so missing audio becomes well-formed silent packets.
    q = ScoOutQueue(max_packets=10)
    q.push(sco(7))
    assert q.take(3) == sco(7) + silence() + silence()


def test_nothing_to_send_before_the_first_packet():
    assert ScoOutQueue(max_packets=10).take(4) == b""


def test_overflow_drops_oldest_to_bound_latency():
    q = ScoOutQueue(max_packets=2)
    for i in (1, 2, 3):
        q.push(sco(i))
    assert q.take(2) == sco(2) + sco(3)


def test_iso_lengths_split_into_whole_usb_frames():
    assert iso_lengths(4 * 51, 17) == [17] * 12
    assert iso_lengths(51, 24) == [24, 24, 3]
