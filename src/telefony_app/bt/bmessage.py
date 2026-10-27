"""MAP message formats: the bMessage envelope (one SMS) and the XML messages listing. Message text is untrusted data."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass
class Message:
    sender: str  # number or name as the phone reports it
    text: str


@dataclass
class ListedMessage:
    handle: str
    sender: str
    subject: str
    datetime: str  # phone-local, e.g. 20261026T102233
    read: bool


def build_sms(recipient: str, text: str) -> bytes:
    """A bMessage for PushMessage. LENGTH counts the BEGIN:MSG..END:MSG block in bytes."""
    content = f"BEGIN:MSG\r\n{text}\r\nEND:MSG\r\n"
    return (
        "BEGIN:BMSG\r\nVERSION:1.0\r\nSTATUS:READ\r\nTYPE:SMS_GSM\r\nFOLDER:telecom/msg/outbox\r\n"
        "BEGIN:BENV\r\n"
        f"BEGIN:VCARD\r\nVERSION:2.1\r\nN:\r\nTEL:{recipient}\r\nEND:VCARD\r\n"
        f"BEGIN:BBODY\r\nCHARSET:UTF-8\r\nLENGTH:{len(content.encode())}\r\n{content}"
        "END:BBODY\r\nEND:BENV\r\nEND:BMSG\r\n"
    ).encode()


def parse_message(data: bytes) -> Message:
    """Sender from the originator vCard (before BENV), text between BEGIN:MSG and END:MSG."""
    text = data.decode("utf-8", errors="replace")
    head, _, rest = text.partition("BEGIN:BENV")
    sender = ""
    for line in head.splitlines():
        key, _, value = line.partition(":")
        prop = key.split(";")[0].upper()
        if prop == "TEL" and value:
            sender = value.strip()
        elif prop in ("FN", "N") and value.strip(";") and not sender:
            sender = value.strip()
    _, _, body = rest.partition("BEGIN:MSG\r\n")
    body, _, _ = body.rpartition("\r\nEND:MSG")
    return Message(sender, body)


def parse_listing(data: bytes) -> list[ListedMessage]:
    root = ET.fromstring(data.decode("utf-8", errors="replace").strip())
    return [
        ListedMessage(
            handle=m.get("handle", ""),
            sender=m.get("sender_addressing") or m.get("sender_name") or "",
            subject=m.get("subject", ""),
            datetime=m.get("datetime", ""),
            read=m.get("read", "no").lower() == "yes",
        )
        for m in root.iter("msg")
    ]
