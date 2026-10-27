from bumble.keys import JsonKeyStore, PairingKeys

from tests.conftest import ALLOWED
from tests.web.conftest import Web

PHONE = "AA:BB:CC:DD:EE:01/P"


async def test_status_waits_for_the_phone_then_reports_the_service(web: Web):
    assert (await web.http.request("GET", "/api/status")).json() == {"phone": "waiting"}
    web.connect_phone()
    status = (await web.http.request("GET", "/api/status")).json()
    assert status == {"phone": "connected", "state": {"link": "up", "engine": "ready", "calls": []}}


async def test_events_stream_live(web: Web):
    service = web.connect_phone()
    stream = web.http.events()
    assert (await anext(stream))["type"] == "snapshot"  # subscribed
    await service.dial(ALLOWED)
    event = await anext(stream)
    assert (event["type"], event["state"]) == ("call.state", "dialing")
    await stream.aclose()


async def test_bonded_phones_are_listed_and_can_be_forgotten(web: Web, keystore: JsonKeyStore):
    await keystore.update(PHONE, PairingKeys(link_key=PairingKeys.Key(bytes(16))))
    assert (await web.http.request("GET", "/api/phones")).json() == {"phones": [PHONE]}
    assert (await web.http.request("POST", "/api/phones/forget", {"address": PHONE})).status == 200
    assert (await web.http.request("GET", "/api/phones")).json() == {"phones": []}
    assert (await web.http.request("POST", "/api/phones/forget", {"address": PHONE})).status == 404
