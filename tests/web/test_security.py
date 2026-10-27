import pytest

from tests.web.conftest import ADMIN, Web

ROUTES = [("GET", "/api/config"), ("POST", "/api/config"), ("GET", "/api/status"), ("GET", "/api/events"),
          ("GET", "/api/phones"), ("POST", "/api/phones/forget")]  # fmt: skip


@pytest.mark.parametrize("method,path", ROUTES)
async def test_api_needs_the_admin_token(web: Web, method: str, path: str):
    assert (await web.http.request(method, path, {}, auth=False)).status == 401
    wrong = {"Authorization": "Bearer guess"}
    assert (await web.http.request(method, path, {}, auth=False, headers=wrong)).status == 401


async def test_login_sets_a_session_cookie(web: Web):
    assert (await web.http.request("POST", "/api/login", {"token": "guess"}, auth=False)).status == 401
    login = await web.http.request("POST", "/api/login", {"token": ADMIN}, auth=False)
    assert login.status == 200
    cookie = login.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    web.http.cookie = cookie.split(";")[0]
    assert (await web.http.request("GET", "/api/config", auth=False)).status == 200


async def test_the_page_itself_is_public(web: Web):
    page = await web.http.request("GET", "/", auth=False)
    assert page.status == 200 and page.headers["content-type"].startswith("text/html")


@pytest.mark.parametrize("headers", [{"X-Telefony": None}, {"Origin": "http://evil.example"}])
async def test_posts_need_the_csrf_header_and_our_origin(web: Web, headers: dict):
    reply = await web.http.request("POST", "/api/config", {"policy": {"country_code": "+44"}}, headers=headers)
    assert reply.status == 403


@pytest.mark.parametrize(
    "data,status",
    [
        (b"NONSENSE\r\n\r\n", 400),
        (b"POST /api/login HTTP/1.1\r\nContent-Length: 999999999\r\n\r\n", 413),
        (b"GET / HTTP/1.1\r\nX-Big: " + b"a" * 70000 + b"\r\n\r\n", 400),
    ],
    ids=["garbage", "huge-body", "huge-head"],
)
async def test_bad_requests_are_refused_and_the_server_carries_on(web: Web, data: bytes, status: int):
    assert (await web.http.raw(data)).status == status
    assert (await web.http.request("GET", "/api/config")).status == 200
