"""A short, user-confirmed Bluetooth pairing window on telefony's Bumble device."""

import asyncio
import logging
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal
from weakref import WeakSet

from bumble import pairing
from bumble.device import Connection, Device

log = logging.getLogger(__name__)

PairingState = Literal[
    "discoverable",
    "awaiting-confirmation",
    "pairing",
    "paired",
    "rejected",
    "cancelled",
    "expired",
    "failed",
]
_ACTIVE_STATES = {"discoverable", "awaiting-confirmation", "pairing"}


class PairingError(Exception):
    """An expected pairing API failure with its HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class _Session:
    pairing_id: str
    state: PairingState
    expires_at: str
    comparison: asyncio.Future[bool] | None = None
    address: str | None = None
    name: str | None = None
    numeric_code: str | None = None
    connection: Connection | None = None
    expiry_task: asyncio.Task[None] | None = None


class _SetupDelegate(pairing.PairingDelegate):
    def __init__(self, controller: "PairingController", connection: Connection) -> None:
        super().__init__(pairing.PairingDelegate.DISPLAY_OUTPUT_AND_YES_NO_INPUT)
        self.controller = controller
        self.connection = connection

    async def accept(self) -> bool:
        return self.controller._can_pair(self.connection)

    async def confirm(self, auto: bool = False) -> bool:
        del auto
        return False

    async def compare_numbers(self, number: int, digits: int) -> bool:
        return await self.controller._compare(self.connection, number, digits)

    async def get_number(self) -> int | None:
        return None

    async def get_string(self, max_length: int) -> str | None:
        del max_length
        return None


class PairingController:
    """Own one discoverability window and its Numeric Comparison decision."""

    def __init__(self, device: Device, device_name: str = "telefony") -> None:
        self.device = device
        self.device_name = device_name
        self._ready = False
        self._session: _Session | None = None
        self._lock = asyncio.Lock()
        self._watched: WeakSet[Connection] = WeakSet()
        self.device.pairing_config_factory = self._pairing_config

    def mark_ready(self) -> None:
        """Allow API requests after the existing adapter has powered on."""
        self._ready = True

    def status(self) -> dict[str, object]:
        self._require_ready()
        session = self._session
        if session is None:
            return {"state": "idle", "deviceName": self.device_name}
        result: dict[str, object] = {
            "pairingId": session.pairing_id,
            "state": session.state,
            "deviceName": self.device_name,
            "expiresAt": session.expires_at,
        }
        if session.state == "awaiting-confirmation":
            result["phone"] = self._phone(session)
            result["numericCode"] = session.numeric_code
        elif session.state in {"pairing", "paired"}:
            result["phone"] = self._phone(session)
        return result

    async def start(self, timeout_seconds: int = 120) -> dict[str, object]:
        if type(timeout_seconds) is not int or not 30 <= timeout_seconds <= 300:
            raise PairingError(400, "timeoutSeconds must be an integer from 30 to 300")
        async with self._lock:
            self._require_ready()
            if self._session is not None and self._session.state in _ACTIVE_STATES:
                raise PairingError(409, "a pairing window is already active")
            expires = datetime.now(UTC) + timedelta(seconds=timeout_seconds)
            session = _Session(
                pairing_id=secrets.token_urlsafe(24),
                state="discoverable",
                expires_at=expires.isoformat(timespec="seconds").replace("+00:00", "Z"),
            )
            self._session = session
            try:
                await self.device.set_discoverable(True)
            except Exception as error:
                session.state = "failed"
                log.exception("could not enable Bluetooth discoverability")
                raise PairingError(409, "Bluetooth adapter unavailable") from error
            session.expiry_task = asyncio.create_task(self._expire(session, timeout_seconds))
            return {
                "pairingId": session.pairing_id,
                "state": session.state,
                "deviceName": self.device_name,
                "expiresAt": session.expires_at,
            }

    async def confirm(
        self, pairing_id: str, address: str, numeric_code: str, accept: bool
    ) -> dict[str, object]:
        async with self._lock:
            self._require_ready()
            session = self._find(pairing_id)
            if session.state != "awaiting-confirmation" or session.comparison is None:
                raise PairingError(404, "no pending confirmation")
            if address.upper() != session.address or numeric_code != session.numeric_code:
                raise PairingError(409, "confirmation does not match the pending phone and code")
            comparison = session.comparison
            if accept:
                if comparison.done():
                    raise PairingError(409, "confirmation is no longer pending")
                session.state = "pairing"
                comparison.set_result(True)
                return {"pairingId": session.pairing_id, "state": session.state}
            else:
                if not comparison.done():
                    comparison.set_result(False)
                await self._finish_locked(session, "rejected")
                return {"pairingId": session.pairing_id, "state": session.state}

                return {"pairingId": session.pairing_id, "state": session.state}

    async def cancel(self, pairing_id: str) -> dict[str, object]:
        async with self._lock:
            self._require_ready()
            session = self._find(pairing_id)
            if session.state not in _ACTIVE_STATES:
                raise PairingError(409, "pairing window is no longer active")
            if session.comparison is not None and not session.comparison.done():
                session.comparison.set_result(False)
            await self._finish_locked(session, "cancelled")
            return {"pairingId": pairing_id, "state": session.state}

    async def close(self) -> None:
        """Stop any active window before the Bumble transport shuts down."""
        async with self._lock:
            self._ready = False
            session = self._session
            if session is not None and session.state in _ACTIVE_STATES:
                try:
                    await self._finish_locked(session, "cancelled")
                except PairingError:
                    pass
                if session.comparison is not None and not session.comparison.done():
                    session.comparison.set_result(False)

    def _pairing_config(self, connection: Connection) -> pairing.PairingConfig:
        if connection not in self._watched:
            self._watched.add(connection)
            connection.on(Connection.EVENT_CLASSIC_PAIRING, lambda *_: self._schedule_result(connection, "paired"))
            connection.on(Connection.EVENT_CLASSIC_PAIRING_FAILURE, lambda *_: self._schedule_result(connection, "failed"))
            connection.on(Connection.EVENT_PAIRING, lambda *_: self._schedule_result(connection, "paired"))
            connection.on(Connection.EVENT_PAIRING_FAILURE, lambda *_: self._schedule_result(connection, "failed"))
        return pairing.PairingConfig(mitm=True, bonding=True, delegate=_SetupDelegate(self, connection))

    def _can_pair(self, connection: Connection) -> bool:
        session = self._session
        return bool(
            self._ready
            and session is not None
            and session.state in _ACTIVE_STATES
            and (session.connection is None or session.connection is connection)
        )

    async def _compare(self, connection: Connection, number: int, digits: int) -> bool:
        if digits != 6 or not 0 <= number <= 999999:
            return False
        async with self._lock:
            session = self._session
            if not self._can_pair(connection) or session is None or session.state != "discoverable":
                return False
            session.connection = connection
            session.address = _address(connection)
            session.name = connection.peer_name
            session.numeric_code = f"{number:06d}"
            session.comparison = asyncio.get_running_loop().create_future()
            session.state = "awaiting-confirmation"
            comparison = session.comparison
        try:
            return await comparison
        except asyncio.CancelledError:
            await self._complete(connection, "failed")
            raise

    async def _expire(self, session: _Session, timeout_seconds: int) -> None:
        try:
            await asyncio.sleep(timeout_seconds)
            async with self._lock:
                if session.state in _ACTIVE_STATES:
                    if session.comparison is not None and not session.comparison.done():
                        session.comparison.set_result(False)
                    await self._finish_locked(session, "expired")
        except asyncio.CancelledError:
            raise
        except PairingError:
            log.exception("could not expire the Bluetooth pairing window")

    def _schedule_result(self, connection: Connection, state: Literal["paired", "failed"]) -> None:
        asyncio.create_task(self._complete(connection, state))

    async def _complete(self, connection: Connection, state: Literal["paired", "failed"]) -> None:
        async with self._lock:
            session = self._session
            if session is None or session.connection is not connection or session.state not in _ACTIVE_STATES:
                return
            if state == "paired" and (
                session.comparison is None
                or not session.comparison.done()
                or session.comparison.cancelled()
                or not session.comparison.result()
            ):
                return
            if session.comparison is not None and not session.comparison.done() and state == "failed":
                session.comparison.set_result(False)
            try:
                await self._finish_locked(session, state)
            except PairingError:
                pass

    async def _finish_locked(self, session: _Session, state: PairingState) -> None:
        session.state = state
        current = asyncio.current_task()
        if session.expiry_task is not None and session.expiry_task is not current:
            session.expiry_task.cancel()
        try:
            await self.device.set_discoverable(False)
        except Exception as error:
            session.state = "failed"
            log.exception("could not disable Bluetooth discoverability")
            raise PairingError(409, "Bluetooth adapter unavailable") from error

    def _find(self, pairing_id: str) -> _Session:
        session = self._session
        if session is None or not secrets.compare_digest(session.pairing_id, pairing_id):
            raise PairingError(404, "pairing session not found")
        return session

    def _require_ready(self) -> None:
        if not self._ready:
            raise PairingError(409, "Bluetooth adapter unavailable")

    @staticmethod
    def _phone(session: _Session) -> dict[str, str | None]:
        return {"address": session.address or "", "name": session.name}


def _address(connection: Connection) -> str:
    address = connection.peer_address
    value = address.to_string(with_type_qualifier=False) if hasattr(address, "to_string") else str(address)
    return value.upper()
