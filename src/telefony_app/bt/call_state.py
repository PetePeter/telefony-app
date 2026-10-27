"""Call state derived from the phone's HFP indicators (+CIEV call / callsetup)."""

import enum
from collections.abc import Callable


class CallState(enum.Enum):
    IDLE = "idle"
    INCOMING = "incoming"
    DIALING = "dialing"
    ALERTING = "alerting"
    ACTIVE = "active"
    ENDED = "ended"


class CallOutcome(enum.Enum):
    COMPLETED = "completed"
    NOT_ANSWERED = "not_answered"


class CallEnded(Exception):
    """The call ended while waiting for a different state."""

    def __init__(self, outcome: CallOutcome | None) -> None:
        super().__init__(outcome)
        self.outcome = outcome


# HFP callsetup indicator values.
_SETUP_STATES = {1: CallState.INCOMING, 2: CallState.DIALING, 3: CallState.ALERTING}


class CallTracker:
    def __init__(self) -> None:
        self.state = CallState.IDLE
        self.outcome: CallOutcome | None = None
        self.on_change: Callable[[CallState], None] | None = None
        self._answered = False
        self.setup = 0  # raw callsetup indicator: a second call can be set up while one is active
        self.held = 0  # raw callheld indicator: 1 = active and held calls, 2 = held calls only
        self.setups_finished = 0  # times callsetup returned to 0; lets waiters spot a setup they never saw

    def begin_call(self) -> None:
        """Forget a finished call before placing a new one, so its ENDED state is not mistaken for this one's."""
        if self.state is CallState.ENDED:
            self.outcome = None
            self._set(CallState.IDLE)

    def on_indicator(self, name: str, value: int) -> None:
        if name == "callheld":
            self.held = value
        elif name == "callsetup":
            if value == 0 and self.setup != 0:
                self.setups_finished += 1
            self.setup = value
            if value in _SETUP_STATES:
                if self.state in (CallState.IDLE, CallState.ENDED):
                    self.outcome = None
                    self._answered = False
                if self.state is not CallState.ACTIVE:
                    self._set(_SETUP_STATES[value])
            elif value == 0 and self.state in _SETUP_STATES.values():
                self._end()
        elif name == "call":
            if value == 1:
                self._answered = True
                self._set(CallState.ACTIVE)
            elif value == 0 and self.state is CallState.ACTIVE:
                self._end()

    def _end(self) -> None:
        self.outcome = CallOutcome.COMPLETED if self._answered else CallOutcome.NOT_ANSWERED
        self._set(CallState.ENDED)

    def _set(self, state: CallState) -> None:
        if state is self.state:
            return
        self.state = state
        if self.on_change:
            self.on_change(state)
