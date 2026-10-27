from telefony_app.bt.call_state import CallOutcome, CallState, CallTracker


def feed(tracker: CallTracker, *updates: tuple[str, int]) -> CallTracker:
    for name, value in updates:
        tracker.on_indicator(name, value)
    return tracker


def test_outgoing_call_answered_then_ended():
    t = CallTracker()
    states = []
    t.on_change = states.append
    feed(t, ("callsetup", 2), ("callsetup", 3), ("call", 1), ("callsetup", 0), ("call", 0))
    assert states == [CallState.DIALING, CallState.ALERTING, CallState.ACTIVE, CallState.ENDED]
    assert t.outcome is CallOutcome.COMPLETED


def test_remote_reject_while_alerting_is_not_answered():
    t = feed(CallTracker(), ("callsetup", 2), ("callsetup", 3), ("callsetup", 0))
    assert t.state is CallState.ENDED
    assert t.outcome is CallOutcome.NOT_ANSWERED


def test_busy_before_alerting_is_not_answered():
    t = feed(CallTracker(), ("callsetup", 2), ("callsetup", 0))
    assert t.outcome is CallOutcome.NOT_ANSWERED


def test_incoming_ring_then_answer():
    t = feed(CallTracker(), ("callsetup", 1))
    assert t.state is CallState.INCOMING
    feed(t, ("call", 1), ("callsetup", 0))
    assert t.state is CallState.ACTIVE


def test_new_call_after_ended_resets_outcome():
    t = feed(CallTracker(), ("callsetup", 2), ("callsetup", 0))
    feed(t, ("callsetup", 2))
    assert t.state is CallState.DIALING
    assert t.outcome is None


def test_unrelated_indicators_ignored():
    t = feed(CallTracker(), ("signal", 4), ("battchg", 2), ("callheld", 0))
    assert t.state is CallState.IDLE


def test_zero_indicators_while_idle_stay_idle():
    # Phones report call=0/callsetup=0 right after connecting; that is not a call ending.
    t = feed(CallTracker(), ("callsetup", 0), ("call", 0))
    assert t.state is CallState.IDLE


def test_consult_call_while_first_is_held_keeps_call_active():
    t = feed(CallTracker(), ("call", 1), ("callheld", 2), ("callsetup", 2), ("callsetup", 3), ("callheld", 1), ("callsetup", 0))
    assert t.state is CallState.ACTIVE
    assert t.held == 1  # one active, one held


def test_failed_consult_call_leaves_held_call_and_does_not_end():
    t = feed(CallTracker(), ("call", 1), ("callheld", 2), ("callsetup", 2), ("callsetup", 0))
    assert t.state is CallState.ACTIVE
    assert t.held == 2
    assert t.setup == 0


def test_hold_and_resume_never_end_the_call():
    t = CallTracker()
    states = []
    t.on_change = states.append
    feed(t, ("call", 1), ("callheld", 2), ("callheld", 0))
    assert states == [CallState.ACTIVE]
    assert t.held == 0
