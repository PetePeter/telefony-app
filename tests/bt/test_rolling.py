"""Rolling transcription: an utterance is transcribed in overlapping windows while the caller talks."""

import numpy as np

from telefony_app.bt.rolling import RollingTranscript, merge_overlap

RATE = 16000


def test_merge_drops_the_words_both_windows_heard():
    assert merge_overlap("please call me back after", "back after dinner tonight") == "please call me back after dinner tonight"


def test_merge_ignores_case_and_punctuation_in_the_overlap():
    assert merge_overlap("I need the ATO,", "the ato and connect me.") == "I need the ATO, and connect me."


def test_merge_appends_when_nothing_overlaps():
    assert merge_overlap("hello there", "general kenobi") == "hello there general kenobi"
    assert merge_overlap("", "first words") == "first words"


def test_windows_are_three_seconds_stepping_by_two_and_the_tail_overlaps_the_last():
    rolling = RollingTranscript(RATE, window_s=3, overlap_s=1)
    audio = np.arange(RATE * 8, dtype=np.int64)  # sample value == its index

    assert rolling.next_window(audio[: RATE * 2]) is None  # not 3 s yet
    first = rolling.next_window(audio[: RATE * 3])
    assert first is not None and (first[0], len(first)) == (0, RATE * 3)
    assert rolling.next_window(audio[: RATE * 4]) is None
    second = rolling.next_window(audio[: RATE * 5])
    assert second is not None and second[0] == RATE * 2

    tail = rolling.tail(audio[: RATE * 6])
    assert (tail[0], len(tail)) == (RATE * 4, RATE * 2)  # from the 1 s overlap to the end


def test_text_builds_up_window_by_window():
    rolling = RollingTranscript(RATE, window_s=3, overlap_s=1)
    rolling.add("could you call")
    rolling.add("call the dentist")
    rolling.add("dentist for me")
    assert rolling.text == "could you call the dentist for me"


def test_default_windows_hand_out_text_every_few_seconds():
    # Seen live: 8 s windows let a 28 s recorded message reach the agent only after the call was given up on.
    rolling = RollingTranscript(8000)
    assert rolling.next_window(np.zeros(8000 * 3 - 1, dtype=np.int16)) is None
    assert rolling.next_window(np.zeros(8000 * 3, dtype=np.int16)) is not None
    assert rolling.next_window(np.zeros(int(8000 * 5.5), dtype=np.int16)) is not None  # the next one, 2.5 s on
