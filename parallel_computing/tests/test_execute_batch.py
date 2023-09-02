# test_execute_batch.py
# pytest suite for execute_batch

import itertools
import types
import time
import pytest
import smart_import_copy
from smart_import_copy import smart_import_guard
with smart_import_guard():
    from ..execute_batch import execute_batch


# ─────────────────────────── helpers ────────────────────────────
def make_flaky(
    *, good_after: int = 1, raise_error: bool = True, bad_return: bool = False
):
    """
    Produce a function that misbehaves `good_after-1` times, then succeeds.

    Options
    -------
    raise_error   raise ValueError on the bad calls
    bad_return    instead return -1 (so check_fn fails)
    Exactly one of raise_error / bad_return should be True.
    """
    counter = itertools.count()

    def fn():
        attempt = next(counter)
        if attempt < good_after - 1:
            if raise_error:
                raise ValueError("boom")
            if bad_return:
                return -1
        return 42

    return fn


def is_positive(x):
    return x > 0


# ──────────────────────────── tests ─────────────────────────────
def test_success_ordering_and_fields():
    fn = lambda a: a * 2
    inputs = [{"a": 1}, {"a": 3}, {"a": 5}]
    out = execute_batch(
        fn,
        inputs=inputs,
        check_function=is_positive,
        retries=1,
        batch_retries=0,
        max_workers=2,
    )

    # ordering preserved
    assert [item["input"] for item in out] == inputs

    # correct results & flags
    assert [item["result"] for item in out] == [2, 6, 10]
    assert all(item["success"] for item in out)

    # required keys exist
    required = {
        "id",
        "input",
        "result",
        "success",
        "function_latency",
        "total_latency",
        "retries",
        "error_count",
        "failure_count",
        "batch_round",
    }
    assert all(required.issubset(item.keys()) for item in out)


@pytest.mark.parametrize("bad_mode", ["error", "bad_output"])
def test_retry_until_success(monkeypatch, bad_mode):
    flaky = make_flaky(
        good_after=3,
        raise_error=(bad_mode == "error"),
        bad_return=(bad_mode == "bad_output"),
    )

    start = time.perf_counter()
    res = execute_batch(
        flaky,
        inputs=[{}],
        check_function=is_positive,
        retries=5,
        batch_retries=0,
        max_workers=1,
        retry_delay=0.01,  # speed up test
        jitter=0.0,
    )[0]
    elapsed = time.perf_counter() - start

    assert res["success"] is True
    assert res["retries"] == 2
    if bad_mode == "error":
        assert res["error_count"] == 2 and res["failure_count"] == 0
    else:
        assert res["failure_count"] == 2 and res["error_count"] == 0
    # total latency includes back-off (basic sanity check)
    assert res["total_latency"] <= elapsed + 0.01


def test_exhausted_retries_failure():
    flaky = make_flaky(good_after=10)  # never recovers within retry budget
    res = execute_batch(
        flaky,
        inputs=[{}],
        check_function=is_positive,
        retries=3,
        batch_retries=0,
        max_workers=1,
        retry_delay=None,
        jitter=None,
    )[0]

    assert res["success"] is False
    assert res["result"] is None
    assert res["retries"] == 3
    assert res["error_count"] == 3


def test_batch_retry_recovers():
    """
    First batch fails; we patch the function before second batch so it succeeds.
    """
    calls = {"attempt": 0}

    def fn():
        calls["attempt"] += 1
        # fail first two global calls, succeed afterwards
        if calls["attempt"] <= 2:
            raise ValueError("boom")
        return 1

    res_list = execute_batch(
        fn,
        inputs=[{}, {}],
        check_function=is_positive,
        retries=1,
        batch_retries=2,
        max_workers=2,
        retry_delay=0.01,
        jitter=0.0,
    )

    # both inputs eventually succeed
    assert all(r["success"] for r in res_list)
    # at least one required batch_round > 0 (failure in first batch)
    assert any(r["batch_round"] > 0 for r in res_list)


def test_backoff_and_jitter(monkeypatch):
    """Verify sleep is called with increasing delay."""
    sleep_calls = []

    def fake_sleep(d):
        sleep_calls.append(d)

    monkeypatch.setattr(time, "sleep", fake_sleep)

    flaky = make_flaky(good_after=3)  # will fail twice
    execute_batch(
        flaky,
        inputs=[{}],
        check_function=is_positive,
        retries=3,
        batch_retries=0,
        max_workers=1,
        retry_delay=0.05,
        jitter=0.0,
    )

    # sleep called exactly retries-1 times with increasing durations
    assert sleep_calls == sorted(sleep_calls)
    assert len(sleep_calls) == 2
    assert sleep_calls[1] > sleep_calls[0]
