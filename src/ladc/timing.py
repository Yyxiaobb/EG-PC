"""Small dependency-free timing utilities for runtime measurements."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter_ns
from typing import Iterator


def now_ns() -> int:
    """Return a monotonic timestamp suitable for elapsed-time measurement."""

    return perf_counter_ns()


def elapsed_ms(start_ns: int, end_ns: int | None = None) -> float:
    """Convert a monotonic interval to milliseconds."""

    finish = now_ns() if end_ns is None else end_ns
    return (finish - start_ns) / 1_000_000.0


@contextmanager
def measure_ms() -> Iterator[list[float]]:
    """Capture one elapsed interval without hiding the measured value."""

    start = now_ns()
    result: list[float] = []
    try:
        yield result
    finally:
        result.append(elapsed_ms(start))
