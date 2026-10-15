import pytest

from ladc.inference.llama_server import _extract_timings


def test_extract_server_timings():
    assert _extract_timings({"timings": {"prompt_ms": 12.5, "predicted_ms": 20.25}}) == (12.5, 20.25)


def test_extract_server_timings_rejects_missing_values():
    with pytest.raises(RuntimeError, match="did not contain"):
        _extract_timings({"timings": {"prompt_ms": 12.5}})
