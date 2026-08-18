"""Tests for optional process-memory telemetry."""

from pathlib import Path
from unittest.mock import patch

from app.services.runtime import _read_proc_memory_mb, memory_snapshot


def test_read_proc_memory_converts_kilobytes_to_megabytes() -> None:
    with (
        patch.object(Path, "exists", return_value=True),
        patch.object(
            Path,
            "read_text",
            return_value="VmRSS:\t1536 kB\nVmHWM:\t2048 kB\n",
        ),
    ):
        assert _read_proc_memory_mb("VmRSS") == 1.5


def test_read_proc_memory_returns_none_for_malformed_value() -> None:
    with (
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "read_text", return_value="VmRSS:\tunknown kB\n"),
    ):
        assert _read_proc_memory_mb("VmRSS") is None


def test_read_proc_memory_returns_none_when_status_cannot_be_read() -> None:
    with (
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "read_text", side_effect=OSError("permission denied")),
    ):
        assert _read_proc_memory_mb("VmRSS") is None


def test_memory_snapshot_reports_both_supported_fields() -> None:
    with patch(
        "app.services.runtime._read_proc_memory_mb",
        side_effect=[12.5, 20.0],
    ) as read_memory:
        snapshot = memory_snapshot()

    assert snapshot == {"current_rss_mb": 12.5, "peak_rss_mb": 20.0}
    assert [call.args[0] for call in read_memory.call_args_list] == ["VmRSS", "VmHWM"]
