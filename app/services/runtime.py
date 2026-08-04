from pathlib import Path


def _read_proc_memory_mb(field: str) -> float | None:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{field}:"):
            value_kb = int(line.split()[1])
            return round(value_kb / 1024, 1)
    return None


def memory_snapshot() -> dict[str, float | None]:
    """Return current and peak Linux process RSS values when available."""
    return {
        "current_rss_mb": _read_proc_memory_mb("VmRSS"),
        "peak_rss_mb": _read_proc_memory_mb("VmHWM"),
    }
