"""Smoke-test the deployed API response contract and memory headroom."""

import argparse
import json
import time
from pathlib import Path

import httpx

FREE_RENDER_MEMORY_MB = 512.0


def parse_args() -> argparse.Namespace:
    """Parse the deployment URL, smoke image, timeout, and memory gate."""
    parser = argparse.ArgumentParser(description="Smoke-test a deployed API.")
    parser.add_argument("base_url")
    parser.add_argument("image", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--memory-limit-mb",
        type=float,
        default=FREE_RENDER_MEMORY_MB,
    )
    return parser.parse_args()


def main() -> None:
    """Verify deployed health, backend identity, inference, and peak RSS."""
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    with httpx.Client(timeout=args.timeout) as client:
        health = client.get(f"{base_url}/health")
        health.raise_for_status()
        version = client.get(f"{base_url}/version")
        version.raise_for_status()
        started = time.perf_counter()
        with args.image.open("rb") as image_file:
            inference = client.post(
                f"{base_url}/v1/box-intake/infer",
                files={"file": (args.image.name, image_file, "image/jpeg")},
            )
        elapsed_seconds = time.perf_counter() - started
        inference.raise_for_status()

    body = inference.json()
    peak_rss_mb = body["runtime_memory"]["peak_rss_mb"]
    report = {
        "health": health.json(),
        "version": version.json(),
        "inference": {
            "status_code": inference.status_code,
            "request_id": inference.headers.get("x-request-id"),
            "visible_box_count": body["visible_box_count"],
            "review_reasons": body["review_reasons"],
            "client_elapsed_seconds": round(elapsed_seconds, 3),
            "server_processing_time_ms": body["processing_time_ms"],
            "runtime_memory": body["runtime_memory"],
        },
    }
    print(json.dumps(report, indent=2))
    if version.json()["model_backend"] != "onnx":
        raise SystemExit("Render smoke failed: deployed backend is not ONNX.")
    if peak_rss_mb is None:
        raise SystemExit("Render smoke failed: peak RSS telemetry is unavailable.")
    if peak_rss_mb >= args.memory_limit_mb:
        raise SystemExit(
            "Render smoke failed: peak RSS "
            f"{peak_rss_mb} MB exceeds the free-instance gate."
        )


if __name__ == "__main__":
    main()
