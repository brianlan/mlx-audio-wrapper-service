from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


ENDPOINTS = {
    "mlx": {
        "url": "http://127.0.0.1:8010/v1/audio/transcriptions",
        "model": "/Volumes/AigoP3500/models/lmstudio/models/mlx-community/Qwen3-ASR-0.6B-bf16",
        "language": "Chinese",
    },
    "qwen_cpp": {
        "url": "http://127.0.0.1:8011/v1/audio/transcriptions",
        "model": "/Volumes/AigoP3500/models/lmstudio/models/FlippyDora/qwen3-asr-0.6b-GGUF/qwen3-asr-0.6b-f16.gguf",
        "language": "chinese",
    },
}


@dataclass
class ReqResult:
    ok: bool
    status: int
    latency_s: float
    text_len: int
    error: str = ""


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def one_request(endpoint: dict[str, str], audio_path: str, timeout_s: int) -> ReqResult:
    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        endpoint["url"],
        "-F",
        f"model={endpoint['model']}",
        "-F",
        f"language={endpoint['language']}",
        "-F",
        f"file=@{audio_path}",
    ]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        elapsed = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        return ReqResult(ok=False, status=0, latency_s=timeout_s, text_len=0, error="timeout")

    if proc.returncode != 0:
        return ReqResult(
            ok=False,
            status=0,
            latency_s=elapsed,
            text_len=0,
            error=(proc.stderr or "curl failed")[-180:],
        )

    payload = proc.stdout.strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ReqResult(ok=False, status=200, latency_s=elapsed, text_len=0, error="bad json")

    text = data.get("text") if isinstance(data, dict) else None
    ok = isinstance(text, str) and len(text.strip()) > 0
    return ReqResult(
        ok=ok,
        status=200,
        latency_s=elapsed,
        text_len=len(text) if isinstance(text, str) else 0,
        error="" if ok else "empty text",
    )


def run_level(endpoint_name: str, endpoint: dict[str, str], audio_path: str, concurrency: int, requests: int, timeout_s: int) -> dict:
    results: list[ReqResult] = []
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(one_request, endpoint, audio_path, timeout_s) for _ in range(requests)]
        for fut in as_completed(futs):
            results.append(fut.result())

    wall = time.perf_counter() - start
    lats = sorted([r.latency_s for r in results if r.ok])
    success = sum(1 for r in results if r.ok)
    failures = len(results) - success
    err_rate = failures / len(results) if results else 1.0

    return {
        "endpoint": endpoint_name,
        "concurrency": concurrency,
        "requests": len(results),
        "success": success,
        "failures": failures,
        "error_rate": err_rate,
        "throughput_rps": (success / wall) if wall > 0 else 0.0,
        "latency_p50_s": percentile(lats, 50),
        "latency_p95_s": percentile(lats, 95),
        "latency_p99_s": percentile(lats, 99),
        "latency_mean_s": statistics.fmean(lats) if lats else math.nan,
        "wall_time_s": wall,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--requests-per-level", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out", default="/tmp/asr_bench_results.json")
    args = parser.parse_args()

    conc_levels = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]

    all_rows: list[dict] = []
    for name, ep in ENDPOINTS.items():
        for _ in range(args.warmup):
            _ = one_request(ep, args.audio, args.timeout)

        for c in conc_levels:
            row = run_level(
                endpoint_name=name,
                endpoint=ep,
                audio_path=args.audio,
                concurrency=c,
                requests=args.requests_per_level,
                timeout_s=args.timeout,
            )
            all_rows.append(row)
            print(json.dumps(row, ensure_ascii=True))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"rows": all_rows}, f, indent=2)


if __name__ == "__main__":
    main()
