"""Reproducible benchmark for the trade journal's flush + fsync cost.

The journal reports success only after ``handle.flush()`` and
``os.fsync(handle.fileno())`` (see ``journal/journal.py``). This script
measures what that durability guarantee costs, using typical small trade
records and a temporary directory. It never touches the real journal and
is not imported by any production module.

Three append paths write byte-identical records:

  raw         json.dumps + open("a") + write + close. Closing flushes the
              process buffer into the kernel; nothing is forced to disk.
  flush_only  the same, plus an explicit ``handle.flush()``. Separates
              Python buffer cost from device cost.
  production  ``journal.journal.append_entry`` exactly as the bot calls
              it, including its parent-directory and rotation checks.

Usage:
    python benchmarks/bench_journal_fsync.py
    python benchmarks/bench_journal_fsync.py --iterations 500 --rounds 5
    python benchmarks/bench_journal_fsync.py --dir /path/to/scratch

Latency is reported in milliseconds per append as median, p95 and maximum.
Use the same filesystem as ``logs/`` for a representative result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from journal import journal as journal_module  # noqa: E402

SYMBOLS = ("BTC", "ETH", "SOL", "XRP")
DECISIONS = ("BUY", "HOLD", "SELL")
VERDICTS = ("APPROVED", "REJECTED")

#: Appends discarded per mode before timing, so the first-call costs
#: (imports, page cache fills) do not skew the distribution.
WARMUP = 20


def _record(index: int) -> dict:
    """Build a typical small trade-journal record."""
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": SYMBOLS[index % len(SYMBOLS)],
        "decision": DECISIONS[index % len(DECISIONS)],
        "risk_verdict": VERDICTS[index % len(VERDICTS)],
    }


def _append_raw(path: Path, index: int) -> None:
    line = json.dumps(_record(index), ensure_ascii=False, allow_nan=False) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)


def _append_flush_only(path: Path, index: int) -> None:
    line = json.dumps(_record(index), ensure_ascii=False, allow_nan=False) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def _append_production(path: Path, index: int) -> bool:
    return journal_module.append_entry(
        _record(index),
        path=path,
        max_bytes=journal_module.DEFAULT_MAX_BYTES,
        backup_count=journal_module.DEFAULT_BACKUP_COUNT,
    )


MODES = (
    ("raw", _append_raw),
    ("flush_only", _append_flush_only),
    ("production", _append_production),
)


def _percentile(sorted_ms: list[float], fraction: float) -> float:
    """Nearest-rank percentile on an already sorted list."""
    if not sorted_ms:
        return math.nan
    rank = max(1, math.ceil(fraction * len(sorted_ms)))
    return sorted_ms[min(rank, len(sorted_ms)) - 1]


def _summarise(latencies_ms: list[float]) -> dict:
    ordered = sorted(latencies_ms)
    return {
        "n": len(ordered),
        "median": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1] if ordered else math.nan,
        "mean": sum(ordered) / len(ordered) if ordered else math.nan,
        "min": ordered[0] if ordered else math.nan,
    }


def _time_mode(fn, path: Path, iterations: int) -> list[float]:
    latencies: list[float] = []
    for index in range(WARMUP, WARMUP + iterations):
        start = time.perf_counter_ns()
        result = fn(path, index)
        elapsed = time.perf_counter_ns() - start
        if result is False:
            raise RuntimeError(f"{fn.__name__} reported a failed append")
        latencies.append(elapsed / 1_000_000.0)
    return latencies


def run(iterations: int, rounds: int, scratch: Path) -> dict:
    scratch.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {
        name: {"rounds": [], "pooled": []} for name, _ in MODES
    }

    for round_index in range(rounds):
        for name, fn in MODES:
            # A fresh file per mode and round, so the file grows exactly as
            # it would in place and no mode inherits another's cache state.
            path = scratch / f"journal_{name}_{round_index}.jsonl"
            latencies = _time_mode(fn, path, iterations)
            results[name]["rounds"].append(_summarise(latencies))
            results[name]["pooled"].extend(latencies)

    return {
        "iterations": iterations,
        "rounds": rounds,
        "scratch": str(scratch),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "modes": {
            name: {
                "rounds": data["rounds"],
                "pooled": _summarise(data["pooled"]),
            }
            for name, data in results.items()
        },
    }


def _print_report(report: dict) -> None:
    print(f"python     : {report['python']}")
    print(f"platform   : {report['platform']}")
    print(f"scratch    : {report['scratch']}")
    print(
        f"iterations : {report['iterations']} per mode per round, "
        f"{report['rounds']} round(s), {WARMUP} warmup appends discarded"
    )
    print()

    header = (
        f"{'mode':<12}{'round':>6}{'median ms':>12}{'p95 ms':>10}"
        f"{'max ms':>10}{'mean ms':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, data in report["modes"].items():
        for index, summary in enumerate(data["rounds"], start=1):
            print(
                f"{name:<12}{index:>6}{summary['median']:>12.4f}"
                f"{summary['p95']:>10.4f}{summary['max']:>10.4f}"
                f"{summary['mean']:>10.4f}"
            )
        pooled = data["pooled"]
        print(
            f"{name:<12}{'all':>6}{pooled['median']:>12.4f}"
            f"{pooled['p95']:>10.4f}{pooled['max']:>10.4f}"
            f"{pooled['mean']:>10.4f}"
        )
        print()

    raw = report["modes"]["raw"]["pooled"]
    production = report["modes"]["production"]["pooled"]
    flush_only = report["modes"]["flush_only"]["pooled"]
    print("durability cost (pooled medians):")
    print(
        f"  production - raw        : "
        f"{production['median'] - raw['median']:+.4f} ms per append"
    )
    print(
        f"  production - flush_only : "
        f"{production['median'] - flush_only['median']:+.4f} ms per append"
    )
    print(
        f"  production / raw        : "
        f"{production['median'] / raw['median']:.2f}x median latency"
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--iterations", type=int, default=200, help="timed appends per mode"
    )
    parser.add_argument(
        "--rounds", type=int, default=3, help="repetitions of each measurement"
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="scratch directory (created and left in place); "
        "defaults to a fresh temporary directory that is removed afterwards",
    )
    parser.add_argument(
        "--json", action="store_true", help="also print the raw report as JSON"
    )
    args = parser.parse_args()

    if args.iterations < 100:
        parser.error("--iterations must be at least 100")
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")

    owned = args.dir is None
    scratch = Path(
        args.dir
        if not owned
        else tempfile.mkdtemp(prefix="journal-fsync-bench-")
    )
    if owned:
        print(f"using temporary directory {scratch} (removed afterwards)")

    try:
        report = run(args.iterations, args.rounds, scratch)
    finally:
        if owned:
            shutil.rmtree(scratch, ignore_errors=True)

    _print_report(report)
    if args.json:
        print()
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
