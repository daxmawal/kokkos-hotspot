#!/usr/bin/env python3
import argparse
import csv
import statistics
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize Kokkos kernel timings and plot hotspots."
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="One or more timing CSV files (from KOKKOS_TIMING_OUT)",
    )
    parser.add_argument(
        "--out",
        dest="plot_path",
        help="Output plot path (default: <csv_dir>/kernel_hotspots.png)",
    )
    parser.add_argument(
        "--summary",
        dest="summary_path",
        help="Output summary CSV path (default: <csv_dir>/kernel_summary.csv)",
    )
    parser.add_argument(
        "--hotspots",
        dest="hotspots_path",
        help="Output hotspots CSV path (default: <csv_dir>/kernel_hotspots.csv)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of kernels to show in the plot (default: 20)",
    )
    return parser.parse_args()


def parse_int_field(row, *keys):
    for key in keys:
        raw = row.get(key)
        if raw is None:
            continue
        raw = raw.strip()
        if not raw:
            continue
        try:
            return int(raw)
        except ValueError:
            try:
                return int(float(raw))
            except ValueError:
                continue
    return None


def load_metrics(csv_paths):
    metrics = {}
    total_rows = 0
    for path in csv_paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                name = (row.get("name") or "").strip()
                if not name:
                    name = "<unnamed>"
                cpu_duration = parse_int_field(row, "duration_ns", "duration")
                if cpu_duration is None or cpu_duration < 0:
                    continue

                entry = metrics.setdefault(
                    name, {"cpu": [], "gpu": [], "scheduling": []}
                )
                entry["cpu"].append(cpu_duration)

                gpu_duration = parse_int_field(row, "gpu_duration_ns")
                if gpu_duration is not None and gpu_duration >= 0:
                    entry["gpu"].append(gpu_duration)

                scheduling_latency = parse_int_field(row, "scheduling_latency_ns")
                if scheduling_latency is None and gpu_duration is not None:
                    scheduling_latency = max(cpu_duration - gpu_duration, 0)
                if scheduling_latency is not None and scheduling_latency >= 0:
                    entry["scheduling"].append(scheduling_latency)

    return metrics, total_rows


def basic_stats(values):
    if not values:
        return None
    total = sum(values)
    count = len(values)
    median = statistics.median(values)
    mean = total / count
    return {
        "count": count,
        "median_ns": median,
        "median_ms": median / 1e6,
        "mean_ns": mean,
        "mean_ms": mean / 1e6,
        "total_ns": total,
        "total_ms": total / 1e6,
    }


def summarize(metrics):
    summary = []
    for name, entry in metrics.items():
        cpu_stats = basic_stats(entry["cpu"])
        if cpu_stats is None:
            continue

        gpu_stats = basic_stats(entry["gpu"])
        scheduling_stats = basic_stats(entry["scheduling"])
        median_scheduling_share_pct = None
        if (
            scheduling_stats is not None
            and cpu_stats["median_ns"] > 0
            and scheduling_stats["median_ns"] is not None
        ):
            median_scheduling_share_pct = (
                100.0 * scheduling_stats["median_ns"] / cpu_stats["median_ns"]
            )

        row = {
            "name": name,
            "count": cpu_stats["count"],
            "median_ns": cpu_stats["median_ns"],
            "median_ms": cpu_stats["median_ms"],
            "mean_ns": cpu_stats["mean_ns"],
            "mean_ms": cpu_stats["mean_ms"],
            "total_ns": cpu_stats["total_ns"],
            "total_ms": cpu_stats["total_ms"],
            "gpu_count": 0 if gpu_stats is None else gpu_stats["count"],
            "median_gpu_ns": None if gpu_stats is None else gpu_stats["median_ns"],
            "median_gpu_ms": None if gpu_stats is None else gpu_stats["median_ms"],
            "mean_gpu_ns": None if gpu_stats is None else gpu_stats["mean_ns"],
            "mean_gpu_ms": None if gpu_stats is None else gpu_stats["mean_ms"],
            "total_gpu_ns": None if gpu_stats is None else gpu_stats["total_ns"],
            "total_gpu_ms": None if gpu_stats is None else gpu_stats["total_ms"],
            "scheduling_count": (
                0 if scheduling_stats is None else scheduling_stats["count"]
            ),
            "median_scheduling_ns": (
                None if scheduling_stats is None else scheduling_stats["median_ns"]
            ),
            "median_scheduling_ms": (
                None if scheduling_stats is None else scheduling_stats["median_ms"]
            ),
            "mean_scheduling_ns": (
                None if scheduling_stats is None else scheduling_stats["mean_ns"]
            ),
            "mean_scheduling_ms": (
                None if scheduling_stats is None else scheduling_stats["mean_ms"]
            ),
            "total_scheduling_ns": (
                None if scheduling_stats is None else scheduling_stats["total_ns"]
            ),
            "total_scheduling_ms": (
                None if scheduling_stats is None else scheduling_stats["total_ms"]
            ),
            "median_scheduling_share_pct": median_scheduling_share_pct,
        }
        summary.append(row)

    summary.sort(key=lambda x: x["total_ns"], reverse=True)
    return summary


def fmt_int(value):
    if value is None:
        return ""
    return int(value)


def fmt_float(value, digits=6):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def write_summary(summary, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "count",
                "median_ns",
                "median_ms",
                "mean_ns",
                "mean_ms",
                "total_ns",
                "total_ms",
                "gpu_count",
                "median_gpu_ns",
                "median_gpu_ms",
                "mean_gpu_ns",
                "mean_gpu_ms",
                "total_gpu_ns",
                "total_gpu_ms",
                "scheduling_count",
                "median_scheduling_ns",
                "median_scheduling_ms",
                "mean_scheduling_ns",
                "mean_scheduling_ms",
                "total_scheduling_ns",
                "total_scheduling_ms",
                "median_scheduling_share_pct",
            ]
        )
        for row in summary:
            writer.writerow(
                [
                    row["name"],
                    row["count"],
                    fmt_int(row["median_ns"]),
                    fmt_float(row["median_ms"]),
                    fmt_int(row["mean_ns"]),
                    fmt_float(row["mean_ms"]),
                    fmt_int(row["total_ns"]),
                    fmt_float(row["total_ms"]),
                    row["gpu_count"],
                    fmt_int(row["median_gpu_ns"]),
                    fmt_float(row["median_gpu_ms"]),
                    fmt_int(row["mean_gpu_ns"]),
                    fmt_float(row["mean_gpu_ms"]),
                    fmt_int(row["total_gpu_ns"]),
                    fmt_float(row["total_gpu_ms"]),
                    row["scheduling_count"],
                    fmt_int(row["median_scheduling_ns"]),
                    fmt_float(row["median_scheduling_ms"]),
                    fmt_int(row["mean_scheduling_ns"]),
                    fmt_float(row["mean_scheduling_ms"]),
                    fmt_int(row["total_scheduling_ns"]),
                    fmt_float(row["total_scheduling_ms"]),
                    fmt_float(row["median_scheduling_share_pct"], digits=2),
                ]
            )


def plot_hotspots(summary, plot_path, top_n):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("error: matplotlib is required for plotting", file=sys.stderr)
        print(f"detail: {exc}", file=sys.stderr)
        return False

    if not summary:
        print("error: no data to plot", file=sys.stderr)
        return False

    top_n = max(1, top_n)
    shown = summary[: min(top_n, len(summary))]

    labels = [f"{row['name']} (n={row['count']})" for row in shown]
    cpu_median_ms = [row["median_ms"] for row in shown]
    total_cpu_ms = [row["total_ms"] for row in shown]
    gpu_median_ms = [
        0.0 if row["median_gpu_ms"] is None else row["median_gpu_ms"] for row in shown
    ]
    scheduling_median_ms = [
        0.0 if row["median_scheduling_ms"] is None else row["median_scheduling_ms"]
        for row in shown
    ]
    has_gpu_or_scheduling = any(
        row["gpu_count"] > 0 or row["scheduling_count"] > 0 for row in shown
    )

    subplot_count = 3 if has_gpu_or_scheduling else 2
    height = max(4.0, 0.35 * len(shown) * subplot_count)
    fig, axes = plt.subplots(subplot_count, 1, figsize=(11, height), sharey=True)
    if subplot_count == 2:
        ax_cpu, ax_total = axes
    else:
        ax_cpu, ax_gpu_sched, ax_total = axes

    ax_cpu.barh(labels, cpu_median_ms, color="#4C72B0")
    ax_cpu.set_xlabel("Median duration (ms)")
    ax_cpu.set_title("Median CPU duration (top by total CPU time)")
    ax_cpu.invert_yaxis()

    if has_gpu_or_scheduling:
        ax_gpu_sched.barh(labels, gpu_median_ms, color="#55A868", label="GPU events")
        ax_gpu_sched.barh(
            labels,
            scheduling_median_ms,
            left=gpu_median_ms,
            color="#C44E52",
            label="Scheduling latency",
        )
        ax_gpu_sched.set_xlabel("Median duration (ms)")
        ax_gpu_sched.set_title("Median GPU time + scheduling latency (CPU - GPU)")
        ax_gpu_sched.legend(loc="lower right")

    ax_total.barh(labels, total_cpu_ms, color="#DD8452")
    ax_total.set_xlabel("Total CPU time (ms)")
    ax_total.set_title("Total CPU time per kernel")

    for ax in axes:
        ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    return True


def write_hotspots(summary, path, top_n):
    if not summary:
        return False
    top_n = max(1, top_n)
    shown = summary[: min(top_n, len(summary))]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "name",
                "count",
                "median_ns",
                "median_ms",
                "total_ns",
                "total_ms",
                "gpu_count",
                "median_gpu_ns",
                "median_gpu_ms",
                "scheduling_count",
                "median_scheduling_ns",
                "median_scheduling_ms",
                "median_scheduling_share_pct",
            ]
        )
        for idx, row in enumerate(shown, start=1):
            writer.writerow(
                [
                    idx,
                    row["name"],
                    row["count"],
                    fmt_int(row["median_ns"]),
                    fmt_float(row["median_ms"]),
                    fmt_int(row["total_ns"]),
                    fmt_float(row["total_ms"]),
                    row["gpu_count"],
                    fmt_int(row["median_gpu_ns"]),
                    fmt_float(row["median_gpu_ms"]),
                    row["scheduling_count"],
                    fmt_int(row["median_scheduling_ns"]),
                    fmt_float(row["median_scheduling_ms"]),
                    fmt_float(row["median_scheduling_share_pct"], digits=2),
                ]
            )
    return True


def main():
    args = parse_args()
    csv_paths = [Path(p) for p in args.csv_files]
    for path in csv_paths:
        if not path.is_file():
            print(f"error: file not found: {path}", file=sys.stderr)
            return 1

    default_dir = csv_paths[0].parent
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else default_dir / "kernel_summary.csv"
    )
    plot_path = (
        Path(args.plot_path) if args.plot_path else default_dir / "kernel_hotspots.png"
    )
    hotspots_path = (
        Path(args.hotspots_path)
        if args.hotspots_path
        else default_dir / "kernel_hotspots.csv"
    )

    metrics, total_rows = load_metrics(csv_paths)
    if total_rows == 0:
        print("error: no rows read from CSV files", file=sys.stderr)
        return 1
    if not metrics:
        print("error: no valid duration data found", file=sys.stderr)
        return 1

    summary = summarize(metrics)
    write_summary(summary, summary_path)
    write_hotspots(summary, hotspots_path, args.top)

    plotted = plot_hotspots(summary, plot_path, args.top)

    print(f"summary: {summary_path}")
    print(f"hotspots: {hotspots_path}")
    if plotted:
        print(f"plot: {plot_path}")
    else:
        print("plot: skipped (matplotlib missing)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
