"""Stream-sample complete user histories from large e-commerce CSV files.

The same deterministic user_id rule is applied to every input file, so a
selected user keeps all events across all supplied months. The script scans
CSV files lazily with Polars and writes one compressed Parquet file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl


REQUIRED_COLUMNS = {
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
}

SCHEMA_OVERRIDES = {
    "event_time": pl.String,
    "event_type": pl.String,
    "product_id": pl.Int64,
    "category_id": pl.Int64,
    "category_code": pl.String,
    "brand": pl.String,
    "price": pl.Float64,
    "user_id": pl.Int64,
    "user_session": pl.String,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply one deterministic user_id sample to multiple large CSV "
            "files and stream the selected events to Parquet."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input CSV files, for example September.csv October.csv",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output Parquet file path",
    )
    parser.add_argument(
        "--sample-percent",
        type=float,
        default=1.0,
        help="Percentage of users to keep. Default: 1.0",
    )
    parser.add_argument(
        "--buckets",
        type=int,
        default=10_000,
        help="Number of deterministic user buckets. Default: 10000",
    )
    parser.add_argument(
        "--separator",
        default=",",
        help="CSV separator. Default: comma",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        help="Audit JSON path. Default: <output>.audit.json",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.sample_percent <= 100:
        raise ValueError("--sample-percent must be greater than 0 and at most 100")
    if args.buckets <= 0:
        raise ValueError("--buckets must be greater than 0")
    if len(args.separator) != 1:
        raise ValueError("--separator must contain exactly one character")

    missing_files = [str(path) for path in args.inputs if not path.is_file()]
    if missing_files:
        raise FileNotFoundError("Input files not found: " + ", ".join(missing_files))


def selected_bucket_count(sample_percent: float, buckets: int) -> int:
    count = round(buckets * sample_percent / 100)
    return max(1, min(count, buckets))


def scan_input(path: Path, separator: str) -> pl.LazyFrame:
    frame = pl.scan_csv(
        path,
        separator=separator,
        schema_overrides=SCHEMA_OVERRIDES,
        null_values=["", "null", "NULL", "None", "NaN"],
        infer_schema_length=10_000,
        low_memory=True,
        try_parse_dates=False,
    )

    columns = set(frame.collect_schema().names())
    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    return frame.with_columns(pl.lit(path.name).alias("source_file"))


def build_sample(
    paths: list[Path],
    separator: str,
    buckets: int,
    selected_buckets: int,
) -> pl.LazyFrame:
    scans = [scan_input(path, separator) for path in paths]
    combined = pl.concat(scans, how="vertical_relaxed")

    # user_id values in this dataset are numeric. Modulo bucketing is stable,
    # reproducible, and applies the same user selection to every month.
    keep_user = (
        pl.col("user_id").is_not_null()
        & ((pl.col("user_id").cast(pl.UInt64) % buckets) < selected_buckets)
    )

    return combined.filter(keep_user)


def write_audit(
    parquet_path: Path,
    audit_path: Path,
    sample_percent: float,
    buckets: int,
    selected_buckets: int,
    input_paths: list[Path],
) -> None:
    sampled = pl.scan_parquet(parquet_path)

    summary = sampled.select(
        pl.len().alias("rows"),
        pl.col("user_id").n_unique().alias("users"),
        pl.col("product_id").n_unique().alias("products"),
        pl.col("user_session").n_unique().alias("sessions"),
        pl.col("event_time").min().alias("min_event_time"),
        pl.col("event_time").max().alias("max_event_time"),
    ).collect(engine="streaming").to_dicts()[0]

    event_counts = (
        sampled.group_by("event_type")
        .agg(pl.len().alias("rows"))
        .sort("rows", descending=True)
        .collect(engine="streaming")
        .to_dicts()
    )

    source_counts = (
        sampled.group_by("source_file")
        .agg(pl.len().alias("rows"))
        .sort("source_file")
        .collect(engine="streaming")
        .to_dicts()
    )

    purchase_rows = sum(
        row["rows"] for row in event_counts if row["event_type"] == "purchase"
    )
    purchase_share = purchase_rows / summary["rows"] if summary["rows"] else 0.0
    purchase_summary = (
        sampled.filter(pl.col("event_type") == "purchase")
        .select(
            pl.col("user_id").n_unique().alias("purchase_users"),
            pl.col("user_session").n_unique().alias("purchase_sessions"),
            pl.col("product_id").n_unique().alias("purchased_products"),
        )
        .collect(engine="streaming")
        .to_dicts()[0]
    )

    report = {
        "inputs": [str(path.resolve()) for path in input_paths],
        "output": str(parquet_path.resolve()),
        "sampling": {
            "method": "user_id modulo buckets",
            "requested_percent": sample_percent,
            "buckets": buckets,
            "selected_buckets": selected_buckets,
            "effective_percent": selected_buckets / buckets * 100,
        },
        "summary": summary,
        "purchase_rows": purchase_rows,
        "purchase_event_share": purchase_share,
        "purchase_summary": purchase_summary,
        "event_counts": event_counts,
        "source_counts": source_counts,
    }

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        selected_buckets = selected_bucket_count(args.sample_percent, args.buckets)
        effective_percent = selected_buckets / args.buckets * 100

        args.output.parent.mkdir(parents=True, exist_ok=True)
        audit_path = args.audit_output or args.output.with_suffix(".audit.json")

        print(f"Input files: {len(args.inputs)}")
        print(f"Requested sample: {args.sample_percent:.4f}% of users")
        print(f"Effective sample: {effective_percent:.4f}% of users")
        print(f"Output: {args.output}")

        started = time.perf_counter()
        sample = build_sample(
            args.inputs,
            args.separator,
            args.buckets,
            selected_buckets,
        )
        sample.sink_parquet(
            args.output,
            compression="zstd",
            statistics=True,
            mkdir=True,
        )

        write_audit(
            args.output,
            audit_path,
            args.sample_percent,
            args.buckets,
            selected_buckets,
            args.inputs,
        )

        elapsed = time.perf_counter() - started
        print(f"Completed in {elapsed / 60:.2f} minutes")
        print(f"Audit report: {audit_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
