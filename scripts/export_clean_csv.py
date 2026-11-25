"""Export cleaned event Parquet to a SQL Server-friendly CSV file."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl


EXPECTED_COLUMNS = [
    "event_at",
    "event_date",
    "event_month",
    "event_hour",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "category_code_missing",
    "brand_clean",
    "brand_missing_original",
    "price",
    "price_zero_flag",
    "price_negative_flag",
    "user_id",
    "user_session",
    "session_key",
    "source_file",
    "metadata_conflict_flag",
]

BOOLEAN_COLUMNS = [
    "category_code_missing",
    "brand_missing_original",
    "price_zero_flag",
    "price_negative_flag",
    "metadata_conflict_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a cleaned Parquet file to a SQL Server-friendly CSV."
    )
    parser.add_argument("input", type=Path, help="Cleaned input Parquet file")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if not args.input.is_file():
            raise FileNotFoundError(f"Input file not found: {args.input}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()

        source = pl.scan_parquet(args.input)
        columns = source.collect_schema().names()
        missing = [column for column in EXPECTED_COLUMNS if column not in columns]
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))

        sql_ready = source.select(EXPECTED_COLUMNS).with_columns(
            pl.col("event_at")
            .dt.strftime("%Y-%m-%d %H:%M:%S.%6f")
            .alias("event_at"),
            pl.col("event_date").dt.strftime("%Y-%m-%d").alias("event_date"),
            *[pl.col(column).cast(pl.UInt8) for column in BOOLEAN_COLUMNS],
        )

        sql_ready.sink_csv(
            args.output,
            include_header=True,
            separator=",",
            quote_char='"',
            quote_style="necessary",
            null_value="",
            float_precision=4,
            maintain_order=True,
            mkdir=True,
            engine="streaming",
        )

        input_rows = pl.scan_parquet(args.input).select(pl.len()).collect().item()
        output_rows = (
            pl.scan_csv(args.output, infer_schema=False)
            .select(pl.len())
            .collect(engine="streaming")
            .item()
        )
        if input_rows != output_rows:
            raise RuntimeError(
                f"Row count mismatch: Parquet={input_rows}, CSV={output_rows}"
            )

        report = {
            "input": str(args.input.resolve()),
            "output": str(args.output.resolve()),
            "rows": input_rows,
            "columns": EXPECTED_COLUMNS,
            "boolean_format": "0/1",
            "event_at_format": "yyyy-MM-dd HH:mm:ss.ffffff",
            "null_format": "empty field",
            "output_bytes": args.output.stat().st_size,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        report_path = args.output.with_suffix(".export.json")
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(f"CSV created: {args.output}")
        print(f"Rows verified: {input_rows}")
        print(f"Export report: {report_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
