"""Clean sampled REES46 events and create modeling-friendly summaries."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl


EXPECTED_COLUMNS = {
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
    "source_file",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean a sampled REES46 Parquet file without changing the source."
    )
    parser.add_argument("input", type=Path, help="Input sampled Parquet")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed"),
        help="Output directory. Default: processed",
    )
    return parser.parse_args()


def collect_one(frame: pl.LazyFrame) -> dict[str, object]:
    return frame.collect(engine="streaming").to_dicts()[0]


def main() -> int:
    args = parse_args()
    try:
        if not args.input.is_file():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()

        raw = pl.scan_parquet(args.input)
        original_columns = raw.collect_schema().names()
        missing = sorted(EXPECTED_COLUMNS - set(original_columns))
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))

        input_rows = int(collect_one(raw.select(pl.len().alias("rows")))["rows"])

        # Exact duplicates are defined over every column in the sampled source.
        deduplicated = raw.unique(subset=original_columns, maintain_order=False)
        deduplicated_rows = int(
            collect_one(deduplicated.select(pl.len().alias("rows")))["rows"]
        )
        duplicate_rows_removed = input_rows - deduplicated_rows

        normalized = deduplicated.with_columns(
            pl.col("event_time")
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S UTC", strict=False)
            .alias("event_at"),
            pl.col("event_type").str.to_lowercase().str.strip_chars(),
            pl.col("brand")
            .str.to_lowercase()
            .str.strip_chars()
            .replace("", None)
            .alias("brand_normalized"),
            pl.col("category_code")
            .str.to_lowercase()
            .str.strip_chars()
            .replace("", None)
            .alias("category_code"),
        )

        parse_failures = int(
            collect_one(
                normalized.select(
                    pl.col("event_at").null_count().alias("parse_failures")
                )
            )["parse_failures"]
        )
        if parse_failures:
            raise ValueError(
                f"{parse_failures} event_time values could not be parsed; source was not changed"
            )

        # Backfill a brand only when the product has exactly one observed,
        # non-empty brand. Conflicting products are left unresolved.
        stable_brand_map = (
            normalized.filter(pl.col("brand_normalized").is_not_null())
            .group_by("product_id")
            .agg(
                pl.col("brand_normalized").n_unique().alias("brand_count"),
                pl.col("brand_normalized").first().alias("stable_brand"),
            )
            .filter(pl.col("brand_count") == 1)
            .select("product_id", "stable_brand")
        )

        product_conflicts = (
            normalized.group_by("product_id")
            .agg(
                pl.col("category_id")
                .drop_nulls()
                .n_unique()
                .alias("category_id_count"),
                pl.col("category_code")
                .drop_nulls()
                .n_unique()
                .alias("category_code_count"),
                pl.col("brand_normalized")
                .drop_nulls()
                .n_unique()
                .alias("brand_count"),
            )
            .with_columns(
                (
                    (pl.col("category_id_count") > 1)
                    | (pl.col("category_code_count") > 1)
                    | (pl.col("brand_count") > 1)
                ).alias("metadata_conflict_flag")
            )
            .select("product_id", "metadata_conflict_flag")
        )

        enriched = normalized.join(stable_brand_map, on="product_id", how="left").join(
            product_conflicts, on="product_id", how="left"
        )

        clean = enriched.with_columns(
            pl.coalesce([pl.col("brand_normalized"), pl.col("stable_brand")])
            .fill_null("unknown_brand")
            .alias("brand_clean"),
            pl.col("brand_normalized").is_null().alias("brand_missing_original"),
            pl.col("category_code").is_null().alias("category_code_missing"),
            (pl.col("price") == 0).fill_null(False).alias("price_zero_flag"),
            (pl.col("price") < 0).fill_null(False).alias("price_negative_flag"),
            pl.col("metadata_conflict_flag").fill_null(False),
            pl.concat_str(
                [pl.col("user_id").cast(pl.String), pl.col("user_session")],
                separator="_",
            ).alias("session_key"),
        ).with_columns(
            pl.col("event_at").dt.date().alias("event_date"),
            pl.col("event_at").dt.strftime("%Y-%m").alias("event_month"),
            pl.col("event_at").dt.hour().alias("event_hour"),
        )

        output_columns = [
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
        clean = clean.select(output_columns)

        clean_path = args.output_dir / "clean_events_2019_10_11_1pct.parquet"
        clean.sink_parquet(
            clean_path,
            compression="zstd",
            statistics=True,
            mkdir=True,
        )

        clean_scan = pl.scan_parquet(clean_path)
        user_summary = clean_scan.group_by("user_id").agg(
            pl.len().alias("event_count"),
            (pl.col("event_type") == "view").sum().alias("view_count"),
            (pl.col("event_type") == "cart").sum().alias("cart_count"),
            (pl.col("event_type") == "purchase").sum().alias("purchase_count"),
            pl.col("session_key").n_unique().alias("session_count"),
            pl.col("product_id").n_unique().alias("unique_product_count"),
            pl.col("category_id").drop_nulls().n_unique().alias("unique_category_count"),
            pl.col("event_date").n_unique().alias("active_day_count"),
            pl.col("event_month").n_unique().alias("active_month_count"),
            pl.col("event_at").min().alias("first_event_at"),
            pl.col("event_at").max().alias("last_event_at"),
        ).with_columns(
            (pl.col("purchase_count") > 0).alias("has_purchase"),
            (pl.col("active_month_count") > 1).alias("is_cross_month_user"),
        )
        user_summary.sink_parquet(
            args.output_dir / "user_summary_clean.parquet",
            compression="zstd",
            mkdir=True,
        )

        session_summary = clean_scan.group_by(
            ["session_key", "user_id", "user_session"]
        ).agg(
            pl.len().alias("event_count"),
            (pl.col("event_type") == "view").sum().alias("view_count"),
            (pl.col("event_type") == "cart").sum().alias("cart_count"),
            (pl.col("event_type") == "purchase").sum().alias("purchase_count"),
            pl.col("product_id").n_unique().alias("unique_product_count"),
            pl.col("category_id").drop_nulls().n_unique().alias("unique_category_count"),
            pl.col("event_at").min().alias("session_start"),
            pl.col("event_at").max().alias("session_end"),
        ).with_columns(
            (pl.col("session_end") - pl.col("session_start")).alias("session_duration"),
            (pl.col("purchase_count") > 0).alias("has_purchase"),
            (pl.col("event_count") == 1).alias("is_single_event_session"),
        )
        session_summary.sink_parquet(
            args.output_dir / "session_summary_clean.parquet",
            compression="zstd",
            mkdir=True,
        )

        product_summary = clean_scan.group_by("product_id").agg(
            pl.len().alias("event_count"),
            (pl.col("event_type") == "view").sum().alias("view_count"),
            (pl.col("event_type") == "cart").sum().alias("cart_count"),
            (pl.col("event_type") == "purchase").sum().alias("purchase_count"),
            pl.col("user_id").n_unique().alias("unique_user_count"),
            pl.col("user_id")
            .filter(pl.col("event_type") == "purchase")
            .n_unique()
            .alias("purchase_user_count"),
            pl.col("category_id")
            .sort_by("event_at")
            .drop_nulls()
            .last()
            .alias("latest_category_id"),
            pl.col("category_code")
            .sort_by("event_at")
            .drop_nulls()
            .last()
            .alias("latest_category_code"),
            pl.col("brand_clean")
            .sort_by("event_at")
            .drop_nulls()
            .last()
            .alias("latest_brand"),
            pl.col("price")
            .sort_by("event_at")
            .drop_nulls()
            .last()
            .alias("latest_observed_price"),
            pl.col("metadata_conflict_flag").max().alias("metadata_conflict_flag"),
            pl.col("event_at").min().alias("first_event_at"),
            pl.col("event_at").max().alias("last_event_at"),
            pl.col("event_month").n_unique().alias("active_month_count"),
        ).with_columns(
            pl.when(pl.col("view_count") > 0)
            .then(pl.col("cart_count") / pl.col("view_count"))
            .otherwise(None)
            .alias("cart_per_view"),
            pl.when(pl.col("view_count") > 0)
            .then(pl.col("purchase_count") / pl.col("view_count"))
            .otherwise(None)
            .alias("purchase_per_view"),
        )
        product_summary.sink_parquet(
            args.output_dir / "product_summary_clean.parquet",
            compression="zstd",
            mkdir=True,
        )

        after_type_rows = (
            clean_scan.group_by("event_type")
            .agg(pl.len().alias("rows"))
            .sort("event_type")
            .collect(engine="streaming")
            .to_dicts()
        )

        brand_stats = collect_one(
            clean_scan.select(
                pl.col("brand_missing_original").sum().alias("original_brand_missing_rows"),
                (pl.col("brand_clean") == "unknown_brand")
                .sum()
                .alias("unknown_brand_rows"),
                pl.col("category_code_missing").sum().alias("category_code_missing_rows"),
                pl.col("price_zero_flag").sum().alias("zero_price_rows"),
                pl.col("price_negative_flag").sum().alias("negative_price_rows"),
                pl.col("metadata_conflict_flag").sum().alias("metadata_conflict_rows"),
            )
        )

        summary = {
            "input": str(args.input.resolve()),
            "output": str(clean_path.resolve()),
            "input_rows": input_rows,
            "clean_rows": deduplicated_rows,
            "exact_duplicate_rows_removed": duplicate_rows_removed,
            "event_type_counts_after_cleaning": after_type_rows,
            "brand_category_price_flags": brand_stats,
            "session_key_definition": "user_id + '_' + user_session",
            "category_code_policy": "keep null; do not synthesize category labels",
            "user_or_product_filtering": "none",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        report_path = args.output_dir / "cleaning_summary.json"
        report_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        # Keep a machine-readable duplicate count in the report and fail if the
        # post-clean event count is inconsistent with the expected input delta.
        if input_rows - deduplicated_rows != duplicate_rows_removed:
            raise RuntimeError("Duplicate-removal row counts do not reconcile")

        print(f"Clean data: {clean_path}")
        print(f"Cleaning report: {report_path}")
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
