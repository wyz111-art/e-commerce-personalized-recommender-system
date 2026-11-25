"""Audit a sampled e-commerce event Parquet without modifying source data."""

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
    "source_file",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate data-quality and behavior audit reports from a Parquet sample."
    )
    parser.add_argument("input", type=Path, help="Sampled events Parquet file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/data_audit"),
        help="Report directory. Default: reports/data_audit",
    )
    return parser.parse_args()


def write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def collect_streaming(frame: pl.LazyFrame) -> pl.DataFrame:
    return frame.collect(engine="streaming")


def main() -> int:
    args = parse_args()
    try:
        if not args.input.is_file():
            raise FileNotFoundError(f"Input file not found: {args.input}")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()

        raw = pl.scan_parquet(args.input)
        schema = raw.collect_schema()
        missing_columns = sorted(REQUIRED_COLUMNS - set(schema.names()))
        if missing_columns:
            raise ValueError("Missing required columns: " + ", ".join(missing_columns))

        events = raw.with_columns(
            pl.col("event_time")
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S UTC", strict=False)
            .alias("event_at"),
            pl.col("event_type").str.to_lowercase().str.strip_chars(),
            pl.col("brand").str.to_lowercase().str.strip_chars(),
            pl.col("category_code").str.to_lowercase().str.strip_chars(),
            pl.concat_str(
                [pl.col("user_id").cast(pl.String), pl.col("user_session")],
                separator="_",
            ).alias("session_key"),
        ).with_columns(
            pl.col("event_at").dt.date().alias("event_date"),
            pl.col("event_at").dt.strftime("%Y-%m").alias("event_month"),
        )

        total_rows = collect_streaming(events.select(pl.len().alias("rows"))).item()

        missing_report = collect_streaming(
            events.select(
                [
                    pl.col(column).null_count().alias(column)
                    for column in schema.names()
                ]
                + [pl.col("event_at").null_count().alias("event_at_parse_failed")]
            )
        ).transpose(include_header=True, header_name="column", column_names=["null_count"])
        missing_report = missing_report.with_columns(
            (pl.col("null_count") / total_rows).alias("null_rate")
        ).sort("null_rate", descending=True)
        missing_report.write_csv(args.output_dir / "missing_values.csv")

        event_distribution = collect_streaming(
            events.group_by("event_type").agg(
                pl.len().alias("event_rows"),
                pl.col("user_id").n_unique().alias("users"),
                pl.col("session_key").n_unique().alias("sessions"),
                pl.col("product_id").n_unique().alias("products"),
            ).with_columns(
                (pl.col("event_rows") / total_rows).alias("event_share")
            ).sort("event_rows", descending=True)
        )
        event_distribution.write_csv(args.output_dir / "event_distribution.csv")

        source_distribution = collect_streaming(
            events.group_by("source_file").agg(
                pl.len().alias("event_rows"),
                pl.col("user_id").n_unique().alias("users"),
                pl.col("session_key").n_unique().alias("sessions"),
                pl.col("product_id").n_unique().alias("products"),
                pl.col("event_at").min().alias("min_event_at"),
                pl.col("event_at").max().alias("max_event_at"),
            ).sort("source_file")
        )
        source_distribution.write_csv(args.output_dir / "source_distribution.csv")

        exact_duplicate_rows = collect_streaming(
            events.group_by(schema.names()).len().filter(pl.col("len") > 1).select(
                (pl.col("len") - 1).sum().fill_null(0).alias("duplicate_rows")
            )
        ).item()

        price_summary = collect_streaming(
            events.select(
                pl.col("price").count().alias("non_null_prices"),
                pl.col("price").null_count().alias("null_prices"),
                (pl.col("price") < 0).sum().alias("negative_prices"),
                (pl.col("price") == 0).sum().alias("zero_prices"),
                pl.col("price").min().alias("min_price"),
                pl.col("price").quantile(0.01).alias("p01_price"),
                pl.col("price").quantile(0.25).alias("p25_price"),
                pl.col("price").median().alias("median_price"),
                pl.col("price").quantile(0.75).alias("p75_price"),
                pl.col("price").quantile(0.99).alias("p99_price"),
                pl.col("price").max().alias("max_price"),
            )
        )
        price_summary.write_csv(args.output_dir / "price_summary.csv")

        session_id_conflicts = collect_streaming(
            events.group_by("user_session")
            .agg(pl.col("user_id").n_unique().alias("user_count"))
            .filter(pl.col("user_count") > 1)
            .sort("user_count", descending=True)
        )
        session_id_conflicts.write_csv(args.output_dir / "session_id_conflicts.csv")

        category_mapping_conflicts = collect_streaming(
            events.filter(
                pl.col("category_id").is_not_null()
                & pl.col("category_code").is_not_null()
            )
            .group_by("category_id")
            .agg(
                pl.col("category_code").n_unique().alias("category_code_count"),
                pl.col("category_code").unique().sort().alias("category_codes"),
            )
            .filter(pl.col("category_code_count") > 1)
            .sort("category_code_count", descending=True)
        ).with_columns(
            pl.col("category_codes").list.join(" | ").alias("category_codes")
        )
        category_mapping_conflicts.write_csv(
            args.output_dir / "category_mapping_conflicts.csv"
        )

        product_metadata_conflicts = collect_streaming(
            events.group_by("product_id")
            .agg(
                pl.col("category_id").drop_nulls().n_unique().alias("category_id_count"),
                pl.col("category_code")
                .drop_nulls()
                .n_unique()
                .alias("category_code_count"),
                pl.col("brand").drop_nulls().n_unique().alias("brand_count"),
                pl.col("price").drop_nulls().n_unique().alias("price_count"),
            )
            .filter(
                (pl.col("category_id_count") > 1)
                | (pl.col("category_code_count") > 1)
                | (pl.col("brand_count") > 1)
            )
            .sort(["category_id_count", "category_code_count", "brand_count"], descending=True)
        )
        product_metadata_conflicts.write_csv(
            args.output_dir / "product_metadata_conflicts.csv"
        )

        user_summary = events.group_by("user_id").agg(
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
            args.output_dir / "user_summary.parquet", compression="zstd", mkdir=True
        )

        session_summary = events.group_by(["session_key", "user_id", "user_session"]).agg(
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
            args.output_dir / "session_summary.parquet", compression="zstd", mkdir=True
        )

        product_summary = events.group_by("product_id").agg(
            pl.len().alias("event_count"),
            (pl.col("event_type") == "view").sum().alias("view_count"),
            (pl.col("event_type") == "cart").sum().alias("cart_count"),
            (pl.col("event_type") == "purchase").sum().alias("purchase_count"),
            pl.col("user_id").n_unique().alias("unique_user_count"),
            pl.col("user_id")
            .filter(pl.col("event_type") == "purchase")
            .n_unique()
            .alias("purchase_user_count"),
            pl.col("category_id").drop_nulls().first().alias("category_id"),
            pl.col("category_code").drop_nulls().first().alias("category_code"),
            pl.col("brand").drop_nulls().first().alias("brand"),
            pl.col("price").drop_nulls().last().alias("latest_price"),
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
            args.output_dir / "product_summary.parquet", compression="zstd", mkdir=True
        )

        user_funnel = collect_streaming(
            events.group_by("user_id")
            .agg(pl.col("event_type").unique().alias("event_types"))
            .select(
                pl.len().alias("total_users"),
                pl.col("event_types").list.contains("view").sum().alias("view_users"),
                pl.col("event_types").list.contains("cart").sum().alias("cart_users"),
                pl.col("event_types").list.contains("purchase").sum().alias("purchase_users"),
            )
        ).with_columns(pl.lit("user").alias("grain"))

        session_funnel = collect_streaming(
            events.group_by("session_key")
            .agg(pl.col("event_type").unique().alias("event_types"))
            .select(
                pl.len().alias("total_users"),
                pl.col("event_types").list.contains("view").sum().alias("view_users"),
                pl.col("event_types").list.contains("cart").sum().alias("cart_users"),
                pl.col("event_types").list.contains("purchase").sum().alias("purchase_users"),
            )
        ).with_columns(pl.lit("session").alias("grain"))

        product_funnel = collect_streaming(
            events.group_by("product_id")
            .agg(pl.col("event_type").unique().alias("event_types"))
            .select(
                pl.len().alias("total_users"),
                pl.col("event_types").list.contains("view").sum().alias("view_users"),
                pl.col("event_types").list.contains("cart").sum().alias("cart_users"),
                pl.col("event_types").list.contains("purchase").sum().alias("purchase_users"),
            )
        ).with_columns(pl.lit("product").alias("grain"))

        funnel_summary = pl.concat([user_funnel, session_funnel, product_funnel]).select(
            "grain", "total_users", "view_users", "cart_users", "purchase_users"
        )
        funnel_summary.write_csv(args.output_dir / "funnel_summary.csv")

        user_metrics = collect_streaming(
            pl.scan_parquet(args.output_dir / "user_summary.parquet").select(
                pl.len().alias("users"),
                (pl.col("session_count") == 1).sum().alias("one_session_users"),
                (pl.col("session_count") == 2).sum().alias("two_session_users"),
                (pl.col("session_count") >= 3).sum().alias("three_plus_session_users"),
                pl.col("has_purchase").sum().alias("purchase_users"),
                pl.col("is_cross_month_user").sum().alias("cross_month_users"),
                pl.col("session_count").median().alias("median_sessions_per_user"),
                pl.col("event_count").median().alias("median_events_per_user"),
            )
        ).to_dicts()[0]

        session_metrics = collect_streaming(
            pl.scan_parquet(args.output_dir / "session_summary.parquet").select(
                pl.len().alias("sessions"),
                pl.col("has_purchase").sum().alias("purchase_sessions"),
                pl.col("is_single_event_session").sum().alias("single_event_sessions"),
                pl.col("event_count").median().alias("median_events_per_session"),
                pl.col("unique_product_count")
                .median()
                .alias("median_products_per_session"),
                pl.col("session_duration").median().alias("median_session_duration"),
                pl.col("session_duration").quantile(0.99).alias("p99_session_duration"),
            )
        ).to_dicts()[0]

        product_metrics = collect_streaming(
            pl.scan_parquet(args.output_dir / "product_summary.parquet").select(
                pl.len().alias("products"),
                (pl.col("purchase_count") > 0).sum().alias("purchased_products"),
                (pl.col("purchase_count") == 1).sum().alias("single_purchase_products"),
                (pl.col("active_month_count") > 1).sum().alias("cross_month_products"),
                pl.col("event_count").median().alias("median_events_per_product"),
            )
        ).to_dicts()[0]

        overall = collect_streaming(
            events.select(
                pl.len().alias("rows"),
                pl.col("user_id").n_unique().alias("users"),
                pl.col("session_key").n_unique().alias("sessions"),
                pl.col("product_id").n_unique().alias("products"),
                pl.col("event_at").min().alias("min_event_at"),
                pl.col("event_at").max().alias("max_event_at"),
                pl.col("source_file").n_unique().alias("source_files"),
            )
        ).to_dicts()[0]

        report = {
            "input": str(args.input.resolve()),
            "schema": {name: str(dtype) for name, dtype in schema.items()},
            "overall": overall,
            "exact_duplicate_rows": exact_duplicate_rows,
            "invalid_event_types": collect_streaming(
                events.filter(
                    ~pl.col("event_type").is_in(["view", "cart", "purchase"])
                ).select(pl.len().alias("rows"))
            ).item(),
            "session_ids_shared_by_multiple_users": session_id_conflicts.height,
            "category_ids_with_multiple_codes": category_mapping_conflicts.height,
            "products_with_metadata_conflicts": product_metadata_conflicts.height,
            "user_metrics": user_metrics,
            "session_metrics": session_metrics,
            "product_metrics": product_metrics,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        write_json(args.output_dir / "data_quality_summary.json", report)

        print(f"Audit completed: {args.output_dir}")
        print(json.dumps(report["overall"], ensure_ascii=False, default=str))
        print(f"Elapsed: {report['elapsed_seconds']:.2f} seconds")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
