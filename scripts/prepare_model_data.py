"""Prepare chronological Item-CF/User-CF inputs, without fitting any model.

Only observed identifiers, events and timestamps are used. Brand/category
imputation from the cleaning stage is deliberately not used for CF inputs.
All cutoffs are UTC, including timezone-naive timestamps in the input file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import polars as pl

DATA_START = datetime(2019, 10, 1)
TRAIN_END = datetime(2019, 11, 11)
TEST_START = datetime(2019, 11, 21)
DATA_END = datetime(2019, 12, 1)
VALID_EVENTS = ["view", "cart", "purchase"]
REQUIRED_COLUMNS = ["event_at", "event_type", "user_id", "product_id", "session_key"]
OUTPUT_NAMES = [
    "train_interactions.parquet",
    "validation_targets.parquet",
    "test_history_interactions.parquet",
    "test_targets.parquet",
    "validation_user_segments.parquet",
    "test_user_segments.parquet",
    "train_user_history_summary.parquet",
    "test_user_history_summary.parquet",
    "train_product_history_summary.parquet",
    "test_product_history_summary.parquet",
    "model_data_manifest.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Cleaned events Parquet")
    parser.add_argument("--output-dir", type=Path, default=Path("model_data"))
    parser.add_argument("--view-weight", type=float, default=1.0)
    parser.add_argument("--cart-weight", type=float, default=3.0)
    parser.add_argument("--purchase-weight", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing model-data outputs")
    return parser.parse_args()


def collect(frame: pl.LazyFrame) -> pl.DataFrame:
    return frame.collect(engine="streaming")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def scan_events(path: Path) -> pl.LazyFrame:
    source = pl.scan_parquet(path)
    schema = source.collect_schema()
    require(set(REQUIRED_COLUMNS) <= set(schema.names()), "Input lacks required event columns")
    require(isinstance(schema["event_at"], pl.Datetime), "event_at must be a Datetime column")
    source = source.select(REQUIRED_COLUMNS)
    if schema["event_at"].time_zone is not None:
        source = source.with_columns(
            pl.col("event_at").dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        )
    return source


def validate_events(events: pl.LazyFrame) -> dict:
    invalid_required = pl.any_horizontal(
        [pl.col(name).is_null() for name in REQUIRED_COLUMNS]
    )
    quality = collect(events.select(
        pl.len().alias("rows"),
        invalid_required.sum().alias("missing_required_rows"),
        (~pl.col("event_type").is_in(VALID_EVENTS)).fill_null(True).sum().alias("invalid_event_rows"),
        ((pl.col("event_at") < DATA_START) | (pl.col("event_at") >= DATA_END))
        .sum().alias("outside_time_range_rows"),
        pl.col("event_at").min().alias("min_time_utc"),
        pl.col("event_at").max().alias("max_time_utc"),
    )).to_dicts()[0]
    require(quality["rows"] > 0, "Input has no events")
    for name in ["missing_required_rows", "invalid_event_rows", "outside_time_range_rows"]:
        require(quality[name] == 0, f"Input quality check failed: {name}={quality[name]}")
    return quality


def build_interactions(history: pl.LazyFrame, view_weight=1.0, cart_weight=3.0,
                       purchase_weight=5.0) -> pl.DataFrame:
    return collect(
        history.group_by(["user_id", "product_id"]).agg(
            pl.len().cast(pl.Int64).alias("event_count"),
            *[(pl.col("event_type") == event).sum().cast(pl.Int64).alias(f"{event}_count")
              for event in VALID_EVENTS],
            pl.col("session_key").n_unique().alias("session_count"),
            pl.col("event_at").min().alias("first_interaction_at"),
            pl.col("event_at").max().alias("last_interaction_at"),
        ).with_columns(
            (pl.col("cart_count") > 0).alias("cart_flag"),
            (pl.col("purchase_count") > 0).alias("purchase_flag"),
        ).with_columns(
            (view_weight * pl.col("view_count").cast(pl.Float64).log1p()
             + cart_weight * pl.col("cart_flag").cast(pl.Float64)
             + purchase_weight * pl.col("purchase_flag").cast(pl.Float64))
            .alias("interaction_score")
        ).sort(["user_id", "product_id"])
    )


def build_user_history_summary(history: pl.LazyFrame) -> pl.DataFrame:
    return collect(
        history.group_by("user_id").agg(
            pl.len().cast(pl.Int64).alias("event_count"),
            pl.col("session_key").n_unique().alias("session_count"),
            pl.col("product_id").n_unique().alias("unique_product_count"),
            pl.col("event_at").dt.date().n_unique().alias("active_day_count"),
            *[(pl.col("event_type") == event).sum().cast(pl.Int64).alias(f"{event}_count")
              for event in VALID_EVENTS],
            pl.col("event_at").min().alias("first_event_at"),
            pl.col("event_at").max().alias("last_event_at"),
        ).with_columns(
            pl.when(pl.col("session_count") == 1).then(pl.lit("single_session"))
            .when(pl.col("session_count") <= 3).then(pl.lit("low_activity"))
            .when(pl.col("session_count") <= 7).then(pl.lit("medium_activity"))
            .otherwise(pl.lit("high_activity")).alias("activity_segment")
        ).sort("user_id")
    )


def build_product_history_summary(history: pl.LazyFrame) -> pl.DataFrame:
    # Counts only: do not reuse full-period brand backfills or metadata-conflict flags.
    return collect(
        history.group_by("product_id").agg(
            pl.len().cast(pl.Int64).alias("event_count"),
            *[(pl.col("event_type") == event).sum().cast(pl.Int64).alias(f"{event}_count")
              for event in VALID_EVENTS],
            pl.col("user_id").n_unique().alias("unique_user_count"),
            pl.col("event_at").min().alias("first_seen_at"),
            pl.col("event_at").max().alias("last_seen_at"),
        ).sort(["purchase_count", "product_id"], descending=[True, False])
    )


def build_targets(future: pl.LazyFrame, interactions: pl.DataFrame,
                  users: pl.DataFrame, products: pl.DataFrame) -> pl.DataFrame:
    targets = collect(
        future.filter(pl.col("event_type") == "purchase")
        .group_by(["user_id", "product_id"]).agg(
            pl.len().cast(pl.Int64).alias("future_purchase_event_count"),
            pl.col("session_key").n_unique().alias("future_purchase_session_count"),
            pl.col("event_at").min().alias("first_future_purchase_at"),
            pl.col("event_at").max().alias("last_future_purchase_at"),
        )
    )
    relation = interactions.select(
        "user_id", "product_id", "view_count", "cart_count", "purchase_count", "last_interaction_at"
    ).rename({
        "view_count": "history_view_count", "cart_count": "history_cart_count",
        "purchase_count": "history_purchase_count",
        "last_interaction_at": "history_last_interaction_at",
    })
    targets = (
        targets.join(relation, on=["user_id", "product_id"], how="left", validate="1:1")
        .join(users.select("user_id").with_columns(pl.lit(True).alias("known_user")),
              on="user_id", how="left", validate="m:1")
        .join(products.select("product_id").with_columns(pl.lit(True).alias("known_product")),
              on="product_id", how="left", validate="m:1")
        .with_columns(
            pl.col("known_user").fill_null(False), pl.col("known_product").fill_null(False),
            *[pl.col(f"history_{event}_count").fill_null(0) for event in VALID_EVENTS],
        )
        .with_columns(
            pl.when(pl.col("known_user") & pl.col("known_product"))
            .then(pl.lit("known_user_known_product"))
            .when(pl.col("known_user")).then(pl.lit("known_user_new_product"))
            .when(pl.col("known_product")).then(pl.lit("new_user_known_product"))
            .otherwise(pl.lit("new_user_new_product")).alias("cold_start_segment"),
            pl.when(pl.col("history_purchase_count") > 0).then(pl.lit("prior_purchase"))
            .when(pl.col("history_cart_count") > 0).then(pl.lit("prior_cart_only"))
            .when(pl.col("history_view_count") > 0).then(pl.lit("prior_view_only"))
            .otherwise(pl.lit("no_prior_user_product_interaction"))
            .alias("prior_user_product_segment"),
            (pl.col("history_purchase_count") == 0).alias("is_new_purchase_for_user"),
            ((pl.col("history_view_count") + pl.col("history_cart_count")
              + pl.col("history_purchase_count")) == 0).alias("is_unseen_to_user"),
        ).sort(["user_id", "product_id"])
    )
    return targets


def build_target_user_segments(targets: pl.DataFrame, history_users: pl.DataFrame) -> pl.DataFrame:
    # Future purchases select evaluation users only, never determine activity labels.
    joined = targets.select("user_id").unique().join(
        history_users, on="user_id", how="left", validate="1:1"
    )
    return joined.with_columns(
        pl.col("event_count").is_not_null().alias("known_user"),
        *[pl.col(name).fill_null(0) for name in [
            "event_count", "session_count", "unique_product_count", "active_day_count",
            "view_count", "cart_count", "purchase_count",
        ]],
        pl.col("activity_segment").fill_null("new_user"),
    ).sort("user_id")


def summarize_stage(history: pl.LazyFrame, future: pl.LazyFrame, interactions: pl.DataFrame,
                    targets: pl.DataFrame, users: pl.DataFrame, products: pl.DataFrame,
                    cutoff: datetime, end: datetime) -> dict:
    past_count = collect(history.select(pl.len())).item()
    future_purchase = collect(future.filter(pl.col("event_type") == "purchase").select(
        pl.len().alias("purchase_events"),
        pl.col("session_key").n_unique().alias("purchase_sessions"),
        pl.col("user_id").n_unique().alias("purchase_users"),
        pl.col("product_id").n_unique().alias("purchased_products"),
    )).to_dicts()[0]
    require(interactions.select("user_id", "product_id").unique().height == interactions.height,
            "Duplicate history user-product pairs")
    require(targets.select("user_id", "product_id").unique().height == targets.height,
            "Duplicate target user-product pairs")
    require(int(interactions["event_count"].sum() or 0) == past_count,
            "History aggregation does not reconcile")
    require(int(targets["future_purchase_event_count"].sum() or 0)
            == future_purchase["purchase_events"], "Target purchase count does not reconcile")
    if interactions.height:
        require(interactions["last_interaction_at"].max() < cutoff, "Future history leakage")
        require(interactions["interaction_score"].is_finite().all(), "Non-finite scores")
        require((interactions["interaction_score"] >= 0).all(), "Negative scores")
    if targets.height:
        require(targets["first_future_purchase_at"].min() >= cutoff, "Target before cutoff")
        require(targets["last_future_purchase_at"].max() < end, "Target beyond evaluation window")

    def breakdown(column: str) -> list[dict]:
        return targets.group_by(column).agg(
            pl.len().alias("user_product_pairs"),
            pl.col("future_purchase_event_count").sum().alias("purchase_events"),
        ).sort(column).to_dicts()

    target_users = targets.select("user_id", "known_user").unique()
    target_products = targets.select("product_id", "known_product").unique()
    return {
        "history_events": past_count, "history_users": users.height,
        "history_products": products.height, "history_user_product_pairs": interactions.height,
        **future_purchase, "target_user_product_pairs": targets.height,
        "known_purchase_users": int(target_users["known_user"].sum() or 0),
        "known_purchased_products": int(target_products["known_product"].sum() or 0),
        "new_purchase_target_pairs": int(targets["is_new_purchase_for_user"].sum() or 0),
        "unseen_target_pairs": int(targets["is_unseen_to_user"].sum() or 0),
        "prior_user_product_breakdown": breakdown("prior_user_product_segment"),
        "cold_start_breakdown": breakdown("cold_start_segment"),
        "checks_passed": True,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(input_path: Path, output_dir: Path, view_weight=1.0, cart_weight=3.0,
            purchase_weight=5.0, overwrite=False) -> dict:
    started = time.perf_counter()
    weights = [view_weight, cart_weight, purchase_weight]
    require(all(math.isfinite(w) and w >= 0 for w in weights) and any(w > 0 for w in weights),
            "Weights must be finite, non-negative, and not all zero")
    require(input_path.is_file(), f"Input file not found: {input_path}")
    existing = [name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    require(overwrite or not existing, "Outputs already exist; use a new output directory or --overwrite")
    require(input_path.resolve() not in [(output_dir / name).resolve() for name in OUTPUT_NAMES],
            "Output must not overwrite input")
    fingerprint = sha256_file(input_path)
    events = scan_events(input_path)
    quality = validate_events(events)

    train = events.filter(pl.col("event_at") < TRAIN_END)
    pretest = events.filter(pl.col("event_at") < TEST_START)
    validation = events.filter((pl.col("event_at") >= TRAIN_END) & (pl.col("event_at") < TEST_START))
    test = events.filter((pl.col("event_at") >= TEST_START) & (pl.col("event_at") < DATA_END))
    tables = {}
    stages = {}
    configurations = [
        ("validation", train, validation, TRAIN_END, TEST_START, "train_interactions.parquet",
         "train_user_history_summary.parquet", "train_product_history_summary.parquet"),
        ("test", pretest, test, TEST_START, DATA_END, "test_history_interactions.parquet",
         "test_user_history_summary.parquet", "test_product_history_summary.parquet"),
    ]
    for name, history, future, cutoff, end, interaction_name, user_name, product_name in configurations:
        interactions = build_interactions(history, *weights)
        users = build_user_history_summary(history)
        products = build_product_history_summary(history)
        targets = build_targets(future, interactions, users, products)
        segments = build_target_user_segments(targets, users)
        require(segments.height == targets["user_id"].n_unique(), "Evaluation user count mismatch")
        stages[name] = summarize_stage(history, future, interactions, targets, users, products, cutoff, end)
        tables.update({interaction_name: interactions, user_name: users, product_name: products,
                       f"{name}_targets.parquet": targets, f"{name}_user_segments.parquet": segments})

    require(sha256_file(input_path) == fingerprint, "Input changed during preparation")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.write_parquet(output_dir / name, compression="zstd", statistics=True)
        require(pl.read_parquet_schema(output_dir / name) == dict(table.schema), f"Schema mismatch: {name}")
        stored_rows = collect(pl.scan_parquet(output_dir / name).select(pl.len())).item()
        require(stored_rows == table.height, f"Output row count mismatch: {name}")

    report = {
        "input": str(input_path.resolve()), "input_sha256": fingerprint,
        "polars_version": pl.__version__, "input_quality": quality,
        "timezone": "UTC; naive event_at is interpreted as UTC, not local time",
        "task": "Fixed cutoff user-level prediction of purchases in the next evaluation window",
        "split": {
            "train": "2019-10-01 <= event_at < 2019-11-11",
            "validation": "2019-11-11 <= event_at < 2019-11-21",
            "test_history": "2019-10-01 <= event_at < 2019-11-21",
            "test": "2019-11-21 <= event_at < 2019-12-01",
        },
        "score": {"formula": "view_weight*ln(1+view_count)+cart_weight*cart_flag+purchase_weight*purchase_flag",
                  "view_weight": view_weight, "cart_weight": cart_weight,
                  "purchase_weight": purchase_weight, "status": "initial defaults, not validated as optimal"},
        "policy": {"low_activity_users_removed": False, "rare_products_removed": False,
                   "prior_purchase_targets_removed": False, "cold_start_targets_removed": False,
                   "full_period_brand_category_features_used": False},
        "validation": stages["validation"], "test": stages["test"],
        "outputs": {name: {"rows": table.height, "bytes": (output_dir / name).stat().st_size}
                    for name, table in tables.items()},
        "checks": "History/target counts, unique keys, temporal boundaries, stored row counts and schemas passed",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (output_dir / "model_data_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return report


def main() -> int:
    args = parse_args()
    try:
        report = prepare(args.input, args.output_dir, args.view_weight,
                         args.cart_weight, args.purchase_weight, args.overwrite)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Model data written to: {args.output_dir}")
    for name, output in report["outputs"].items():
        print(f"  {name}: {output['rows']:,} rows")
    print(f"Checks passed. Elapsed: {report['elapsed_seconds']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())