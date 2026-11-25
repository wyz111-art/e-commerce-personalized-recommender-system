"""Build a compact static data bundle for the Chinese recommendation workbench."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl

MODELS = ["user_cf_n100", "item_cf_n100", "rrf_fusion"]
MODEL_LABELS = {"user_cf_n100": "User-CF", "item_cf_n100": "改进版 Item-CF", "rrf_fusion": "两路融合"}


def iso(value):
    return value.isoformat(sep=" ") if value is not None else None


def rounded(value):
    return round(float(value), 6) if value is not None else None


def build(model_data_dir: Path, report_dir: Path, output: Path, recent_limit: int = 12):
    recs = pl.read_parquet(report_dir / "recommendations.parquet").filter(pl.col("model").is_in(MODELS))
    outcomes = pl.read_parquet(report_dir / "purchase_outcomes.parquet")
    history = pl.read_parquet(model_data_dir / "test_history_interactions.parquet")
    users = pl.read_parquet(model_data_dir / "test_user_segments.parquet")
    metrics = pl.read_csv(report_dir / "metrics_summary.csv")
    per_user = pl.read_parquet(report_dir / "user_metrics.parquet").filter(pl.col("model").is_in(MODELS))
    user_ids = sorted(recs["user_id"].unique().to_list())
    history = history.filter(pl.col("user_id").is_in(user_ids)).sort(["user_id", "last_interaction_at", "product_id"], descending=[False, True, False])
    users = users.filter(pl.col("user_id").is_in(user_ids))
    outcome_map = {(int(row["user_id"]), int(row["product_id"])): row for row in outcomes.iter_rows(named=True)}

    history_map = {}
    for key, rows in history.group_by("user_id", maintain_order=True):
        history_map[str(int(key[0]))] = [{
            "product_id": int(row["product_id"]), "event_count": int(row["event_count"]),
            "view_count": int(row["view_count"]), "cart_count": int(row["cart_count"]),
            "purchase_count": int(row["purchase_count"]), "session_count": int(row["session_count"]),
            "last_interaction_at": iso(row["last_interaction_at"]), "interaction_score": rounded(row["interaction_score"]),
        } for row in rows.head(recent_limit).iter_rows(named=True)]

    user_stats = {str(int(row["user_id"])): {
        "event_count": int(row["event_count"]), "session_count": int(row["session_count"]),
        "unique_product_count": int(row["unique_product_count"]), "active_day_count": int(row["active_day_count"]),
        "view_count": int(row["view_count"]), "cart_count": int(row["cart_count"]),
        "purchase_count": int(row["purchase_count"]), "activity_segment": row["activity_segment"],
    } for row in users.iter_rows(named=True)}

    recommendation_map = {}
    for key, rows in recs.sort(["user_id", "model", "rank"]).group_by(["user_id", "model"], maintain_order=True):
        user, model = str(int(key[0])), key[1]
        recommendation_map.setdefault(user, {})[model] = [{
            "rank": int(row["rank"]), "product_id": int(row["product_id"]), "score": rounded(row["score"]),
            "source": row["source"], "reason_product_id": int(row["reason_product_id"]) if row["reason_product_id"] is not None else None,
            "reason_user_id": int(row["reason_user_id"]) if row["reason_user_id"] is not None else None,
            "reason_similarity": rounded(row["reason_similarity"]),
            "item_cf_rank": int(row["item_cf_rank"]) if row["item_cf_rank"] is not None else None,
            "user_cf_rank": int(row["user_cf_rank"]) if row["user_cf_rank"] is not None else None,
            "is_actual_purchase": bool(outcome_map.get((int(row["user_id"]), int(row["product_id"])), {}).get("eligible_for_evaluation", False)),
        } for row in rows.iter_rows(named=True)]

    purchase_map = {}
    for key, rows in outcomes.filter(pl.col("user_id").is_in(user_ids)).sort(["user_id", "first_future_purchase_at", "product_id"]).group_by("user_id", maintain_order=True):
        purchase_map[str(int(key[0]))] = [{
            "product_id": int(row["product_id"]), "first_future_purchase_at": iso(row["first_future_purchase_at"]),
            "prior_purchase": bool(row["prior_purchase"]), "prior_cart": bool(row["prior_cart"]),
            "in_train_catalog": bool(row["in_train_catalog"]), "eligible_for_evaluation": bool(row["eligible_for_evaluation"]),
        } for row in rows.iter_rows(named=True)]

    metric_map = {}
    for row in metrics.filter(pl.col("model").is_in(MODELS) & pl.col("user_group").is_in(["overall", "known_users"])).iter_rows(named=True):
        metric_map.setdefault(row["model"], {})[row["user_group"]] = {
            "recall_at_k": rounded(row["recall_at_k"]), "ndcg_at_k": rounded(row["ndcg_at_k"]),
            "hit_rate_at_k": rounded(row["hit_rate_at_k"]), "catalog_coverage": rounded(row["catalog_coverage"]),
            "fallback_slot_share": rounded(row["fallback_slot_share"]),
        }
    user_metric_map = {}
    for row in per_user.iter_rows(named=True):
        user_metric_map.setdefault(str(int(row["user_id"])), {})[row["model"]] = {
            "hits": int(row["hits"]), "recall_at_k": rounded(row["recall_at_k"]), "ndcg_at_k": rounded(row["ndcg_at_k"]),
        }
    examples = [str(user) for user in [565270076, 570020065, 572480049, 560390005] if str(user) in recommendation_map]
    data = {
        "version": 1, "split": "final_test", "timezone": "UTC", "history_cutoff": "2019-11-21 00:00:00",
        "answer_window": "2019-11-21 至 2019-11-30", "models": [{"id": model, "label": MODEL_LABELS[model]} for model in MODELS],
        "users": [{"user_id": user, **user_stats.get(user, {}), "history_count": len(history_map.get(user, []))} for user in sorted(recommendation_map)],
        "history": history_map, "recommendations": recommendation_map, "purchases": purchase_map,
        "metrics": metric_map, "user_metrics": user_metric_map, "examples": examples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {output} ({output.stat().st_size / 1024 / 1024:.1f} MB, {len(recommendation_map):,} users)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data-dir", type=Path, default=ROOT / "model_data")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "cf_final_test")
    parser.add_argument("--output", type=Path, default=ROOT / "ui" / "data" / "recommender_data.json")
    parser.add_argument("--recent-limit", type=int, default=12)
    args = parser.parse_args()
    build(args.model_data_dir, args.report_dir, args.output, args.recent_limit)


if __name__ == "__main__":
    main()
