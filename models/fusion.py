"""Equal-weight reciprocal rank fusion for Item-CF and User-CF candidates."""
from __future__ import annotations

from collections import defaultdict

import polars as pl

from .cf_common import make_recommendations

KEYS = ["user_id", "product_id"]
EVIDENCE = {
    "item_cf_rank": pl.Int64, "item_cf_seed_product_id": pl.Int64, "item_cf_similarity": pl.Float64,
    "user_cf_rank": pl.Int64, "user_cf_peer_user_id": pl.Int64, "user_cf_similarity": pl.Float64,
}
ALS_EVIDENCE = {"als_rank": pl.Int64, "als_score": pl.Float64}


def reciprocal_rank_pool(item_candidates: pl.DataFrame, user_candidates: pl.DataFrame,
                         rank_constant: int = 60, *, als_candidates: pl.DataFrame | None = None) -> pl.DataFrame:
    """Rank the de-duplicated pool without voting for popular fallback rows."""
    if rank_constant < 1:
        raise ValueError("rank_constant must be positive")

    def channel(data, name, *evidence):
        data = data.filter(pl.col("source") == name).select(
            *KEYS, pl.col("rank").alias(f"{name}_rank"), *evidence,
        )
        rank = f"{name}_rank"
        if data[rank].null_count() or (data.height and data[rank].min() < 1):
            raise ValueError("Candidate ranks must be positive")
        if data.select(*KEYS).is_duplicated().any() or data.select("user_id", rank).is_duplicated().any():
            raise ValueError("Duplicate product or rank within a channel")
        return data

    item = channel(item_candidates, "item_cf", pl.col("reason_product_id").alias("item_cf_seed_product_id"),
                   pl.col("reason_similarity").alias("item_cf_similarity"))
    user = channel(user_candidates, "user_cf", pl.col("reason_user_id").alias("user_cf_peer_user_id"),
                   pl.col("reason_similarity").alias("user_cf_similarity"))
    pool = item.join(user, on=KEYS, how="full", coalesce=True)
    if als_candidates is not None:
        pool = pool.join(channel(als_candidates, "als", pl.col("score").alias("als_score")),
                         on=KEYS, how="full", coalesce=True)
    return rank_pool(pool, rank_constant)


def rank_pool(pool: pl.DataFrame, rank_constant: int) -> pl.DataFrame:
    if rank_constant < 1:
        raise ValueError("rank_constant must be positive")
    ranks = [pl.col(col) for col in ("item_cf_rank", "user_cf_rank", "als_rank") if col in pool.columns]
    return (pool
            .with_columns(
                pl.sum_horizontal(*[(1. / (rank_constant + rank)).fill_null(0.) for rank in ranks]).alias("rrf_score"),
                pl.min_horizontal(*ranks).alias("best_source_rank"),
            )
            .sort(["user_id", "rrf_score", "best_source_rank", "product_id"], descending=[False, True, False, False])
            .drop("best_source_rank"))


def recommend_fused(pool, users, k, blocked, popularity_ranking, *, source="rrf_fusion"):
    candidates = defaultdict(list)
    for user, item, score in pool.select(*KEYS, "rrf_score").iter_rows():
        if item not in blocked.get(user, ()) and len(candidates[user]) < k:
            candidates[user].append({"product_id": item, "score": score,
                                     "reason_product_id": None, "reason_user_id": None,
                                     "reason_similarity": None})
    recs = make_recommendations(users, k, blocked, popularity_ranking, source,
                                lambda user, _blocked, _k: candidates[user])
    evidence = [col for col in (*EVIDENCE, *ALS_EVIDENCE) if col in pool.columns]
    return (recs.join(pool.select(*KEYS, *evidence), on=KEYS, how="left", validate="1:1")
            .sort(["user_id", "rank"]))
