"""Deterministic popularity baselines using historical interactions only."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math

import polars as pl


class PopularityRecommender:
    def __init__(self, mode: str = "purchase"):
        if mode not in {"purchase", "interaction"}:
            raise ValueError("mode must be purchase or interaction")
        self.mode = mode
        self.ranking: pl.DataFrame | None = None

    def fit(self, interactions: pl.DataFrame) -> "PopularityRecommender":
        required = {"user_id", "product_id", "purchase_count", "interaction_score"}
        if not required <= set(interactions.columns):
            raise ValueError("Interaction table lacks required model columns")
        score = "purchase_events" if self.mode == "purchase" else "interaction_strength"
        self.ranking = (
            interactions.group_by("product_id").agg(
                pl.col("purchase_count").sum().cast(pl.Float64).alias("purchase_events"),
                pl.col("interaction_score").sum().alias("interaction_strength"),
                pl.col("user_id").n_unique().alias("interaction_users"),
            ).sort([score, "interaction_users", "product_id"], descending=[True, True, False])
            .select("product_id", pl.col(score).alias("score"))
            .with_row_index("catalog_rank", offset=1)
        )
        if not self.ranking["score"].is_finite().all() or (self.ranking["score"] < 0).any():
            raise ValueError("Popularity scores must be finite and non-negative")
        return self

    def head_products(self, fraction: float = 0.01) -> set[int]:
        if self.ranking is None:
            raise RuntimeError("Call fit first")
        if not 0 < fraction <= 1:
            raise ValueError("Head fraction must be in (0, 1]")
        n = math.ceil(self.ranking.height * fraction)
        return set(self.ranking.head(n)["product_id"].to_list())

    def recommend(self, users: Iterable[int], k: int = 10,
                  blocked: Mapping[int, set[int]] | None = None) -> pl.DataFrame:
        if self.ranking is None:
            raise RuntimeError("Call fit first")
        if k <= 0:
            raise ValueError("k must be positive")
        catalog = list(self.ranking.select("product_id", "score").iter_rows())
        records = []
        for user in sorted(set(users)):
            exclusions = (blocked or {}).get(user, set())
            rank = 0
            for product, score in catalog:
                if product in exclusions:
                    continue
                rank += 1
                records.append((int(user), int(product), rank, float(score)))
                if rank == k:
                    break
        return pl.DataFrame(records, schema={
            "user_id": pl.Int64, "product_id": pl.Int64,
            "rank": pl.Int64, "score": pl.Float64,
        }, orient="row")