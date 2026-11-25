"""Shared, binary-relevance Top-K evaluation for popularity, Item-CF and User-CF.

Recall/NDCG/MRR/HitRate are macro-averages over users with nonempty eligible
purchase labels. Cold-start products remain in the main relevance denominator.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import math

import polars as pl


POLICIES = {"new_purchase", "exclude_cart_and_purchase", "unseen", "all"}


def history_exclusions(interactions: pl.DataFrame, policy: str = "new_purchase") -> dict[int, set[int]]:
    if policy not in POLICIES:
        raise ValueError(f"Unknown target policy: {policy}")
    if policy == "all":
        return {}
    blocked: dict[int, set[int]] = defaultdict(set)
    purchased = pl.col("purchase_count") > 0
    if policy == "new_purchase":
        source = interactions.filter(purchased)
    elif policy == "exclude_cart_and_purchase":
        source = interactions.filter(purchased | (pl.col("cart_count") > 0))
    else:
        source = interactions
    for user, product in source.select("user_id", "product_id").iter_rows():
        blocked[int(user)].add(int(product))
    return dict(blocked)


def make_truth(targets: pl.DataFrame, blocked: Mapping[int, set[int]]) -> tuple[dict[int, set[int]], dict]:
    all_truth: dict[int, set[int]] = defaultdict(set)
    truth: dict[int, set[int]] = defaultdict(set)
    for user, product in targets.select("user_id", "product_id").iter_rows():
        user, product = int(user), int(product)
        all_truth[user].add(product)
        if product not in blocked.get(user, set()):
            truth[user].add(product)
    report = {
        "original_purchase_users": len(all_truth),
        "original_target_pairs": sum(map(len, all_truth.values())),
        "evaluated_users": len(truth),
        "eligible_target_pairs": sum(map(len, truth.values())),
        "removed_target_pairs": sum(map(len, all_truth.values())) - sum(map(len, truth.values())),
        "removed_empty_label_users": len(all_truth) - len(truth),
    }
    return dict(truth), report


def evaluate_users(recommendations: pl.DataFrame, truth: Mapping[int, set[int]],
                   catalog: set[int], user_info: Mapping[int, dict], k: int = 10) -> pl.DataFrame:
    if k <= 0 or not truth or any(not products for products in truth.values()):
        raise ValueError("Need positive k and nonempty relevance sets")
    ordered: dict[int, list[int]] = defaultdict(list)
    for user, product, rank in recommendations.sort(["user_id", "rank"]).select(
        "user_id", "product_id", "rank"
    ).iter_rows():
        if user not in truth:
            raise ValueError("Recommendations contain an unevaluated user")
        if product not in catalog:
            raise ValueError("Recommendation is outside the historical catalog")
        if rank != len(ordered[user]) + 1:
            raise ValueError("Recommendation ranks must be contiguous and unique")
        if product in ordered[user] or rank > k:
            raise ValueError("Duplicate item or recommendation list longer than k")
        ordered[user].append(product)

    rows = []
    for user in sorted(truth):
        relevant = truth[user]
        recommended = ordered.get(user, [])
        hit_ranks = [rank for rank, item in enumerate(recommended, 1) if item in relevant]
        dcg = math.fsum(1 / math.log2(rank + 1) for rank in hit_ranks)
        ideal = math.fsum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
        info = user_info[user]
        rows.append({
            "user_id": user,
            "activity_segment": info["activity_segment"],
            "known_user": bool(info["known_user"]),
            "relevant_items": len(relevant),
            "reachable_relevant_items": len(relevant & catalog),
            "recommended_items": len(recommended),
            "hits": len(hit_ranks),
            "recall_at_k": len(hit_ranks) / len(relevant),
            "precision_at_k": len(hit_ranks) / k,
            "ndcg_at_k": dcg / ideal,
            "mrr_at_k": 1 / hit_ranks[0] if hit_ranks else 0.0,
            "hit_rate_at_k": float(bool(hit_ranks)),
        })
    return pl.DataFrame(rows)


def aggregate_metrics(per_user: pl.DataFrame, recommendations: pl.DataFrame,
                      catalog: set[int], head: set[int], k: int) -> list[dict]:
    groups = [("overall", per_user)] + [
        (name, per_user.filter(pl.col("activity_segment") == name))
        for name in ["new_user", "single_session", "low_activity", "medium_activity", "high_activity"]
    ]
    rows = []
    for label, users in groups:
        if not users.height:
            continue
        displayed = recommendations.join(users.select("user_id"), on="user_id", how="semi")
        products = displayed["product_id"].to_list()
        head_slots = sum(product in head for product in products)
        positives = int(users["relevant_items"].sum())
        reachable = int(users["reachable_relevant_items"].sum())
        rows.append({
            "user_group": label, "k": k, "users": users.height,
            "relevant_pairs": positives, "reachable_relevant_pairs": reachable,
            "reachable_target_share": reachable / positives,
            "recall_at_k": users["recall_at_k"].mean(),
            "precision_at_k": users["precision_at_k"].mean(),
            "ndcg_at_k": users["ndcg_at_k"].mean(),
            "mrr_at_k": users["mrr_at_k"].mean(),
            "hit_rate_at_k": users["hit_rate_at_k"].mean(),
            "users_with_hits": int((users["hits"] > 0).sum()),
            "catalog_size": len(catalog), "unique_recommended_products": len(set(products)),
            "catalog_coverage": len(set(products)) / len(catalog) if catalog else 0.0,
            "recommendation_slots": len(products),
            "head_slot_share": head_slots / len(products) if products else 0.0,
            "non_head_slot_share": 1 - head_slots / len(products) if products else 0.0,
            "list_fill_rate": len(products) / (k * users.height),
        })
    return rows
