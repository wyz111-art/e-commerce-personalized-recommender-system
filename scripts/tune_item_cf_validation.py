"""Scan a small Item-CF grid using frozen training history and validation only."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from itertools import product
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
import scipy

from models.cf_common import RECOMMENDATION_SCHEMA, SparseInteractions
from models.item_cf import ItemCF
from models.metrics import evaluate_users, history_exclusions, make_truth
from models.popularity import PopularityRecommender
from scripts.run_cf_validation import group_metrics, load_validation
from scripts.run_popularity_baseline import INPUT_NAMES, VALIDATION_END, VALIDATION_START, digest, require


@dataclass(frozen=True)
class Config:
    min_common_users: int = 1
    shrinkage: float = 0.
    iuf: bool = False
    popularity_alpha: float = 0.

    @property
    def name(self):
        return (f"item_cf_m{self.min_common_users}_s{self.shrinkage:g}"
                f"_iuf{int(self.iuf)}_a{self.popularity_alpha:g}")


def run(model_dir: Path, output_dir: Path, neighbors=100, k=10, block_size=64, quiet=False):
    require(not output_dir.exists(), "Output directory exists; choose a new --output-dir")
    require(neighbors > 0 and k > 0 and block_size > 0, "Neighbors, k and block size must be positive")
    started = time.perf_counter()
    context = load_validation(model_dir)
    interactions = context["interactions"]
    data = SparseInteractions(interactions)
    policy = "new_purchase"
    blocked = history_exclusions(interactions, policy)
    truth, selection = make_truth(context["targets"], blocked)
    require(bool(truth), "No remaining validation purchase targets")
    catalog = set(data.product_ids.tolist())
    purchase = PopularityRecommender("purchase").fit(interactions)
    head = purchase.head_products(.01)
    metrics, runs = [], {}
    best = None
    best_key = None
    best_predictions = None
    by_config = {}

    def assess(name, predictions, stage, config, started, fit_seconds=None, edges=None):
        require(all(item not in blocked.get(user, ()) for user, item in
                    predictions.select("user_id", "product_id").iter_rows()), "Blocked item recommended")
        per_user = evaluate_users(predictions, truth, catalog, context["user_info"], k)
        groups = group_metrics(per_user, predictions, catalog, head, k)
        meta = {"model": name, "stage": stage, **(asdict(config) if config is not None
                else dict.fromkeys(asdict(Config())))}
        metrics.extend({**meta, **row} for row in groups)
        overall = groups[0]
        runs[name] = {**meta, "fit_seconds": fit_seconds, "neighbor_edges": edges,
                      "elapsed_seconds": round(time.perf_counter() - started, 3)}
        if not quiet:
            print(f"{stage} {name}: recall={overall['recall_at_k']:.4%}, "
                  f"coverage={overall['catalog_coverage']:.3%}, "
                  f"fallback={overall['fallback_slot_share']:.2%}", flush=True)
        return overall["recall_at_k"], overall["ndcg_at_k"]

    for mode in ("purchase", "interaction"):
        start = time.perf_counter()
        model = purchase if mode == "purchase" else PopularityRecommender(mode).fit(interactions)
        predictions = model.recommend(truth, k, blocked).with_columns(
            pl.lit("global_popularity").alias("source"),
            *(pl.lit(None, dtype=RECOMMENDATION_SCHEMA[col]).alias(col)
              for col in ("reason_product_id", "reason_user_id", "reason_similarity")),
        )
        assess(f"popularity_{mode}", predictions, "baseline", None, start)

    def score(config, stage):
        nonlocal best, best_key, best_predictions
        if config in by_config:
            return
        start = time.perf_counter()
        model = ItemCF(data, neighbors, block_size, **asdict(config)).fit()
        fit_seconds = round(time.perf_counter() - start, 3)
        predictions = model.recommend(truth, neighbors, k, blocked, purchase.ranking)
        key = assess(config.name, predictions, stage, config, start, fit_seconds, model.graph.nnz)
        # Retain the earlier, simpler configuration if both metrics tie.
        by_config[config] = key
        if best_key is None or key > best_key:
            best, best_key, best_predictions = config, key, predictions

    for support, shrinkage in product((1, 2, 3), (0., 5., 15.)):
        score(Config(support, shrinkage), "support_shrinkage")
    support_winner = best
    for base in (Config(), support_winner):
        score(replace(base, iuf=True), "iuf")
    iuf_winner = best
    for alpha in (.1, .3):
        score(replace(iuf_winner, popularity_alpha=alpha), "popularity_penalty")

    require(context["hashes"] == {name: digest(model_dir / name) for name in INPUT_NAMES},
            "Input files changed during the scan")
    report = {
        "split": "validation_only", "test_evaluated": False,
        "input_dir": str(model_dir.resolve()), "input_sha256": context["hashes"],
        "versions": {"polars": pl.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "history_end_exclusive_utc": str(VALIDATION_START),
        "answer_end_exclusive_utc": str(VALIDATION_END),
        "target_policy": policy, "k": k, "neighbors": neighbors, "block_size": block_size,
        "selection": selection,
        "matrix": {"users": data.matrix.shape[0], "products": data.matrix.shape[1],
                   "nonzeros": data.matrix.nnz},
        "best_selection": "overall macro Recall, then NDCG, then first evaluated",
        "support_grid_winner": support_winner.name, "pre_penalty_winner": iuf_winner.name,
        "best_item_cf": {"model": best.name, **asdict(best)},
        "runs": list(runs.values()),
        "overall_results": [row for row in metrics if row["user_group"] == "overall"],
        "known_user_results": [row for row in metrics if row["user_group"] == "known_users"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    pl.DataFrame(metrics, infer_schema_length=None).write_csv(output_dir / "metrics_summary.csv", include_bom=True)
    best_predictions.with_columns(pl.lit(best.name).alias("model")).write_parquet(
        output_dir / "best_item_cf_recommendations.parquet", compression="zstd")
    (output_dir / "run_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "RESULTS.md").write_text(render_report(report), encoding="utf-8")
    return report


def render_report(report):
    k = report["k"]
    best = report["best_item_cf"]["model"]
    old = Config().name
    overall = {row["model"]: row for row in report["overall_results"]}
    known = {row["model"]: row for row in report["known_user_results"]}
    popular = max(("popularity_purchase", "popularity_interaction"),
                  key=lambda name: overall[name]["recall_at_k"])
    title = {"popularity_purchase": "购买热门", "popularity_interaction": "综合行为热门",
             old: "原版 Item-CF"}

    def table(rows):
        lines = [f"| 配置 | Recall@{k} | NDCG@{k} | 至少命中一件 | 商品覆盖率 | 头部推荐位 | 热门补齐位 |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            name = row["model"]
            label = title.get(name, f"`{name}`") + ("（最优 Item-CF）" if name == best else "")
            lines.append(f"| {label} | {row['recall_at_k']:.2%} | {row['ndcg_at_k']:.4f} | "
                         f"{row['hit_rate_at_k']:.2%} | {row['catalog_coverage']:.3%} | "
                         f"{row['head_slot_share']:.2%} | {row['fallback_slot_share']:.2%} |")
        return lines

    delta_old = (overall[best]["recall_at_k"] - overall[old]["recall_at_k"]) * 100
    delta_popular = (overall[best]["recall_at_k"] - overall[popular]["recall_at_k"]) * 100
    delta_fallback = (overall[best]["fallback_slot_share"] - overall[old]["fallback_slot_share"]) * 100
    lines = ["# Item-CF 改进：验证集结果", "",
             "历史只使用 2019 年 11 月 11 日之前的数据；购买答案来自 11 月 11 日至 20 日（UTC）。"
             "未读取或评估最终测试集。", "",
             f"共评价 {report['selection']['evaluated_users']:,} 位用户和 "
             f"{report['selection']['eligible_target_pairs']:,} 个购买答案。使用 `new_purchase`：排除历史已购买商品，"
             "历史浏览、加购但未购买的商品仍可推荐。训练中没出现的答案不从分母剔除。", "",
             "## 总体对照", "", *table(overall.values()), "",
             f"- 按总体 Recall 优先、NDCG 次之，扫描中最好的 Item-CF 是 `{best}`。"
             f"其 Recall 对原版差 {delta_old:+.2f} 个百分点，对较强热门基线差 {delta_popular:+.2f} 个百分点。",
             f"- 相比原版，热门补齐位占比变化 {delta_fallback:+.2f} 个百分点。"
             "这是 Item-CF 加热门补齐后的整体指标，不能将所有命中都归因给个性化。", "",
             "## 有训练历史的用户", ""]
    choices = dict.fromkeys(("popularity_purchase", "popularity_interaction", old, best))
    if known:
        lines += table(known[name] for name in choices)
    else:
        lines.append("该验证样本没有有历史用户。")
    lines += ["", "## 如何比较", "",
              "- `m` 为最低共同用户数，`s` 为共现收缩，`iuf` 为活跃用户降权开关，`a` 为热门商品降权强度。",
              "- 头部商品定义为训练商品库中购买热门榜前 1%；覆盖率是推荐过的去重商品数占训练商品库比例。",
              "- 热门补齐位只统计 Item-CF 的回退位，不把单独的热门榜当作补齐。",
              "- 参数是在同一个验证集选出的，结果不是独立测试表现，也不能证明线上购买率提升。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data-dir", type=Path, default=ROOT / "model_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "itemcf_tuning_validation")
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=64)
    args = parser.parse_args()
    try:
        report = run(args.model_data_dir, args.output_dir, args.neighbors, args.k, args.block_size)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {len(report['runs'])} validation runs to {args.output_dir}. Test was not evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
