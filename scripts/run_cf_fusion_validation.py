"""Compare fixed Item-CF/User-CF rank fusion with separate models on validation only."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
import scipy

from models.cf_common import RECOMMENDATION_SCHEMA, SparseInteractions
from models.fusion import EVIDENCE, reciprocal_rank_pool, recommend_fused
from models.item_cf import ItemCF
from models.metrics import evaluate_users, history_exclusions, make_truth
from models.popularity import PopularityRecommender
from models.user_cf import UserCF
from scripts.run_cf_validation import ITEM_CF_PRESETS, group_metrics, load_validation
from scripts.run_popularity_baseline import INPUT_NAMES, VALIDATION_END, VALIDATION_START, digest, require

POLICY = "exclude_cart_and_purchase"
NAMES = {"popularity_purchase": "购买热门", "popularity_interaction": "综合行为热门",
         "item_cf_n100": "改进版 Item-CF", "user_cf_n100": "User-CF", "rrf_fusion": "两路融合（RRF）"}


def candidate_diagnostics(pool, truth):
    pools = {}
    for user, product, item_rank, user_rank in pool.select(
        "user_id", "product_id", "item_cf_rank", "user_cf_rank"
    ).iter_rows():
        by_product = pools.setdefault(user, {})
        by_product[product] = (item_rank is not None, user_rank is not None)
    buckets = Counter({name: 0 for name in ("both", "item_only", "user_only", "neither")})
    recall = Counter({name: 0. for name in ("item_cf", "user_cf", "union")})
    for user, answers in truth.items():
        candidates = pools.get(user, {})
        groups = Counter()
        for item in answers:
            item_cf, user_cf = candidates.get(item, (False, False))
            groups["both" if item_cf and user_cf else "item_only" if item_cf else "user_only" if user_cf else "neither"] += 1
        buckets.update(groups)
        for name, misses in (("item_cf", groups["user_only"]), ("user_cf", groups["item_only"]), ("union", 0)):
            recall[name] += (len(answers) - misses - groups["neither"]) / len(answers)
    return {"answer_pairs_by_source": dict(buckets), "macro_candidate_recall": {k: v / len(truth) for k, v in recall.items()},
            "unique_candidate_rows": pool.height, "users_with_personalized_candidates": len(pools)}


def evaluate_split(context, *, k, neighbors, candidate_depth, rank_constant, preset, quiet=False):
    require(0 < k <= candidate_depth and rank_constant >= 1, "Invalid recommendation depth or RRF constant")
    history = context["interactions"]
    blocked = history_exclusions(history, POLICY)
    truth, selection = make_truth(context["targets"], blocked)
    require(bool(truth), "No eligible purchase labels")
    data = SparseInteractions(history)
    catalog = set(data.product_ids.tolist())
    popularity = PopularityRecommender("purchase").fit(history)
    head = popularity.head_products(.01)
    empty_fallback = popularity.ranking.head(0)
    predictions, metrics, users_metrics = [], [], []

    def assess(name, rows):
        require(all(item not in blocked.get(user, ()) for user, item in rows.select("user_id", "product_id").iter_rows()),
                "A blocked item was recommended")
        per_user = evaluate_users(rows, truth, catalog, context["user_info"], k)
        grouped = group_metrics(per_user, rows, catalog, head, k)
        metrics.extend({"model": name, **row} for row in grouped)
        predictions.append(rows.with_columns(pl.lit(name).alias("model")))
        users_metrics.append(per_user.with_columns(pl.lit(name).alias("model")))
        if not quiet:
            print(f"{name}: Recall@{k}={grouped[0]['recall_at_k']:.4%}, "
                  f"NDCG@{k}={grouped[0]['ndcg_at_k']:.6f}", flush=True)

    for mode in ("purchase", "interaction"):
        model = popularity if mode == "purchase" else PopularityRecommender(mode).fit(history)
        assess(f"popularity_{mode}", model.recommend(truth, k, blocked).with_columns(
            pl.lit("global_popularity").alias("source"),
            *(pl.lit(None, dtype=RECOMMENDATION_SCHEMA[col]).alias(col) for col in
              ("reason_product_id", "reason_user_id", "reason_similarity"))))

    item = ItemCF(data, max_neighbors=neighbors, **preset).fit()
    assess("item_cf_n100", item.recommend(truth, neighbors, k, blocked, popularity.ranking))
    item_candidates = item.recommend(truth, neighbors, candidate_depth, blocked, empty_fallback)
    del item
    user = UserCF(data, max_neighbors=neighbors).fit().prepare_neighbors(truth)
    assess("user_cf_n100", user.recommend(truth, neighbors, k, blocked, popularity.ranking))
    user_candidates = user.recommend(truth, neighbors, candidate_depth, blocked, empty_fallback)
    del user
    pool = reciprocal_rank_pool(item_candidates, user_candidates, rank_constant)
    require(all(item not in blocked.get(user, ()) for user, item in pool.select("user_id", "product_id").iter_rows()),
            "A blocked item entered the fusion pool")
    fused = recommend_fused(pool, truth, k, blocked, popularity.ranking)
    assess("rrf_fusion", fused)

    evaluation = pl.concat(users_metrics)
    by_user = evaluation.filter(pl.col("model") == "user_cf_n100").select("user_id", pl.col("hits").alias("user_hits"))
    paired = evaluation.filter(pl.col("model") == "rrf_fusion").join(by_user, on="user_id", validate="1:1")
    known_truth = {user: answers for user, answers in truth.items() if context["user_info"][user]["known_user"]}
    paired_counts = {
        "fusion_only_users": paired.filter((pl.col("hits") > 0) & (pl.col("user_hits") == 0)).height,
        "user_cf_only_users": paired.filter((pl.col("hits") == 0) & (pl.col("user_hits") > 0)).height,
    }
    scores = {row["model"]: row for row in metrics if row["user_group"] == "overall"}
    summary = {
        "selection": selection, "overall_results": scores,
        "known_user_results": {row["model"]: row for row in metrics if row["user_group"] == "known_users"},
        "candidate_pool": candidate_diagnostics(pool, truth),
        "known_user_candidate_pool": candidate_diagnostics(pool.filter(pl.col("user_id").is_in(list(known_truth))), known_truth) if known_truth else None,
        "fusion_vs_user_cf": paired_counts,
    }
    return summary, pl.DataFrame(metrics), pool, pl.concat(predictions, how="diagonal").select(
        *RECOMMENDATION_SCHEMA, *EVIDENCE, "model"), evaluation


def write_evaluation(output_dir, metrics, pool, predictions, evaluation):
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics.write_csv(output_dir / "metrics_summary.csv", include_bom=True)
    pool.write_parquet(output_dir / "candidate_pool.parquet", compression="zstd")
    predictions.write_parquet(output_dir / "recommendations.parquet", compression="zstd")
    evaluation.write_parquet(output_dir / "user_metrics.parquet", compression="zstd")


def run(model_dir: Path, output_dir: Path, candidate_depth=50, rank_constant=60, quiet=False):
    require(not output_dir.exists(), "Output directory exists; choose a new --output-dir")
    k, neighbors = 10, 100
    preset = ITEM_CF_PRESETS["shrinkage_iuf"]
    context = load_validation(model_dir)
    summary, *tables = evaluate_split(context, k=k, neighbors=neighbors, candidate_depth=candidate_depth,
                                      rank_constant=rank_constant, preset=preset, quiet=quiet)
    report = {
        "split": "validation_only", "test_evaluated": False, "target_policy": POLICY,
        "history_end_exclusive_utc": str(VALIDATION_START), "answer_end_exclusive_utc": str(VALIDATION_END),
        "input_sha256": context["hashes"], "versions": {"polars": pl.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "k": k, "neighbors": neighbors, "candidate_depth_per_channel": candidate_depth, "rank_constant": rank_constant,
        "model_definition": {"item_cf": preset, "fusion": "sum of 1/(rank_constant + rank), equal weights, ties by best channel rank then product ID",
                             "fallback": "purchase-popularity only after fusion and never as a source vote"},
        **summary,
    }
    require(context["hashes"] == {name: digest(model_dir / name) for name in INPUT_NAMES}, "Validation inputs changed")
    write_evaluation(output_dir, *tables)
    (output_dir / "run_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "RESULTS.md").write_text(render_report(report), encoding="utf-8")
    return report


def render_report(report):
    scores, known = report["overall_results"], report["known_user_results"]
    cohort = report["selection"]
    diagnostic = report["candidate_pool"]
    pairs = diagnostic["answer_pairs_by_source"]
    candidate_recall = diagnostic["macro_candidate_recall"]
    fused, user = scores["rrf_fusion"], scores["user_cf_n100"]
    delta = 100 * (fused["recall_at_k"] - user["recall_at_k"])
    choice = ("融合在本轮验证 Recall 高于单路 User-CF；仍需独立测试。" if delta > 0 else
              "融合没有超过 User-CF 的验证 Recall；不宜仅凭合并候选更多就替换现有默认模型。")
    lines = ["# 两路 CF 融合：验证结果", "",
             f"验证期为 2019 年 11 月 11 日至 20 日（UTC），共 {cohort['evaluated_users']:,} 位用户、"
             f"{cohort['eligible_target_pairs']:,} 组购买答案。历史已购和已加购商品同时从推荐与评估答案中排除。未读取最终测试集。",
             "", f"Item-CF 和 User-CF 邻居数固定为 {report['neighbors']}，各取最多 {report['candidate_depth_per_channel']} 个纯个性化候选。"
             f"合并去重后，对每个来源加 `1 / ({report['rank_constant']} + 名次)`，取分数最高的 10 件；"
             "候选不足再以购买热门补齐。两个模型的原始分数不参与融合。", "",
             "## Top 10 对比", "",
             "| 模型 | Recall@10 | NDCG@10 | 有历史用户 Recall@10 | 商品覆盖率 | 头部推荐位 | 热门补齐位 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in scores.items():
        history_recall = f"{known[name]['recall_at_k']:.2%}" if name in known else "—"
        lines.append(f"| {NAMES[name]} | {row['recall_at_k']:.2%} | {row['ndcg_at_k']:.4f} | {history_recall} | "
                     f"{row['catalog_coverage']:.3%} | {row['head_slot_share']:.2%} | {row['fallback_slot_share']:.2%} |")
    paired = report["fusion_vs_user_cf"]
    lines += ["", f"融合相对 User-CF 的 Recall@10 差值：**{delta:+.2f} 个百分点**。"
              f"{paired['fusion_only_users']:,} 位用户仅融合命中，{paired['user_cf_only_users']:,} 位用户仅 User-CF 命中。{choice}",
              "", "## 候选池诊断（不含热门补齐）", "",
              f"Item-CF / User-CF / 两路并集的候选 Recall 分别为 **{candidate_recall['item_cf']:.2%}** / "
              f"**{candidate_recall['user_cf']:.2%}** / **{candidate_recall['union']:.2%}**。"
              "候选 Recall 衡量购买商品有没有进入候选池，不是最终 Top 10 Recall。",
              "", f"按去重的购买用户－商品对计数：两路都找到 {pairs['both']:,}，仅 Item-CF 找到 {pairs['item_only']:,}，"
              f"仅 User-CF 找到 {pairs['user_only']:,}，两路都没找到 {pairs['neither']:,}。",
              "", "## 文件与复现", "",
              "- `candidate_pool.parquet` 保留去重候选、两路名次、两路推荐理由和 RRF 分数，可用于后续排序实验。",
              "- `recommendations.parquet` 保存各模型完整 Top 10；融合分数和热门补齐分数不是购买概率。",
              "- `metrics_summary.csv` 记录总体和用户分层指标；`user_metrics.parquet` 记录逐用户指标；"
              "`run_manifest.json` 保留配置与输入哈希。", "",
              "```powershell", r"Set-Location D:\user_behavior",
              (r".\.venv\Scripts\python.exe -B scripts\run_cf_fusion_validation.py"
               f" --candidate-depth {report['candidate_depth_per_channel']} --rank-constant {report['rank_constant']}"
               r" --output-dir reports\cf_fusion_validation_rerun"),
              "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data-dir", type=Path, default=ROOT / "model_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "cf_fusion_validation")
    parser.add_argument("--candidate-depth", type=int, default=50)
    parser.add_argument("--rank-constant", type=int, default=60)
    args = parser.parse_args()
    try:
        run(args.model_data_dir, args.output_dir, args.candidate_depth, args.rank_constant)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved validation-only comparison to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
