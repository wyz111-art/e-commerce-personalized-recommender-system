"""Measure the incremental value of ALS against a frozen two-channel validation run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import implicit
import polars as pl

from models.als import ImplicitALS
from models.cf_common import RECOMMENDATION_SCHEMA, SparseInteractions
from models.fusion import ALS_EVIDENCE, EVIDENCE, KEYS, reciprocal_rank_pool, recommend_fused
from models.metrics import evaluate_users, history_exclusions, make_truth
from models.popularity import PopularityRecommender
from scripts.run_cf_fusion_validation import NAMES, POLICY, candidate_diagnostics
from scripts.run_cf_validation import group_metrics, load_validation
from scripts.run_popularity_baseline import INPUT_NAMES, digest, require

THREE = "rrf_fusion_als"
LABELS = {**NAMES, "als": "ALS 单路", THREE: "三路融合（Item-CF + User-CF + ALS）"}


def recover_channel(pool, name):
    reason = "reason_product_id" if name == "item_cf" else "reason_user_id"
    evidence = "item_cf_seed_product_id" if name == "item_cf" else "user_cf_peer_user_id"
    return pool.filter(pl.col(f"{name}_rank").is_not_null()).select(
        *KEYS, pl.col(f"{name}_rank").alias("rank"), pl.lit(name).alias("source"),
        pl.col(evidence).alias(reason), pl.col(f"{name}_similarity").alias("reason_similarity"),
    )


def run(model_dir, fusion_dir, output_dir, *, als_options=None):
    require(not output_dir.exists(), "Output directory exists; choose a new --output-dir")
    sources = {name: fusion_dir / name for name in ("run_manifest.json", "candidate_pool.parquet", "recommendations.parquet")}
    source_hashes = {name: digest(path) for name, path in sources.items()}
    prior = json.loads(sources["run_manifest.json"].read_text(encoding="utf-8"))
    require(prior["split"] == "validation_only" and not prior["test_evaluated"] and prior["target_policy"] == POLICY,
            "Expected the discovery-policy two-channel validation run")
    context = load_validation(model_dir)
    require(prior["input_sha256"] == context["hashes"], "Validation inputs differ from the two-channel run")
    history = context["interactions"]
    blocked = history_exclusions(history, POLICY)
    truth, selection = make_truth(context["targets"], blocked)
    require(bool(truth) and selection == prior["selection"], "Different validation population")
    k, depth, constant = prior["k"], prior["candidate_depth_per_channel"], prior["rank_constant"]
    data = SparseInteractions(history)
    catalog = set(data.product_ids.tolist())
    popularity = PopularityRecommender("purchase").fit(history)
    head = popularity.head_products(.01)
    two = pl.read_parquet(sources["candidate_pool.parquet"])
    require(two.height == two.select(*KEYS).n_unique(), "Duplicate two-channel candidates")
    for uid, item in two.select(*KEYS).iter_rows():
        require(uid in truth and item in catalog and item not in blocked.get(uid, ()), "Ineligible two-channel candidate")

    metrics, predictions, per_user = [], [], []

    def assess(name, rows):
        require(all(item not in blocked.get(uid, ()) for uid, item in rows.select(*KEYS).iter_rows()),
                "Ineligible recommendation")
        evaluated = evaluate_users(rows, truth, catalog, context["user_info"], k)
        grouped = group_metrics(evaluated, rows, catalog, head, k)
        metrics.extend({"model": name, **row} for row in grouped)
        per_user.append(evaluated.with_columns(pl.lit(name).alias("model")))
        predictions.append(rows.with_columns(pl.lit(name).alias("model")))
        return grouped[0]

    previous_recs = pl.read_parquet(sources["recommendations.parquet"])
    require(set(previous_recs["model"]) == set(NAMES), "Incomplete comparison lists")
    for name in NAMES:
        row = assess(name, previous_recs.filter(pl.col("model") == name))
        require(all(abs(row[field] - prior["overall_results"][name][field]) < 1e-12
                    for field in ("recall_at_k", "ndcg_at_k", "catalog_coverage")), "Comparison differs from saved metrics")

    als = ImplicitALS(data, **(als_options or {}))
    started = perf_counter()
    als.fit()
    fit_seconds = round(perf_counter() - started, 3)
    raw = als.recommend(truth, depth, blocked, popularity.ranking.head(0))
    als_row = assess("als", als.recommend(truth, k, blocked, popularity.ranking))
    partial = {name: recover_channel(two, name) for name in ("item_cf", "user_cf")}
    three = reciprocal_rank_pool(partial["item_cf"], partial["user_cf"], constant, als_candidates=raw)
    fused_row = assess(THREE, recommend_fused(three, truth, k, blocked, popularity.ranking, source=THREE))

    keys_two = two.select(*KEYS)
    keys_als = raw.select(*KEYS)
    new = keys_als.join(keys_two, on=KEYS, how="anti")
    require(three.height == two.height + new.height, "Fusion candidate union count mismatch")
    answer_rows = pl.DataFrame([(user, product) for user, answers in truth.items() for product in answers],
                               schema={"user_id": pl.Int64, "product_id": pl.Int64}, orient="row")
    new_answers = new.join(answer_rows, on=KEYS, how="semi")
    by_user = {int(uid): count for uid, count in new_answers.group_by("user_id").len().iter_rows()}
    als_only_top_hits = new_answers.join(predictions[-1].select(*KEYS), on=KEYS, how="semi")
    votes = pl.sum_horizontal(pl.col("item_cf_rank", "user_cf_rank", "als_rank").is_not_null().cast(pl.Int32))
    crowded = three.filter(votes >= 2).group_by("user_id").len().filter(pl.col("len") >= k)
    guaranteed_outranked = (new_answers.join(crowded, on="user_id", how="semi").height
                           if 2 / (constant + depth) > 1 / (constant + 1) else 0)
    base_diagnostic = candidate_diagnostics(two, truth)
    base_recall = base_diagnostic["macro_candidate_recall"]["union"]
    require(abs(base_recall - prior["candidate_pool"]["macro_candidate_recall"]["union"]) < 1e-12,
            "The two-channel candidate recall does not match its manifest")
    als_keys = set(keys_als.iter_rows())
    als_recall = sum(sum((user, item) in als_keys for item in answers) / len(answers) for user, answers in truth.items()) / len(truth)
    increment = sum(by_user.get(user, 0) / len(answers) for user, answers in truth.items()) / len(truth)
    report = {
        "split": "validation_only", "test_evaluated": False, "target_policy": POLICY,
        "history_end_exclusive_utc": prior["history_end_exclusive_utc"], "answer_end_exclusive_utc": prior["answer_end_exclusive_utc"],
        "input_sha256": context["hashes"], "two_channel_artifact_sha256": source_hashes, "two_channel_run": str(fusion_dir.resolve()),
        "k": k, "candidate_depth_per_channel": depth, "rank_constant": constant, "als_parameters": als.options,
        "als_version": implicit.__version__, "als_fit_seconds": fit_seconds,
        "als_definition": "CPU implicit ALS; observed positive confidence 1 + scale*interaction_score; unseen confidence 1; positive dot-product candidates only",
        "selection": selection,
        "candidate_pool": {"two_channel_rows": two.height, "als_rows": raw.height, "three_channel_rows": three.height,
                           "two_channel_macro_recall": base_recall, "als_macro_recall": als_recall,
                           "three_channel_macro_recall": base_recall + increment,
                           "als_extra_eligible_purchase_pairs": new_answers.height,
                           "users_with_als_extra_purchase_pairs": len(by_user),
                           "als_extra_pairs_outranked_by_at_least_k_multi_source_candidates": guaranteed_outranked,
                           "als_extra_eligible_purchases_in_fused_top_k": als_only_top_hits.height},
        "overall_results": {row["model"]: row for row in metrics if row["user_group"] == "overall"},
        "known_user_results": {row["model"]: row for row in metrics if row["user_group"] == "known_users"},
    }
    require(context["hashes"] == {name: digest(model_dir / name) for name in INPUT_NAMES}, "Validation inputs changed")
    require(source_hashes == {name: digest(path) for name, path in sources.items()}, "Two-channel reference files changed")
    output_dir.mkdir(parents=True, exist_ok=False)
    pl.DataFrame(metrics).write_csv(output_dir / "metrics_summary.csv", include_bom=True)
    three.write_parquet(output_dir / "candidate_pool.parquet", compression="zstd")
    pl.concat(predictions, how="diagonal").select(*RECOMMENDATION_SCHEMA, *EVIDENCE, *ALS_EVIDENCE, "model").write_parquet(
        output_dir / "recommendations.parquet", compression="zstd")
    pl.concat(per_user).write_parquet(output_dir / "user_metrics.parquet", compression="zstd")
    (output_dir / "run_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "RESULTS.md").write_text(render(report), encoding="utf-8")
    print(f"ALS Recall@{k}={als_row['recall_at_k']:.4%}; three-channel Recall@{k}={fused_row['recall_at_k']:.4%}", flush=True)
    return report


def render(report):
    s, p = report["selection"], report["candidate_pool"]
    k = report["k"]
    lines = ["# ALS 第三路召回：验证结果", "",
             f"验证期为 2019 年 11 月 11 日至 20 日（UTC），{s['evaluated_users']:,} 位用户、{s['eligible_target_pairs']:,} 组购买答案。"
             "训练期已购买或加购的商品从各路候选和答案中排除；历史上只浏览过的商品仍可推荐。未读取最终测试集。", "",
             f"复用上轮两路候选和榜单，ALS 也为每位用户最多生成 {report['candidate_depth_per_channel']} 个纯个性化候选；"
             f"三路按 `1/({report['rank_constant']}+名次)` 等权投票，同一个商品合并为一条，最后才用购买热门补齐。"
             "ALS 用训练期行为分数构造置信度，潜在向量内积是排序分数。ALS 没有 CF 那样的种子商品或相似用户理由。", "",
             f"| 模型 | Recall@{k} | NDCG@{k} | 有历史用户 Recall@{k} | 商品覆盖率 | 头部推荐位 | 热门补齐位 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in report["overall_results"].items():
        known = report["known_user_results"].get(name)
        known_recall = f"{known['recall_at_k']:.2%}" if known else "—"
        lines.append(f"| {LABELS[name]} | {row['recall_at_k']:.2%} | {row['ndcg_at_k']:.4f} | {known_recall} | "
                     f"{row['catalog_coverage']:.3%} | {row['head_slot_share']:.2%} | {row['fallback_slot_share']:.2%} |")
    previous = report["overall_results"]["rrf_fusion"]["recall_at_k"]
    current = report["overall_results"][THREE]["recall_at_k"]
    lines += ["", "## 候选增量（不含热门补齐）", "",
              f"- 两路并集 Recall：**{p['two_channel_macro_recall']:.2%}**；ALS 单路：**{p['als_macro_recall']:.2%}**；"
              f"三路并集：**{p['three_channel_macro_recall']:.2%}**。这里衡量有没有进候选池，不等于最终名次表现。",
              f"- ALS 补回了 **{p['als_extra_eligible_purchase_pairs']:,} 组**两路都遗漏的购买答案，覆盖 "
              f"{p['users_with_als_extra_purchase_pairs']:,} 位用户；其中 **{p['als_extra_eligible_purchases_in_fused_top_k']:,} 组**进入三路融合前 {k}。",
              f"- 其中 {p['als_extra_pairs_outranked_by_at_least_k_multi_source_candidates']:,} 组由于同一用户有至少 {k} 件多路候选，"
              "且本轮每件多路候选的最低票分也高于单路最高票分，因此在当前 RRF 规则下进不了前十。",
              f"- 三路相对两路的最终 Recall 差值为 **{(current-previous)*100:+.2f} 个百分点**。这是固定参数的验证集对照，不是最终测试结论。",
              "", "## 文件与复现", "",
              "`candidate_pool.parquet` 保存去重候选、三路名次及证据；`recommendations.parquet` 保存各模型最终榜单。"
              "`metrics_summary.csv` 和 `user_metrics.parquet` 保存评估结果；`run_manifest.json` 保存参数和输入哈希。", "",
              "```powershell", r"Set-Location D:\user_behavior",
              r".\.venv\Scripts\python.exe -m pip install -r requirements-als.txt",
              r".\.venv\Scripts\python.exe -B scripts\run_als_validation.py --output-dir reports\als_validation_rerun",
              "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data-dir", type=Path, default=ROOT / "model_data")
    parser.add_argument("--fusion-dir", type=Path, default=ROOT / "reports" / "cf_fusion_validation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "als_validation")
    args = parser.parse_args()
    try:
        run(args.model_data_dir, args.fusion_dir, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved validation-only comparison to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
