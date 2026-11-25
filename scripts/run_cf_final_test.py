"""Evaluate the frozen CF variants once on the held-out November 21-30 split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
import scipy

from scripts.prepare_model_data import DATA_END, DATA_START, TEST_START, TRAIN_END
from scripts.run_cf_fusion_validation import NAMES, POLICY, evaluate_split, write_evaluation
from scripts.run_cf_validation import load_temporal_inputs
from scripts.run_popularity_baseline import digest, require

TEST_FILES = ("test_history_interactions.parquet", "test_targets.parquet", "test_user_segments.parquet")
DEFAULT_MODEL = "user_cf_n100"
FROZEN = {
    "k": 10, "neighbors": 100, "candidate_depth_per_channel": 50, "rank_constant": 60,
    "item_cf": {"min_common_users": 2, "shrinkage": 15., "iuf": True, "popularity_alpha": 0.},
}


def validate_freeze(path):
    settings = json.loads(path.read_text(encoding="utf-8"))
    require(settings["split"] == "validation_only" and settings["test_evaluated"] is False
            and settings["target_policy"] == POLICY and settings["history_end_exclusive_utc"] == str(TRAIN_END)
            and settings["answer_end_exclusive_utc"] == str(TEST_START), "Invalid validation reference")
    require(all(settings[key] == value for key, value in FROZEN.items() if key != "item_cf")
            and settings["model_definition"]["item_cf"] == FROZEN["item_cf"],
            "Validation reference differs from the frozen final-test configuration")


def purchase_outcomes(context):
    history = context["interactions"]
    keys = ["user_id", "product_id"]
    flags = history.select(*keys, (pl.col("purchase_count") > 0).alias("prior_purchase"),
                           (pl.col("cart_count") > 0).alias("prior_cart"))
    catalog = history.select("product_id").unique().with_columns(pl.lit(True).alias("in_train_catalog"))
    return (context["targets"].select(*keys, "first_future_purchase_at", "last_future_purchase_at")
            .join(flags, on=keys, how="left", validate="1:1")
            .join(catalog, on="product_id", how="left", validate="m:1")
            .with_columns(pl.col("prior_purchase", "prior_cart", "in_train_catalog").fill_null(False))
            .with_columns((~(pl.col("prior_purchase") | pl.col("prior_cart"))).alias("eligible_for_evaluation"))
            .sort(keys))


def run(model_dir: Path, reference: Path, output_dir: Path, *, quiet=False):
    require(not output_dir.exists(), "Output directory exists; no test output will be overwritten")
    reference_hash = digest(reference)
    validate_freeze(reference)
    context = load_temporal_inputs(model_dir, TEST_FILES, TEST_START, DATA_END, history_start=DATA_START)
    summary, *tables = evaluate_split(
        context, k=FROZEN["k"], neighbors=FROZEN["neighbors"], candidate_depth=FROZEN["candidate_depth_per_channel"],
        rank_constant=FROZEN["rank_constant"], preset=FROZEN["item_cf"], quiet=quiet,
    )
    answers = purchase_outcomes(context)
    eligible = answers.filter(pl.col("eligible_for_evaluation"))
    removed_purchase = answers.filter(pl.col("prior_purchase")).height
    removed_cart = answers.filter(pl.col("prior_cart") & ~pl.col("prior_purchase")).height
    require(eligible.height == summary["selection"]["eligible_target_pairs"]
            and eligible["user_id"].n_unique() == summary["selection"]["evaluated_users"]
            and removed_purchase + removed_cart == summary["selection"]["removed_target_pairs"], "Inconsistent answer exclusions")
    summary["selection"].update({"removed_prior_purchase_pairs": removed_purchase, "removed_prior_cart_only_pairs": removed_cart})
    require(context["hashes"] == {name: digest(model_dir / name) for name in TEST_FILES}, "Test inputs changed")
    require(reference_hash == digest(reference), "Validation reference changed")
    report = {
        "split": "final_test", "test_evaluated": True, "target_policy": POLICY,
        "history_start_inclusive_utc": str(DATA_START), "history_end_exclusive_utc": str(TEST_START),
        "answer_end_exclusive_utc": str(DATA_END), "frozen_configuration": FROZEN,
        "default_model_chosen_on_validation": DEFAULT_MODEL, "validation_reference": str(reference.resolve()),
        "validation_reference_sha256": reference_hash, "input_sha256": context["hashes"],
        "versions": {"polars": pl.__version__, "numpy": np.__version__, "scipy": scipy.__version__}, **summary,
    }
    write_evaluation(output_dir, *tables)
    answers.write_parquet(output_dir / "purchase_outcomes.parquet", compression="zstd")
    (output_dir / "run_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "RESULTS.md").write_text(render_report(report), encoding="utf-8")
    return report


def render_report(report):
    s, results, known = report["selection"], report["overall_results"], report["known_user_results"]
    popular = max((results["popularity_purchase"], results["popularity_interaction"]), key=lambda row: row["recall_at_k"])
    default, fused = results[DEFAULT_MODEL], results["rrf_fusion"]
    rrf_delta = (fused["recall_at_k"] - default["recall_at_k"]) * 100
    candidate = report["candidate_pool"]["macro_candidate_recall"]
    lines = ["# 协同过滤推荐：最终测试", "",
             "用 2019 年 10 月 1 日至 11 月 20 日（UTC）的行为重新训练，预测 11 月 21 日至 30 日的购买。"
             "测试期购买只作为答案，不参与模型训练、评分特征或参数选择。", "",
             "固定验证阶段的参数：两种 CF 保留 100 个邻居；Item-CF 收缩系数 15、至少 2 位共同用户、开启 IUF、不做热门降权；"
             "两路各取前 50 个候选，用等权 RRF（常数 60）融合。个性化候选不足时以购买热门补齐。未使用 ALS 或 LightGBM。", "",
             f"- 原有 {s['original_purchase_users']:,} 名购买用户、{s['original_target_pairs']:,} 组去重用户－商品答案。"
             f"过滤后评价 {s['evaluated_users']:,} 名用户、{s['eligible_target_pairs']:,} 组答案。",
             f"- 排除历史已购 {s['removed_prior_purchase_pairs']:,} 组、历史仅加购未购买 {s['removed_prior_cart_only_pairs']:,} 组；"
             f"{s['removed_empty_label_users']:,} 名用户无剩余答案。历史只浏览的商品仍可推荐，训练库没有出现的答案仍计入分母。", "",
             "| 模型 | Recall@10 | NDCG@10 | 有历史用户 Recall@10 | 商品覆盖率 | 头部推荐位 | 热门补齐位 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in results.items():
        history_recall = f"{known[name]['recall_at_k']:.2%}" if name in known else "—"
        lines.append(f"| {NAMES[name]} | {row['recall_at_k']:.2%} | {row['ndcg_at_k']:.4f} | {history_recall} | "
                     f"{row['catalog_coverage']:.3%} | {row['head_slot_share']:.2%} | {row['fallback_slot_share']:.2%} |")
    lines += ["", "## 结论", "",
              f"- 验证阶段已经指定 User-CF 为界面默认模型。最终测试中，它的 Recall 相对较强热门基线"
              f"（{NAMES[popular['model']]}）相差 **{(default['recall_at_k'] - popular['recall_at_k']) * 100:+.2f} 个百分点**。",
              f"- 两路融合相对 User-CF 的最终 Recall 差值为 **{rrf_delta:+.2f} 个百分点**。"
              f"不含热门补齐时，候选并集 Recall 为 {candidate['union']:.2%}，单路 User-CF 为 {candidate['user_cf']:.2%}；"
              "候选池命中不等于进入前十。",
              "- Recall 按用户计算后取平均。头部定义是重训商品库中购买热门前 1%。以上是离线结果，不能解释为线上购买率变化，也不用于重新调参。", "",
              "## 产出", "", "`recommendations.parquet` 保存排名和理由；`purchase_outcomes.parquet` 单独保存真实购买与是否计入评估。"
              "`metrics_summary.csv` 包含用户分层，`run_manifest.json` 保存固定参数、时间边界与输入哈希。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data-dir", type=Path, default=ROOT / "model_data")
    parser.add_argument("--validation-reference", type=Path, default=ROOT / "reports" / "cf_fusion_validation" / "run_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "cf_final_test")
    args = parser.parse_args()
    try:
        run(args.model_data_dir, args.validation_reference, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved frozen final-test results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
