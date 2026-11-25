"""Run popularity baselines on VALIDATION only; never score the test set here."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl
from models.metrics import aggregate_metrics, evaluate_users, history_exclusions, make_truth, POLICIES
from models.popularity import PopularityRecommender

VALIDATION_START = datetime(2019, 11, 11)
VALIDATION_END = datetime(2019, 11, 21)
INPUT_NAMES = ["train_interactions.parquet", "validation_targets.parquet", "validation_user_segments.parquet"]
OUTPUT_NAMES = ["metrics_summary.csv", "user_metrics.parquet", "recommendations.parquet",
                "run_manifest.json", "RESULTS.md"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run(model_dir: Path, output_dir: Path, k: int = 10,
        target_policy: str = "new_purchase", overwrite: bool = False) -> dict:
    started = time.perf_counter()
    require(k > 0, "k must be positive")
    require(target_policy in POLICIES, "Unknown target policy")
    require(overwrite or not any((output_dir / name).exists() for name in OUTPUT_NAMES),
            "Baseline outputs already exist; choose a new directory or use --overwrite")
    for name in INPUT_NAMES:
        require((model_dir / name).is_file(), f"Missing {name}")
    hashes = {name: digest(model_dir / name) for name in INPUT_NAMES}
    interactions = pl.read_parquet(model_dir / INPUT_NAMES[0])
    targets = pl.read_parquet(model_dir / INPUT_NAMES[1])
    segments = pl.read_parquet(model_dir / INPUT_NAMES[2])
    require(interactions.height > 0 and targets.height > 0, "Empty history or targets")
    require(interactions.select("user_id", "product_id").unique().height == interactions.height,
            "History user-product pairs must be unique")
    require(targets.select("user_id", "product_id").unique().height == targets.height,
            "Target user-product pairs must be unique")
    require(interactions["last_interaction_at"].null_count() == 0
            and interactions["last_interaction_at"].max() < VALIDATION_START,
            "History timestamp is missing or leaks into validation")
    require(targets["first_future_purchase_at"].null_count() == 0
            and targets["first_future_purchase_at"].min() >= VALIDATION_START
            and targets["last_future_purchase_at"].null_count() == 0
            and targets["last_future_purchase_at"].max() < VALIDATION_END,
            "Target timestamps outside validation window")
    require(segments["user_id"].n_unique() == segments.height, "Duplicate user segment rows")
    catalog = set(interactions["product_id"].to_list())
    historical_users = set(interactions["user_id"].to_list())
    user_info = {int(row["user_id"]): row for row in segments.to_dicts()}
    for user in targets["user_id"].unique().to_list():
        require(user in user_info, "Missing evaluation-user segment")
        require(bool(user_info[user]["known_user"]) == (user in historical_users),
                "User-history segment mismatch")
    blocked = history_exclusions(interactions, target_policy)
    truth, selection = make_truth(targets, blocked)
    require(bool(truth), "No eligible validation answers after applying policy")

    # One shared head definition for both models, fitted solely on train history.
    purchase_model = PopularityRecommender("purchase").fit(interactions)
    head = purchase_model.head_products(0.01)
    all_metrics, all_predictions, all_user_metrics = [], [], []
    for mode in ["purchase", "interaction"]:
        name = f"popularity_{mode}"
        model = purchase_model if mode == "purchase" else PopularityRecommender(mode).fit(interactions)
        predictions = model.recommend(truth.keys(), k, blocked)
        for user, product in predictions.select("user_id", "product_id").iter_rows():
            require(product not in blocked.get(user, set()), "Excluded historical item recommended")
        per_user = evaluate_users(predictions, truth, catalog, user_info, k)
        rows = aggregate_metrics(per_user, predictions, catalog, head, k)
        all_metrics.extend([{"model": name, **row} for row in rows])
        all_predictions.append(predictions.with_columns(pl.lit(name).alias("model")))
        all_user_metrics.append(per_user.with_columns(pl.lit(name).alias("model")))

    require(hashes == {name: digest(model_dir / name) for name in INPUT_NAMES},
            "Model input changed during evaluation")
    metrics = pl.DataFrame(all_metrics)
    recommendations = pl.concat(all_predictions)
    per_user = pl.concat(all_user_metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.write_csv(output_dir / "metrics_summary.csv", include_bom=True)
    recommendations.write_parquet(output_dir / "recommendations.parquet", compression="zstd")
    per_user.write_parquet(output_dir / "user_metrics.parquet", compression="zstd")
    for name, expected in [("recommendations.parquet", recommendations.height),
                           ("user_metrics.parquet", per_user.height)]:
        actual = pl.scan_parquet(output_dir / name).select(pl.len()).collect().item()
        require(actual == expected, f"Output row mismatch: {name}")
    report = {
        "evaluation_split": "validation_only", "test_evaluated": False,
        "history_end_exclusive_utc": str(VALIDATION_START),
        "target_end_exclusive_utc": str(VALIDATION_END),
        "input_dir": str(model_dir.resolve()), "input_sha256": hashes,
        "polars_version": pl.__version__, "k": k, "target_policy": target_policy,
        "selection": selection, "history_users": len(historical_users), "catalog_size": len(catalog),
        "head_definition": "Top ceil(1% * historical catalog) by purchase events, tie by users then product_id",
        "head_products": len(head),
        "metrics_definition": {
            "relevance": "binary user-product purchase relevance; repeat future purchases count once",
            "averaging": "macro over eligible buying users; users with no hits remain",
            "precision_denominator": "k, also for short recommendation lists",
            "cold_start": "new users use same popularity list; new items stay in main truth denominator",
            "coverage_denominator": "all products observed in training history",
            "head_slot_share": "offline recommended positions, not observed page impressions",
        },
        "scope": "Conditional on validation buyers; not an online conversion or causal revenue estimate",
        "overall_results": [row for row in all_metrics if row["user_group"] == "overall"],
        "checks": "unique keys, boundaries, exclusions, input hashes and output counts passed",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    write_results(output_dir / "RESULTS.md", report)
    return report


def write_results(path: Path, report: dict) -> None:
    s = report["selection"]
    lines = [
        "# 热门推荐基线：验证集结果", "",
        "本轮不训练Item-CF/User-CF，不评价最终测试集，也不连接SQL Server。", "",
        "## 评估口径", "",
        "- 历史：2019年10月1日至11月10日；答案：11月11日至20日。",
        f"- 推荐长度：Top {report['k']}；过滤策略：`{report['target_policy']}`。",
        "- 默认排除历史已购买商品；购买答案同时排除同一部分。历史浏览/加购但未购买的商品允许推荐。",
        f"- 原始购买用户：{s['original_purchase_users']:,}；最终评价用户：{s['evaluated_users']:,}。",
        f"- 原始答案组合：{s['original_target_pairs']:,}；排除：{s['removed_target_pairs']:,}；保留：{s['eligible_target_pairs']:,}。",
        f"- 因答案全部为历史已购买商品而排除的用户：{s['removed_empty_label_users']:,}。",
        "- 新用户和新商品答案没有从总体评价中删除。新用户使用热门榜；训练期未见商品无法被这个冻结商品库推荐。",
        "- Recall/NDCG/MRR/HitRate是对有有效购买答案用户的宏平均，不是访问购买转化率。", "",
        "## 总体效果", "",
        "| 策略 | Recall@K | NDCG@K | MRR@K | HitRate@K | 商品覆盖率 | 头部推荐位占比 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["overall_results"]:
        label = "购买热门" if row["model"] == "popularity_purchase" else "综合行为热门"
        lines.append(f"| {label} | {row['recall_at_k']:.2%} | {row['ndcg_at_k']:.4f} | "
                     f"{row['mrr_at_k']:.4f} | {row['hit_rate_at_k']:.2%} | "
                     f"{row['catalog_coverage']:.3%} | {row['head_slot_share']:.2%} |")
    lines += [
        "", "## 两种模型", "",
        "- 购买热门：累计历史purchase事件数，按不同交互用户数、商品ID打破同分。",
        "- 综合行为热门：累计历史用户—商品兴趣分数，沿用准备阶段的初始权重，尚未调优。",
        "- 两种方法只有全局热度；除过滤已买商品之外，不根据个人兴趣定制。",
        "- 头部统一定义为历史商品库按购买热门排序的Top 1%；覆盖率分母为全部历史商品，而非只购买过的商品。",
        "- 头部/非头部推荐位占比是离线推荐列表分布，不是真实线上曝光。", "",
        "## 文件怎么用", "",
        "1. 先看本文件；`metrics_summary.csv`包含总体及各历史活跃度层的指标。",
        "2. `recommendations.parquet`每行是一条用户推荐；`user_metrics.parquet`每行是某模型对一个用户的评分。",
        "3. `run_manifest.json`保存输入哈希、口径和校验信息，不是额外训练数据。", "",
        "## 初步观察", "",
        "- 两种热门策略的验证效果接近，目前不能据此宣称存在显著差异。",
        "- 全体用户的推荐仅涉及少量商品，头部推荐位占比为100%，可作为个性化模型的比较起点。",
        "- 热门策略不是根据个人兴趣推荐；Item-CF/User-CF仍需真实验证，不保证一定胜过热门。", "",
        "## 复现", "", "```powershell",
        r"cd D:\user_behavior",
        r".\.venv\Scripts\python.exe scripts\run_popularity_baseline.py --overwrite",
        "```", "",
        "只需查看本目录的结果，不需要把这些文件导入SQL Server，也不需要重新导出业务SQL。", "",
        "下一步将Item-CF、User-CF接入相同的验证集和指标。最终测试集暂不打分，不用它选参数。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data-dir", type=Path, default=ROOT / "model_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "popularity_validation")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--target-policy", choices=sorted(POLICIES), default="new_purchase")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = run(args.model_data_dir, args.output_dir, args.k, args.target_policy, args.overwrite)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Validation baseline complete: {args.output_dir}")
    for row in report["overall_results"]:
        print(f"{row['model']}: Recall@{args.k}={row['recall_at_k']:.4%}; "
              f"NDCG={row['ndcg_at_k']:.6f}; HitRate={row['hit_rate_at_k']:.4%}")
    print("Final test set was NOT evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())