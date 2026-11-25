"""Validate Item-CF/User-CF against popularity using identical frozen history and labels.

This command opens only the three validation input files. Final test files are
never loaded. Exact cosine is computed in sparse blocks, not dense full grids.
"""
from __future__ import annotations
import argparse
from datetime import datetime
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
from models.cf_common import SparseInteractions
from models.item_cf import ItemCF
from models.user_cf import UserCF
from models.popularity import PopularityRecommender
from models.metrics import aggregate_metrics, evaluate_users, history_exclusions, make_truth, POLICIES
from scripts.run_popularity_baseline import digest, require, INPUT_NAMES, VALIDATION_START, VALIDATION_END

OUTPUT_NAMES = ["metrics_summary.csv", "recommendations.parquet", "user_metrics.parquet",
                "run_manifest.json", "RESULTS.md"]
ITEM_CF_PRESETS = {
    "baseline": {"min_common_users": 1, "shrinkage": 0., "iuf": False, "popularity_alpha": 0.},
    "shrinkage_iuf": {"min_common_users": 2, "shrinkage": 15., "iuf": True, "popularity_alpha": 0.},
}
POLICY_LABELS = {
    "new_purchase": "排除历史已购买商品",
    "exclude_cart_and_purchase": "排除历史买过或加购过的商品，保留只浏览过的商品",
    "unseen": "排除历史有任何交互的商品",
    "all": "不按历史交互排除商品",
}


def load_validation(model_dir: Path) -> dict:
    return load_temporal_inputs(model_dir, INPUT_NAMES, VALIDATION_START, VALIDATION_END)


def load_temporal_inputs(model_dir, names, cutoff, answer_end, *, history_start=None):
    for name in names:
        require((model_dir / name).is_file(), f"Missing {name}")
    hashes = {name: digest(model_dir / name) for name in names}
    interactions, targets, segments = (pl.read_parquet(model_dir / name) for name in names)
    require(interactions.height > 0 and targets.height > 0, "History/targets must be nonempty")
    require(targets.select("user_id", "product_id").unique().height == targets.height,
            "Duplicate purchase target pairs")
    require(interactions["last_interaction_at"].null_count() == 0
            and interactions["last_interaction_at"].max() < cutoff,
            "Training history must precede the recommendation cutoff")
    if history_start is not None:
        require(interactions["first_interaction_at"].null_count() == 0
                and interactions["first_interaction_at"].min() >= history_start
                and not interactions.select((pl.col("first_interaction_at") > pl.col("last_interaction_at")).any()).item(),
                "Training history is outside its window or has inconsistent timestamps")
    require(targets["first_future_purchase_at"].null_count() == 0
            and targets["first_future_purchase_at"].min() >= cutoff
            and targets["last_future_purchase_at"].null_count() == 0
            and targets["last_future_purchase_at"].max() < answer_end
            and not targets.select((pl.col("first_future_purchase_at") > pl.col("last_future_purchase_at")).any()).item(),
            "Purchase answers must lie within their evaluation window")
    require(segments["user_id"].n_unique() == segments.height, "Duplicate user segments")
    known_users = set(interactions["user_id"].to_list())
    info = {int(row["user_id"]): row for row in segments.to_dicts()}
    for user in targets["user_id"].unique().to_list():
        require(user in info and bool(info[user]["known_user"]) == (user in known_users),
                "Incorrect/missing historical user segment")
    if "last_event_at" in segments.columns:
        require(not segments.filter(pl.col("last_event_at") >= cutoff).height,
                "User segments must precede the recommendation cutoff")
    return {"interactions": interactions, "targets": targets, "user_info": info, "hashes": hashes}


def group_metrics(per_user, predictions, catalog, head, k):
    rows = aggregate_metrics(per_user, predictions, catalog, head, k)
    known = per_user.filter(pl.col("known_user"))
    if known.height:
        row = aggregate_metrics(known, predictions, catalog, head, k)[0]
        rows.append({**row, "user_group": "known_users"})
    for row in rows:
        group = row["user_group"]
        users = per_user if group == "overall" else (
            known if group == "known_users" else per_user.filter(pl.col("activity_segment") == group)
        )
        recs = predictions.join(users.select("user_id"), on="user_id", how="semi")
        fallback = recs.filter(pl.col("source") == "popularity_fallback").height
        fully_fallback = recs.group_by("user_id").agg(
            (pl.col("source") == "popularity_fallback").all().alias("all_fallback")
        )
        row.update({
            "fallback_slot_share": fallback / recs.height if recs.height else 0.,
            "fully_fallback_users": int(fully_fallback["all_fallback"].sum() or 0),
        })
    return rows


def run(model_dir: Path, output_dir: Path, neighbors=(20, 50, 100), k=10,
        target_policy="new_purchase", block_size=64, overwrite=False,
        quiet=False, *, item_preset="baseline") -> dict:
    started = time.perf_counter()
    neighbor_values = sorted(set(neighbors))
    require(bool(neighbor_values) and min(neighbor_values) > 0 and k > 0 and block_size > 0,
            "Neighbors, k and block size must be positive")
    require(target_policy in POLICIES, "Unknown target policy")
    require(item_preset in ITEM_CF_PRESETS, "Unknown Item-CF preset")
    item_options = ITEM_CF_PRESETS[item_preset]
    require(overwrite or not any((output_dir / name).exists() for name in OUTPUT_NAMES),
            "Outputs exist; use a new directory or --overwrite")
    context = load_validation(model_dir)
    interactions, targets, user_info = context["interactions"], context["targets"], context["user_info"]
    data = SparseInteractions(interactions)
    blocked = history_exclusions(interactions, target_policy)
    truth, selection = make_truth(targets, blocked)
    require(bool(truth), "No remaining relevance labels")
    if target_policy == "exclude_cart_and_purchase":
        _, purchase_only = make_truth(targets, history_exclusions(interactions, "new_purchase"))
        selection.update({
            "removed_prior_purchase_pairs": purchase_only["removed_target_pairs"],
            "removed_additional_prior_cart_pairs": selection["removed_target_pairs"] - purchase_only["removed_target_pairs"],
            "removed_additional_prior_cart_users": purchase_only["evaluated_users"] - selection["evaluated_users"],
        })
    if not quiet:
        print(f"{target_policy}: {selection['evaluated_users']:,}/{selection['original_purchase_users']:,} users, "
              f"{selection['eligible_target_pairs']:,}/{selection['original_target_pairs']:,} targets retained", flush=True)
    catalog = set(data.product_ids.tolist())
    popularity = PopularityRecommender("purchase").fit(interactions)
    head = popularity.head_products(0.01)
    all_metrics, all_predictions, all_user_metrics, timings = [], [], [], {}

    def report_progress(label):
        last = [-1]
        def progress(done, total):
            step = (done * 5 // max(total, 1))
            if not quiet and step != last[0]:
                print(f"{label}: {done:,}/{total:,}", flush=True)
                last[0] = step
        return progress

    def assess(name, predictions, elapsed):
        for user, product in predictions.select("user_id", "product_id").iter_rows():
            require(product not in blocked.get(user, set()), "Blocked product returned")
        per_user = evaluate_users(predictions, truth, catalog, user_info, k)
        metrics = group_metrics(per_user, predictions, catalog, head, k)
        all_metrics.extend([{"model": name, **row} for row in metrics])
        all_predictions.append(predictions.with_columns(pl.lit(name).alias("model")))
        all_user_metrics.append(per_user.with_columns(pl.lit(name).alias("model")))
        timings[name] = round(elapsed, 3)
        if not quiet:
            overall = next(row for row in metrics if row["user_group"] == "overall")
            print(f"{name}: Recall@{k}={overall['recall_at_k']:.4%}, "
                  f"NDCG={overall['ndcg_at_k']:.6f}, coverage={overall['catalog_coverage']:.3%}", flush=True)

    for mode in ["purchase", "interaction"]:
        start = time.perf_counter()
        model = popularity if mode == "purchase" else PopularityRecommender(mode).fit(interactions)
        predictions = model.recommend(truth.keys(), k, blocked).with_columns(
            pl.lit("global_popularity").alias("source"),
            pl.lit(None, dtype=pl.Int64).alias("reason_product_id"),
            pl.lit(None, dtype=pl.Int64).alias("reason_user_id"),
            pl.lit(None, dtype=pl.Float64).alias("reason_similarity"),
        )
        assess(f"popularity_{mode}", predictions, time.perf_counter() - start)

    start = time.perf_counter()
    item = ItemCF(data, max(neighbor_values), block_size, **item_options).fit(report_progress("Item cosine"))
    item_fit_seconds = time.perf_counter() - start
    item_edges = item.graph.nnz
    for n in neighbor_values:
        start = time.perf_counter()
        predictions = item.recommend(truth.keys(), n, k, blocked, popularity.ranking)
        assess(f"item_cf_n{n}", predictions, time.perf_counter() - start)
    del item

    start = time.perf_counter()
    user = UserCF(data, max(neighbor_values), block_size).fit()
    user.prepare_neighbors(truth.keys(), report_progress("User cosine"))
    user_fit_seconds = time.perf_counter() - start
    user_edges = user.graph.nnz
    for n in neighbor_values:
        start = time.perf_counter()
        predictions = user.recommend(truth.keys(), n, k, blocked, popularity.ranking)
        assess(f"user_cf_n{n}", predictions, time.perf_counter() - start)
    del user

    require(context["hashes"] == {name: digest(model_dir / name) for name in INPUT_NAMES},
            "Input files changed during experiment")
    metrics = pl.DataFrame(all_metrics)
    predictions = pl.concat(all_predictions)
    per_user = pl.concat(all_user_metrics)
    overall = [row for row in all_metrics if row["user_group"] == "overall"]
    best = {}
    for family in ["item_cf", "user_cf"]:
        choices = [row for row in overall if row["model"].startswith(family)]
        # Primary: macro Recall. Tie: NDCG. Further tie: fewer neighbors.
        winner = max(choices, key=lambda row: (row["recall_at_k"], row["ndcg_at_k"],
                                               -int(row["model"].split("_n")[-1])))
        best[family] = winner["model"]

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.write_csv(output_dir / "metrics_summary.csv", include_bom=True)
    predictions.write_parquet(output_dir / "recommendations.parquet", compression="zstd")
    per_user.write_parquet(output_dir / "user_metrics.parquet", compression="zstd")
    for name, expected in [("recommendations.parquet", predictions.height),
                           ("user_metrics.parquet", per_user.height)]:
        stored = pl.scan_parquet(output_dir / name).select(pl.len()).collect().item()
        require(stored == expected, f"Output row mismatch: {name}")
    report = {
        "split": "validation_only", "test_evaluated": False,
        "input_dir": str(model_dir.resolve()), "input_sha256": context["hashes"],
        "versions": {"polars": pl.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "history_end_exclusive_utc": str(VALIDATION_START),
        "answer_end_exclusive_utc": str(VALIDATION_END),
        "target_policy": target_policy, "k": k, "neighbor_grid": neighbor_values,
        "item_cf_preset": item_preset, "item_cf_parameters": item_options,
        "selection": selection,
        "matrix": {"users": data.matrix.shape[0], "products": data.matrix.shape[1],
                   "nonzeros": data.matrix.nnz, "dtype": str(data.matrix.dtype),
                   "block_size": block_size, "item_neighbor_edges": item_edges,
                   "queried_user_neighbor_edges": user_edges},
        "model_definition": {
            "similarity": "positive cosine on weighted historical interaction vectors; self excluded",
            "item_cf": "sum of historical seed score * adjusted seed-to-candidate neighbor similarity",
            "user_cf": "sum of peer similarity * peer item score, divided by selected similarities sum",
            "similarity_ties": "ascending numeric entity ID", "score_ties": "ascending product ID",
            "user_cf_minimum_shared_interactions": 1,
            "cold_start": "unknown users, zero-overlap users, or short lists use purchase-popularity fallback",
            "fusion": "CF ranks before fallback; scores from different sources are not mixed or calibrated",
        },
        "evaluation": {
            "metrics": "same macro binary-relevance Top-K evaluator as popularity baselines",
            "head": "top ceil(1% of historical catalog) by purchase-popularity ranking",
            "cold_start_targets": "kept in overall denominator, not dropped",
            "activity_labels": "history only; known_users aggregate also included",
            "best_selection": "validation Recall, then NDCG, then fewer neighbors",
        },
        "fit_seconds": {"item_cf": round(item_fit_seconds, 3), "user_cf": round(user_fit_seconds, 3)},
        "recommend_and_evaluate_seconds": timings,
        "best_validation_config": best, "overall_results": overall,
        "selected_segment_results": [row for row in all_metrics if row["model"] in
            {"popularity_purchase", "popularity_interaction", *best.values()}],
        "checks": "boundaries, unique keys, blocked items, rank validity, input hashes, output counts passed",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_results(output_dir / "RESULTS.md", report)
    return report


def write_results(path, report):
    by_name = {row["model"]: row for row in report["overall_results"]}
    names = ["popularity_purchase", "popularity_interaction",
             report["best_validation_config"]["item_cf"], report["best_validation_config"]["user_cf"]]
    labels = {"popularity_purchase": "购买热门", "popularity_interaction": "综合行为热门"}
    s = report["selection"]
    policy = report["target_policy"]
    preset = report["item_cf_preset"]
    options = report["item_cf_parameters"]
    neighbors = " ".join(map(str, report["neighbor_grid"]))
    neighbor_caption = "两种 CF 使用固定邻居数" if len(report["neighbor_grid"]) == 1 else "各显示验证集内最优邻居数"
    selection_detail = []
    if policy == "exclude_cart_and_purchase":
        selection_detail = [
            f"- 其中历史买过 {s['removed_prior_purchase_pairs']:,} 组，历史加购但未购买 {s['removed_additional_prior_cart_pairs']:,} 组；"
            f"相比只排除已购，有 {s['removed_additional_prior_cart_users']:,} 名用户因此没有可评估答案。",
            "- 加购和购买仍可作为历史兴趣信号，只过滤输出候选及验证答案。按训练截点的历史加购事件过滤，"
            "不是还原实时购物车状态，也不使用验证期的新加购事件。",
            "- 旧口径与本轮的用户和答案不同，不应把两轮 Recall 的差值当作模型提升或退化。",
        ]
    lines = [
        "# Item-CF与User-CF：验证集对比", "",
        "本轮只评价验证集；没有连接SQL Server，没有修改原始数据，没有评价最终测试集。", "",
        "## 统一口径", "",
        "- 历史截至2019年11月11日之前，购买答案为11月11日至20日。",
        f"- Top {report['k']}；`{policy}`：{POLICY_LABELS[policy]}。全部模型使用相同过滤规则。",
        f"- 原有 {s['original_purchase_users']:,} 名购买用户、{s['original_target_pairs']:,} 组答案；"
        f"保留 {s['evaluated_users']:,} 名用户、{s['eligible_target_pairs']:,} 组答案。",
        f"- 排除 {s['removed_target_pairs']:,} 组答案，{s['removed_empty_label_users']:,} 名用户过滤后没有可评估答案。",
        *selection_detail,
        "- 新用户、新商品答案仍在总体指标中；另有历史已知用户组`known_users`。",
        "- Recall、NDCG、MRR、HitRate是用户级宏平均，不是访问购买转化率。", "",
        f"## 总体结果（{neighbor_caption}）", "",
        "| 模型 | Recall@K | NDCG@K | HitRate@K | 商品覆盖率 | 头部推荐位占比 | 热门补齐位占比 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        r = by_name[name]
        label = labels.get(name, name)
        lines.append(f"| {label} | {r['recall_at_k']:.2%} | {r['ndcg_at_k']:.4f} | "
                     f"{r['hit_rate_at_k']:.2%} | {r['catalog_coverage']:.3%} | "
                     f"{r['head_slot_share']:.2%} | {r['fallback_slot_share']:.2%} |")
    lines += ["", "以上CF参数是在本验证集选择，不是最终测试结论，也不是显著性证明。", "",
              "## 全部邻居参数实验", "",
              "| 配置 | Recall@K | NDCG@K | 商品覆盖率 |", "|---|---:|---:|---:|"]
    for name, r in by_name.items():
        if name.startswith(("item_cf", "user_cf")):
            lines.append(f"| {name} | {r['recall_at_k']:.2%} | {r['ndcg_at_k']:.4f} | {r['catalog_coverage']:.3%} |")
    lines += ["", "## 如何理解结果", ""]
    popularity_best = max((by_name[n] for n in ["popularity_purchase", "popularity_interaction"]),
                          key=lambda row: row["recall_at_k"])
    user_best = by_name[report["best_validation_config"]["user_cf"]]
    item_best = by_name[report["best_validation_config"]["item_cf"]]
    user_delta = (user_best["recall_at_k"] - popularity_best["recall_at_k"]) * 100
    item_delta = (item_best["recall_at_k"] - popularity_best["recall_at_k"]) * 100
    lines += [
        f"- 相对较强热门基线，User-CF 的验证 Recall 相差 {user_delta:+.2f} 个百分点，"
        f"Item-CF 相差 {item_delta:+.2f} 个百分点。这不是线上转化提升或最终测试结论。",
        f"- Item-CF 预设 `{preset}`：共同用户门槛 {options['min_common_users']}、"
        f"收缩 {options['shrinkage']:g}、IUF {'开启' if options['iuf'] else '关闭'}、热门降权 {options['popularity_alpha']:g}。",
        f"- 邻居数：{neighbors}。" + ("本轮固定此配置，不重新调参。" if len(report["neighbor_grid"]) == 1 else "仅为本轮候选，不能证明全局最优。"), "",
        "## 有历史用户的对比", "",
        "| 模型 | 有历史用户数 | Recall@K | NDCG@K | 热门补齐位占比 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report.get("selected_segment_results", []):
        if row["user_group"] == "known_users":
            label = labels.get(row["model"], row["model"])
            lines.append(f"| {label} | {row['users']:,} | {row['recall_at_k']:.2%} | "
                         f"{row['ndcg_at_k']:.4f} | {row['fallback_slot_share']:.2%} |")
    lines += [
        "", "所有新用户的CF推荐都来自购买热门回退，所以这一组的命中不是CF学到的个性化能力。",
        "有历史用户组仍可能有少量无共同商品或候选不足用户，必须结合热门补齐占比理解。",
        "", "## 模型与推荐理由", "",
        "- Item-CF：对历史商品的用户向量计算余弦相似度，由用户历史商品投票。",
        "- User-CF：对用户的商品向量计算余弦相似度，由最相似的其他用户投票。",
        "- 推荐包含`source`；`popularity_fallback`表示无历史、无相似邻居或候选不足而补齐。",
        "- Item-CF的`reason_product_id`是贡献最大的历史种子商品；User-CF的`reason_user_id`是贡献最大的相似用户。",
        "- 协同行为相似不保证商品语义/品类相同；不要将行为相似解释成相同型号或品牌。",
        "- CF分数是排序分数，不是购买概率；补齐商品的分数是热门分数，不跨来源直接比大小。", "",
        "## 分布与计算边界", "",
        "- 头部统一是训练商品库购买热门Top 1%，非头部推荐位占比为其补集，不是真实线上曝光。",
        "- 相似度使用稀疏矩阵分块计算，只保留Top邻居，不生成完整用户/商品两两稠密矩阵。",
        "- Item-CF 按上面记录的预设调整相似度；User-CF 保持原实现。", "",
        "## 查看文件", "",
        "- 本文件：直接阅读结果。",
        "- `metrics_summary.csv`：所有参数和用户分层指标，`known_users`可查看有历史用户表现。",
        "- `recommendations.parquet`：每个配置的用户Top-N列表与历史依据；不同model是不同配置，不要混成一张榜。",
        "- `user_metrics.parquet`：各配置的逐用户指标；`run_manifest.json`保存参数、哈希、版本和耗时。", "",
        "## 复现", "", "```powershell", r"cd D:\user_behavior",
        r".\.venv\Scripts\python.exe -B scripts\run_cf_validation.py"
        f" --neighbors {neighbors} --k {report['k']} --block-size {report['matrix']['block_size']}"
        f" --target-policy {policy} --item-preset {preset}"
        + r" --output-dir reports\cf_validation_rerun",
        "```", "", "下一步根据验证结果选择或调整方案。参数与评估口径确定后，再用测试前历史重建并只评价一次测试集。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data-dir", type=Path, default=ROOT / "model_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "cf_validation")
    parser.add_argument("--neighbors", nargs="+", type=int, default=[20, 50, 100])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--target-policy", choices=sorted(POLICIES), default="new_purchase")
    parser.add_argument("--item-preset", choices=sorted(ITEM_CF_PRESETS), default="baseline")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = run(args.model_data_dir, args.output_dir, args.neighbors, args.k,
                     args.target_policy, args.block_size, args.overwrite, item_preset=args.item_preset)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Validation comparison saved: {args.output_dir}")
    print(f"Best configs on validation: {report['best_validation_config']}")
    print(f"Elapsed {report['elapsed_seconds']:.2f}s. Final test set was NOT evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
