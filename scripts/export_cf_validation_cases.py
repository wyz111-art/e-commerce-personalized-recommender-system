"""Export reproducible Chinese case notes and UI fixtures from the saved validation run."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl

from models.metrics import history_exclusions, make_truth
from scripts.run_cf_validation import load_validation
from scripts.run_popularity_baseline import digest, require

CASE_LABELS = {
    "both": "两种 CF 都命中",
    "item_only": "仅 Item-CF 命中",
    "user_only": "仅 User-CF 命中",
    "known_miss": "都没命中：商品在训练中出现过",
    "cold_miss": "都没命中：商品在训练中从未出现",
}
OUTCOME_LABELS = {**CASE_LABELS, "neither": "两种 CF 都没命中"}
MODEL_LABELS = {"item_cf": "Item-CF", "user_cf": "User-CF",
                "popularity_interaction": "综合行为热门", "popularity_purchase": "购买热门"}
BEHAVIOR = ("view_count", "cart_count", "purchase_count")


def outcome(item_hit, user_hit):
    return ("both" if user_hit else "item_only") if item_hit else ("user_only" if user_hit else "neither")


def observed_behavior(row):
    return "、".join(f"{label} {row[col]} 次" for col, label in zip(BEHAVIOR, ("浏览", "加购", "购买"))
                    if row[col]) or "无记录"


def build_cases(model_dir: Path, run_dir: Path):
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    policy = manifest["target_policy"]
    require(manifest["split"] == "validation_only" and not manifest["test_evaluated"]
            and policy == "exclude_cart_and_purchase", "Expected a discovery-policy validation run")
    context = load_validation(model_dir)
    require(context["hashes"] == manifest["input_sha256"], "Validation inputs do not match the saved run")
    history, target_rows, user_info = context["interactions"], context["targets"], context["user_info"]
    blocked = history_exclusions(history, policy)
    truth, selection = make_truth(target_rows, blocked)
    require(all(manifest["selection"][key] == value for key, value in selection.items()), "Selection mismatch")
    product_support = Counter(history["product_id"].to_list())
    model_map = {**manifest["best_validation_config"], "popularity_interaction": "popularity_interaction",
                 "popularity_purchase": "popularity_purchase"}
    alias = {model: short for short, model in model_map.items()}
    rec_path = run_dir / "recommendations.parquet"
    rec_hash = digest(rec_path)
    recs = defaultdict(lambda: defaultdict(list))
    for row in pl.read_parquet(rec_path).filter(pl.col("model").is_in(list(alias))).sort("rank").iter_rows(named=True):
        who, item = row.pop("user_id"), row["product_id"]
        require(who in truth and item in product_support and item not in blocked.get(who, ()),
                "Recommendation outside the validation population or eligible catalog")
        rows = recs[who][alias[row.pop("model")]]
        require(row["rank"] == len(rows) + 1 <= manifest["k"]
                and all(previous["product_id"] != item for previous in rows), "Invalid recommendation ranks or items")
        rows.append({**row, "hit": item in truth[who]})
    counts, known_counts, misses, selected = Counter(), Counter(), Counter(), {}
    for who, answers in sorted(truth.items()):
        models = recs[who]
        require(set(models) == set(model_map), "Missing model recommendation list")
        hits = {short: [row for row in models[short] if row["hit"]] for short in model_map}
        kind = outcome(hits["item_cf"], hits["user_cf"])
        counts[kind] += 1
        if not user_info[who]["known_user"]:
            continue
        known_counts[kind] += 1
        reachable = sum(item in product_support for item in answers)
        if kind == "neither":
            misses["all_reachable" if reachable == len(answers) else "all_cold" if not reachable else "mixed"] += 1
        # Prefer clear demonstrations: no popular-list hit, true CF evidence, few answers and short histories.
        if hits["popularity_interaction"] or hits["popularity_purchase"]:
            continue
        natural = {
            model: [row for row in hits[model] if row["source"] == model]
            for model in ("item_cf", "user_cf")
        }
        if kind == "neither":
            if not all(any(row["source"] == model for row in models[model]) for model in natural):
                continue
            if reachable == len(answers):
                case = "known_miss"
            elif not reachable:
                case = "cold_miss"
            else:
                continue
        else:
            if any(hits[model] and not natural[model] for model in natural):
                continue
            case = kind
        if case == "both" and not ({r["product_id"] for r in natural["item_cf"]}
                                   & {r["product_id"] for r in natural["user_cf"]}):
            continue
        best_rank = min((r["rank"] for rows in natural.values() for r in rows), default=0)
        support = min(product_support[item] for item in answers) if case == "known_miss" else 0
        key = (len(answers), support, best_rank, user_info[who]["unique_product_count"], who)
        if case not in selected or key < selected[case][0]:
            selected[case] = key, who

    peer_ids = set()
    evidence_items = set()
    for _, who in selected.values():
        for rows in recs[who].values():
            for row in rows:
                if row["source"] == "user_cf":
                    peer_ids.add(row["reason_user_id"])
                elif row["source"] == "item_cf":
                    evidence_items.update((row["product_id"], row["reason_product_id"]))
    users = {who for _, who in selected.values()}
    user_history = defaultdict(dict)
    for row in history.filter(pl.col("user_id").is_in(sorted(users | peer_ids))).iter_rows(named=True):
        user_history[row.pop("user_id")][row["product_id"]] = row
    item_users = defaultdict(set)
    for item, who in history.filter(pl.col("product_id").is_in(sorted(evidence_items))).select("product_id", "user_id").iter_rows():
        item_users[item].add(who)
    targets = defaultdict(list)
    for row in target_rows.filter(pl.col("user_id").is_in(sorted(users))).sort("product_id").iter_rows(named=True):
        who, item = row["user_id"], row["product_id"]
        known = item in product_support
        require(row["known_product"] == known, "Inconsistent cold-start label")
        targets[who].append({"product_id": item, "eligible": item in truth[who], "known_product": known,
                             "training_users": product_support[item], "first_purchase_at": row["first_future_purchase_at"],
                             "history_relation": row["prior_user_product_segment"]})

    cases = []
    for case, label in CASE_LABELS.items():
        if case not in selected:
            continue
        who = selected[case][1]
        own = user_history[who]
        for model in recs[who]:
            for row in recs[who][model]:
                item = row["product_id"]
                if row["source"] == "item_cf":
                    seed = row["reason_product_id"]
                    common = len(item_users[item] & item_users[seed])
                    require(common >= manifest["item_cf_parameters"]["min_common_users"], "Item evidence below fitted threshold")
                    evidence = {"seed_product_id": seed, "seed_history": own[seed], "common_users": common}
                elif row["source"] == "user_cf":
                    peer = row["reason_user_id"]
                    shared = sorted(own.keys() & user_history[peer].keys())
                    require(peer != who and shared, "Invalid peer evidence")
                    evidence = {"peer_user_id": peer, "peer_candidate_history": user_history[peer][item],
                                "shared_product_count": len(shared), "shared_product_ids_sample": shared[:5]}
                else:
                    evidence = None
                row["evidence"] = evidence
        cases.append({"case_id": case, "label_zh": label, "user_id": who, "historical_summary": user_info[who],
                      "history": sorted(own.values(), key=lambda row: (row["last_interaction_at"], -row["product_id"]), reverse=True),
                      "validation_purchases": targets[who], "recommendations": recs[who]})

    for model in ("item_cf", "user_cf"):
        expected = next(row["users_with_hits"] for row in manifest["overall_results"] if row["model"] == model_map[model])
        require(counts["both"] + counts["item_only" if model == "item_cf" else "user_only"] == expected,
                "Recomputed user hit counts differ from the run")
    require(digest(rec_path) == rec_hash, "Recommendations changed during export")
    return {"split": "validation_only", "timezone": "UTC", "policy": policy, "k": manifest["k"],
            "history_end_exclusive_utc": manifest["history_end_exclusive_utc"],
            "answer_end_exclusive_utc": manifest["answer_end_exclusive_utc"],
            "source_run": str(run_dir.resolve()), "input_sha256": context["hashes"],
            "recommendations_sha256": rec_hash, "item_cf_parameters": manifest["item_cf_parameters"],
            "selection": selection, "outcomes_all_users": dict(counts), "outcomes_known_users": dict(known_counts),
            "known_user_misses_by_target_availability": dict(misses), "cases": cases}


def render_report(data):
    lines = ["# 个性化推荐案例分析", "",
             f"按 {data['history_end_exclusive_utc']}（UTC）固定截点生成 Top {data['k']} 推荐，"
             f"之后至 {data['answer_end_exclusive_utc']}（不含）发生的购买只用于核对。"
             "排除历史买过或加购过的商品；只浏览过的可推荐。没有读取最终测试集。", "",
             "## 全量对照", "", "| 至少命中一件 | 全部用户 | 有训练历史的用户 |", "|---|---:|---:|"]
    for key in ("both", "item_only", "user_only", "neither"):
        lines.append(f"| {OUTCOME_LABELS[key]} | {data['outcomes_all_users'].get(key, 0):,} | "
                     f"{data['outcomes_known_users'].get(key, 0):,} |")
    miss = data["known_user_misses_by_target_availability"]
    lines += ["", f"有历史且两种 CF 都没命中的用户中，{miss.get('all_reachable', 0):,} 人的答案全部在训练商品库，"
              f"{miss.get('all_cold', 0):,} 人的答案全部不在训练商品库，{miss.get('mixed', 0):,} 人两种都有。", "",
              "上表按模型最终榜单统计，含热门补齐；以下演示特意挑选两个热门基线都未命中的用户，成功案例的命中来自 CF 本身。"
              "这些例子用来解释模型，不代表总体比例。", ""]
    for case in data["cases"]:
        who, summary = case["user_id"], case["historical_summary"]
        lines += [f"## {case['label_zh']} · 用户 `{who}`", "",
                  f"历史：{summary['session_count']} 个会话、{summary['unique_product_count']} 件商品；{observed_behavior(summary)}。", ""]
        history = "；".join(f"`{row['product_id']}`（{observed_behavior(row)}）" for row in case["history"][:3])
        lines.append(f"最近交互的商品（最多 3 件）：{history}。")
        for row in case["validation_purchases"]:
            prior = {"prior_view_only": "此前只浏览过", "prior_cart_only": "此前加购过",
                     "prior_purchase": "此前买过", "no_prior_user_product_interaction": "用户此前未交互"}[row["history_relation"]]
            status = f"计入评估，训练期有 {row['training_users']} 位用户交互过" if row["eligible"] else "按规则不计入评估"
            if row["eligible"] and not row["known_product"]:
                status = "计入评估，但训练期从未出现，当前模型无法推荐"
            lines.append(f"- `{row['first_purchase_at']:%Y-%m-%d %H:%M:%S}` UTC 购买 `{row['product_id']}`：{prior}；{status}。")
        lines += ["", "| 路径 | 排名 | 推荐商品 | 命中 | 推荐依据 |", "|---|---:|---|---|---|"]
        for model, label in MODEL_LABELS.items():
            rows = case["recommendations"][model]
            for row in (row for row in rows if row["rank"] <= 3 or row["hit"]):
                source, ev = row["source"], row["evidence"]
                if source == "item_cf":
                    reason = (f"历史商品 `{ev['seed_product_id']}`（{observed_behavior(ev['seed_history'])}）；"
                              f"{ev['common_users']} 位训练用户两件都交互过")
                elif source == "user_cf":
                    reason = (f"相似访客 `{ev['peer_user_id']}` 与该用户共同交互 {ev['shared_product_count']} 件商品；"
                              f"该访客对候选商品：{observed_behavior(ev['peer_candidate_history'])}")
                elif source == "popularity_fallback":
                    reason = "购买热门榜补齐"
                else:
                    reason = "按训练期全局热门排序"
                lines.append(f"| {label} | {row['rank']} | `{row['product_id']}` | {'是' if row['hit'] else '否'} | {reason} |")
        lines += ["", "表中展示前 3 位和其余命中项；JSON 保留完整榜单。"]
        if case["case_id"] == "known_miss":
            lines.append("目标商品在商品库里，但没进入任何一张 Top 10。只凭榜单不能判定是没有召回还是排名不够靠前。")
        lines.append("")
    lines += ["## 用于界面", "", "- `validation_cases.json` 提供案例用户、截点前按商品聚合的行为、"
              "完整推荐榜单和验证购买；后续购买放在单独的对照区。",
              "- 推荐原因来自贡献最大的种子商品或相似用户，不表示相同品类，也不保证只有这一条依据。"
              "分数是排序分数，不是购买概率。", "", "聚合历史不是逐事件会话时间线；不能把一行当成一次点击。", "",
              "## 复现", "", "```powershell", r"Set-Location D:\user_behavior",
              r".\.venv\Scripts\python.exe -B scripts\export_cf_validation_cases.py --output-dir reports\cf_validation_cases_rerun",
              "```", ""]
    return "\n".join(lines)


def iso_datetime(value):
    if not isinstance(value, datetime):
        raise TypeError(f"Unexpected JSON value: {type(value).__name__}")
    return value.isoformat() + "Z"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data-dir", type=Path, default=ROOT / "model_data")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "reports" / "cf_discovery_validation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "cf_validation_cases")
    args = parser.parse_args()
    require(not args.output_dir.exists(), "Output directory exists; provide a new --output-dir")
    data = build_cases(args.model_data_dir, args.run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "validation_cases.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=iso_datetime, allow_nan=False) + "\n",
        encoding="utf-8")
    (args.output_dir / "CASES.md").write_text(render_report(data), encoding="utf-8")
    print(f"Exported {len(data['cases'])} cases to {args.output_dir}")


if __name__ == "__main__":
    main()
