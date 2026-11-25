# 两路 CF 融合：验证结果

验证期为 2019 年 11 月 11 日至 20 日（UTC），共 2,274 位用户、3,287 组购买答案。历史已购和已加购商品同时从推荐与评估答案中排除。未读取最终测试集。

Item-CF 和 User-CF 邻居数固定为 100，各取最多 50 个纯个性化候选。合并去重后，对每个来源加 `1 / (60 + 名次)`，取分数最高的 10 件；候选不足再以购买热门补齐。两个模型的原始分数不参与融合。

## Top 10 对比

| 模型 | Recall@10 | NDCG@10 | 有历史用户 Recall@10 | 商品覆盖率 | 头部推荐位 | 热门补齐位 |
|---|---:|---:|---:|---:|---:|---:|
| 购买热门 | 11.97% | 0.0685 | 9.80% | 0.031% | 100.00% | 0.00% |
| 综合行为热门 | 12.14% | 0.0688 | 9.91% | 0.032% | 100.00% | 0.00% |
| 改进版 Item-CF | 12.92% | 0.0756 | 11.20% | 6.953% | 67.42% | 34.01% |
| User-CF | 15.37% | 0.0973 | 14.83% | 5.088% | 77.49% | 36.18% |
| 两路融合（RRF） | 14.49% | 0.0865 | 13.53% | 5.676% | 73.05% | 32.94% |

融合相对 User-CF 的 Recall@10 差值：**-0.88 个百分点**。36 位用户仅融合命中，62 位用户仅 User-CF 命中。融合没有超过 User-CF 的验证 Recall；不宜仅凭合并候选更多就替换现有默认模型。

## 候选池诊断（不含热门补齐）

Item-CF / User-CF / 两路并集的候选 Recall 分别为 **16.63%** / **15.49%** / **20.53%**。候选 Recall 衡量购买商品有没有进入候选池，不是最终 Top 10 Recall。

按去重的购买用户－商品对计数：两路都找到 368，仅 Item-CF 找到 172，仅 User-CF 找到 128，两路都没找到 2,619。

## 文件与复现

- `candidate_pool.parquet` 保留去重候选、两路名次、两路推荐理由和 RRF 分数，可用于后续排序实验。
- `recommendations.parquet` 保存各模型完整 Top 10；融合分数和热门补齐分数不是购买概率。
- `metrics_summary.csv` 记录总体和用户分层指标；`user_metrics.parquet` 记录逐用户指标；`run_manifest.json` 保留配置与输入哈希。

```powershell
Set-Location D:\user_behavior
.\.venv\Scripts\python.exe -B scripts\run_cf_fusion_validation.py --candidate-depth 50 --rank-constant 60 --output-dir reports\cf_fusion_validation_rerun
```
