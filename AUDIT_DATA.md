# 数据审计

对抽样后的 Parquet 进行第一轮质量检查，不会修改或删除原始数据。

## 运行

```powershell
cd D:\user_behavior

.\.venv\Scripts\python.exe scripts\audit_sample_polars.py `
  "D:\user_behavior\sampled\events_2019_09_10_1pct.parquet" `
  --output-dir "D:\user_behavior\reports\data_audit"
```

## 输出

- `data_quality_summary.json`：总体质量摘要；
- `missing_values.csv`：各字段缺失数量和比例；
- `event_distribution.csv`：浏览、加购、购买的事件、用户、会话和商品数量；
- `source_distribution.csv`：两个月的数据规模和时间范围；
- `price_summary.csv`：价格缺失、异常和分位数；
- `session_id_conflicts.csv`：被多个用户共享的原始会话 ID；
- `category_mapping_conflicts.csv`：一个类别 ID 对应多个类别名称的情况；
- `product_metadata_conflicts.csv`：商品类别或品牌不一致的情况；
- `user_summary.parquet`：用户级汇总；
- `session_summary.parquet`：会话级汇总；
- `product_summary.parquet`：商品级汇总；
- `funnel_summary.csv`：用户、会话和商品三个粒度的漏斗。

先根据审计结果确定清洗规则，再生成 `clean_events.parquet`。不要在审计阶段直接删除低活跃用户、单事件会话或低频商品。
