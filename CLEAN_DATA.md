# 样本数据清洗

清洗脚本不会覆盖抽样 Parquet 原件。默认只删除完全重复行，保留单会话用户、单事件会话和低频商品。

## 运行

```powershell
cd D:\user_behavior

.\.venv\Scripts\python.exe scripts\clean_sample_polars.py `
  "D:\user_behavior\sampled\events_2019_09_10_1pct.parquet" `
  --output-dir "D:\user_behavior\processed"
```

## 清洗规则

- 仅删除所有源字段完全相同的重复记录；
- `event_time` 转换为 UTC 时间字段 `event_at`，并生成日期、月份和小时；
- 保留原始 `category_id`，`category_code` 缺失继续保持为空；
- 品牌缺失时，仅在同一商品存在唯一稳定品牌的情况下补齐，否则填为 `unknown_brand`；
- 零价格保留并标记 `price_zero_flag`，负价格保留并标记 `price_negative_flag`；
- 创建 `session_key = user_id + '_' + user_session`；
- 商品类别或品牌元数据冲突用 `metadata_conflict_flag` 标记，不删除商品；
- 不删除低活跃用户、单事件会话或低频商品。

## 输出

- `clean_events_2019_10_11_1pct.parquet`：清洗后的事件明细；
- `user_summary_clean.parquet`：用户级汇总；
- `session_summary_clean.parquet`：会话级汇总；
- `product_summary_clean.parquet`：商品级汇总；
- `cleaning_summary.json`：清洗前后行数及标记统计。
