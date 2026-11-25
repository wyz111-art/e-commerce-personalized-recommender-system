# SQL Server 业务分析

这些脚本用于分析 Polars 清洗后的事件数据。

## 最简导入流程

### 1. 使用 Polars 导出 CSV

```powershell
cd D:\user_behavior

.\.venv\Scripts\python.exe scripts\export_clean_csv.py `
  "D:\user_behavior\processed\clean_events_2019_10_11_1pct.parquet" `
  --output "D:\user_behavior\processed\clean_events_2019_10_11_1pct.csv"
```

脚本会将布尔值转换成 `0/1`，并验证 CSV 与 Parquet 的行数一致。

### 2. 在 DBeaver 执行建表脚本

执行：

```text
00_create_clean_events_table.sql
```

### 3. 使用 DBeaver 导入 CSV

右键表 `dbo.clean_events_2019_10_11_1pct`，选择 `Import Data`，然后：

- 选择 CSV 文件；
- 编码选择 UTF-8；
- 第一行作为列名；
- 空字符串作为 NULL；
- 检查每一列是否映射到同名字段；
- 执行导入。

导入完成后验证：

```sql
SELECT COUNT_BIG(*) AS row_count
FROM dbo.clean_events_2019_10_11_1pct;
```

正确行数应为：

```text
1093742
```

### 4. 导入后创建索引

执行：

```text
00_create_indexes_after_import.sql
```

### 5. 运行分析脚本

脚本默认使用的字段包括：

```text
event_at
event_type
product_id
category_id
category_code
brand_clean
price
user_id
user_session
session_key
event_month
```

如果你的 SQL Server 表名不同，只需要在各个脚本中替换：

```sql
dbo.clean_events_2019_10_11_1pct
```

## 完整执行顺序

1. `00_create_clean_events_table.sql`：创建目标表；
2. 使用 DBeaver 导入 CSV；
3. `00_create_indexes_after_import.sql`：创建分析索引；
4. `01_conversion_funnel.sql`：整体会话转化漏斗，不按月份拆分；
5. `02_user_segmentation.sql`：输出每个用户的行为指标和用户分层；
6. `03_user_segment_summary.sql`：汇总各用户层的规模和转化表现。
7. `04_session_path_analysis.sql`：将每个会话划分为互斥行为路径。
8. `05_product_long_tail_analysis.sql`：分析商品购买覆盖、头部集中度和跨月商品结构。
9. `06_time_split_analysis.sql`：验证训练、验证、测试时间边界及冷启动样本规模。

统计范围为数据中实际存在的全部时间，即 2019 年 10 月至 11 月。漏斗脚本会检查事件先后顺序，不会把同一会话中时间顺序错误的行为直接当成转化路径。

用户分层阈值是第一版分析规则：

- 1 个会话：单会话用户；
- 2—3 个会话：低活跃用户；
- 4—7 个会话：中活跃用户；
- 8 个及以上会话：高活跃用户。

跑完用户分层后，再根据实际分布决定是否调整阈值。不要用这一步的分层结果删除用户。
