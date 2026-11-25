# 大文件用户抽样

该脚本对所有输入 CSV 使用同一套 `user_id` 固定抽样规则，保留抽中用户在 9 月和 10 月的全部行为，并输出一个压缩 Parquet 文件。

## 安装

建议把虚拟环境放在 D 盘项目目录，避免大型依赖占用系统盘：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 运行

把下面两个输入路径替换成你的真实文件路径：

```powershell
.\.venv\Scripts\python.exe scripts/sample_users_polars.py `
  "D:\data\2019-Sep.csv" `
  "D:\data\2019-Oct.csv" `
  --sample-percent 1 `
  --output "D:\data\sampled\events_2019_09_10_1pct.parquet"
```

脚本会生成：

- `events_2019_09_10_1pct.parquet`：抽中用户的两个月完整行为；
- `events_2019_09_10_1pct.audit.json`：样本行数、用户数、商品数、会话数、事件分布、购买占比、购买用户数和购买会话数。

如果 1% 样本中的购买用户或购买会话不足，可将参数改成：

```text
--sample-percent 2
```

不要分别对两个文件使用随机抽样。该脚本通过 `user_id` 固定分桶，确保同一个用户在两个月中要么全部保留，要么全部不保留。
