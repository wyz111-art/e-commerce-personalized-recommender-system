# 推荐模型数据准备

本阶段将已经完成的时间切分落实为模型输入。使用 Polars 读取清洗后的 Parquet，不连接 SQL Server、不使用 DuckDB、不训练模型，也不修改源文件。

第一版模型范围：热门推荐基线、Item-CF、User-CF。ALS、LightGBM 和相似会话不是本阶段任务。

## 1. 推荐任务与时间边界

采用“固定时点，用历史预测未来十天购买”的用户级任务，而不是读取购买前全部购物车来预测同次购买。

全部时间按原始 UTC 解释，不转换为电脑本地时区。

| 阶段 | 模型可用的历史 | 购买答案范围 |
|---|---|---|
| 验证 | 2019-10-01 ≤ 时间 < 2019-11-11 | 2019-11-11 ≤ 时间 < 2019-11-21 |
| 最终测试 | 2019-10-01 ≤ 时间 < 2019-11-21 | 2019-11-21 ≤ 时间 < 2019-12-01 |

- 验证期选择权重、邻居数量等参数；最终参数确定后，才使用测试前历史重新训练并评价测试期。
- 未来购买只作为答案，不能把目标表的未来次数、未来时间或历史关系标签作为训练特征。
- 不按未来活跃度筛掉用户，不按未来商品热度筛掉商品。
- 原清洗阶段曾使用全周期信息补齐品牌。为避免它污染时间评估，本次 CF 文件完全不使用品牌、类别填充或全周期冲突标记，只用行为、标识和时间。

## 2. 生成命令

在 PowerShell 中执行：

```powershell
cd D:\user_behavior

.\.venv\Scripts\python.exe scripts\prepare_model_data.py `
  "D:\user_behavior\processed\clean_events_2019_10_11_1pct.parquet" `
  --output-dir "D:\user_behavior\model_data"
```

文件已生成时脚本会拒绝覆盖。需要重建时使用新的目录，或明确添加 `--overwrite`。

默认交互分数：

```text
interaction_score
  = 1 × ln(1 + view_count)
  + 3 × cart_flag
  + 5 × purchase_flag
```

`cart_flag`、`purchase_flag` 是是否发生过该行为，不随重复事件线性增加；这些只是第一轮默认值，不代表已经选出的最佳参数。行为原始计数同时保留，后续可比较其他权重，不必重读原始 CSV。

## 3. 四个核心文件

| 文件 | 粒度 | 首次生成行数 | 用途 |
|---|---|---:|---|
| train_interactions.parquet | 每个用户＋商品一行 | 323,383 | 训练热门、Item-CF、User-CF，评价验证集 |
| validation_targets.parquet | 每个验证购买用户＋商品一行 | 3,513 | 验证答案，对应4,201条购买事件 |
| test_history_interactions.parquet | 每个测试前用户＋商品一行 | 479,478 | 参数确定后重建模型，评价测试集 |
| test_targets.parquet | 每个测试购买用户＋商品一行 | 1,976 | 测试答案，对应2,549条购买事件 |

3,513和1,976不是商品总数或订单数。同一用户在未来期内反复购买同一商品，只形成一个二元相关性答案，但次数保留在 `future_purchase_event_count`。

### 交互表字段

- `user_id`、`product_id`：用户和商品标识；
- `event_count`、`view_count`、`cart_count`、`purchase_count`：该阶段历史行为次数；
- `session_count`：该用户对该商品交互所在的不同会话数；
- `first_interaction_at`、`last_interaction_at`：该用户与该商品的历史交互时间；
- `cart_flag`、`purchase_flag`：是否加购/购买过；
- `interaction_score`：供后续协同过滤使用的初始兴趣分数，不是购买概率。

### 购买答案字段

- `future_purchase_event_count`：未来购买事件数；
- `future_purchase_session_count`：该用户购买该商品的不同会话数，不能跨商品直接求和当作订单数；
- `first_future_purchase_at`、`last_future_purchase_at`：答案期内购买时间；
- `history_view_count`、`history_cart_count`、`history_purchase_count`：截断点之前该用户与该商品的历史关系；
- `known_user`、`known_product`：是否在该阶段历史中有过任何行为，不要求购买；
- `cold_start_segment`：用户/商品是否历史已知的四种组合；
- `prior_user_product_segment`：按下述优先级划分的互斥关系；
- `is_new_purchase_for_user`：该用户历史没有买过此商品；
- `is_unseen_to_user`：该用户历史没有对此商品产生任何行为。

互斥历史关系优先级：

```text
prior_purchase                   历史购买过（也可能浏览/加购过）
prior_cart_only                  历史加购过、没有购买（也可能浏览过）
prior_view_only                  历史只有浏览，没有加购/购买
no_prior_user_product_interaction 用户历史完全没接触过该商品
```

“用户没接触过的商品”不等于“整个商品库中的新商品”，后者由 `known_product` 判断。

## 4. 辅助文件

- `validation_user_segments.parquet`、`test_user_segments.parquet`：购买答案用户的历史活跃度，分别2,388和1,355行；新用户历史次数为0；
- `train_user_history_summary.parquet`、`test_user_history_summary.parquet`：全部历史用户统计，不限于未来购买用户；
- `train_product_history_summary.parquet`、`test_product_history_summary.parquet`：只用该阶段历史生成的商品热度，不使用未来热度；
- `model_data_manifest.json`：输入 SHA-256、版本、时间边界、参数、输出规模、历史关系与冷启动分布和校验结果。

辅助用户层为 `new_user`（无历史）、`single_session`（1会话）、`low_activity`（2—3）、`medium_activity`（4—7）、`high_activity`（8及以上）。只使用各自截断点之前的数据，不复用 SQL 的全周期用户分层。

## 5. 当前答案保留策略

暂不删除历史已购买答案或冷启动答案，也不删除低频商品。后续若采用“排除历史已购买商品”的推荐策略，必须同时将对应历史已购买答案从该任务分母剔除，并单独报告被剔除比例和空答案用户数。

第一次生成的测试答案中：

| 用户—商品历史关系 | 去重答案组合数 | 购买事件数 |
|---|---:|---:|
| 历史购买过 | 160 | 338 |
| 历史加购、未购买 | 61 | 78 |
| 历史仅浏览 | 174 | 219 |
| 用户历史未接触 | 1,581 | 1,914 |
| 合计 | 1,976 | 2,549 |

这些是在固定测试开始时点之前的历史关系，与先前“同一会话购买前有没有加购”的SQL统计不是同一个口径。

纯 CF 对无历史用户应回退到热门推荐；对历史中未出现的新商品无法学习相似度。评价时应同时保留总体表现和历史已知分组结果，不应只挑容易命中的用户或商品。

## 6. 校验与复现

脚本校验：输入关键字段、事件类型、日期范围、历史计数、答案购买计数、用户—商品唯一键、时间不重叠、分数有限、输出行数和字段类型。输入文件 SHA-256 在运行前后保持一致。

运行模拟数据测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_prepare_model_data.py" -v
```

已在当前 Polars 1.44.2 环境通过5项测试，并对真实清洗数据完成生成及校验。源Parquet共1,093,742行；SQL Server先前多出的1条记录不影响本次文件。

## 7. 下一步

本次只完成模型数据准备。下一步先实现热门推荐，建立统一的Top-K评估，然后在同一套历史、答案和过滤规则下实现Item-CF和User-CF。最终测试集不用于调参。