# 中文推荐工作台

这是基于最终测试结果的静态演示界面，不会在浏览器中重新训练模型。

## 启动

```powershell
Set-Location D:\user_behavior
.\.venv\Scripts\python.exe -B ui\server.py --port 8765
```

然后打开：`http://127.0.0.1:8765`

## 重新生成界面数据

当 `reports\cf_final_test` 中的最终测试结果更新后，重新执行：

```powershell
.\.venv\Scripts\python.exe -B scripts\build_recommender_ui_data.py
```

界面当前展示：

- 输入或搜索匿名用户 ID；
- 推荐截点前最近 12 件按商品聚合的行为；
- 改进版 Item-CF、User-CF、两路融合三张 Top 10 榜单；
- 推荐来源、推荐理由、排序分数和后续购买命中标记；
- 测试窗口内的后续购买，以及历史已购/已加购过滤提示；
- 最终测试集的 Recall@10、NDCG@10、商品覆盖率和热门补齐位。

商品 ID、用户 ID 和排序分数均保持匿名语义。排序分数不是购买概率，后续购买只用于离线核对。
