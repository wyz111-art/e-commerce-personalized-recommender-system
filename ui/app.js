const state = { data: null, user: null, model: "user_cf_n100" };
const $ = (id) => document.getElementById(id);
const fmt = (value) => new Intl.NumberFormat("zh-CN").format(value || 0);
const pct = (value) => `${((value || 0) * 100).toFixed(2)}%`;
const date = (value) => value ? value.replace("T", " ").slice(0, 16) : "-";

function reasonText(row, model) {
  if (row.source === "popularity_fallback") return "个性化候选不足，由训练期购买热门榜补齐";
  if (model === "item_cf_n100") return row.reason_product_id ? `基于历史商品 ${row.reason_product_id} 的共现关系` : "根据历史商品相似关系";
  if (model === "user_cf_n100") return row.reason_user_id ? `相似访客 ${row.reason_user_id} 的行为提供依据` : "根据相似访客行为";
  const routes = [];
  if (row.item_cf_rank) routes.push(`Item-CF 第 ${row.item_cf_rank} 名`);
  if (row.user_cf_rank) routes.push(`User-CF 第 ${row.user_cf_rank} 名`);
  return routes.length ? `两路共同候选：${routes.join("、")}` : "由两路候选融合排序";
}

function setUser(value) {
  const user = String(value).trim();
  if (!state.data.recommendations[user]) { $("searchHint").textContent = "没有找到这个用户。请输入最终测试集中有推荐结果的匿名 ID。"; return; }
  state.user = user; $("userInput").value = user; $("suggestions").hidden = true; $("emptyState").hidden = true; $("dashboard").hidden = false; renderUser();
}

function renderUser() {
  const user = state.user;
  const stats = state.data.users.find((row) => row.user_id === user) || {};
  $("userTitle").textContent = `访客 ${user}`;
  $("segmentBadge").textContent = stats.activity_segment || "未知分层";
  $("userStats").innerHTML = [[stats.session_count, "会话"], [stats.unique_product_count, "交互商品"], [stats.view_count, "浏览"], [stats.cart_count, "加购"], [stats.purchase_count, "购买"]].map(([value, label]) => `<div class="stat"><strong>${fmt(value)}</strong><span>${label}</span></div>`).join("");
  const history = state.data.history[user] || [];
  $("historyCount").textContent = `展示最近 ${history.length} 件`;
  $("historyList").innerHTML = history.length ? history.map((row) => `<div class="history-row"><div><div class="item-id">商品 ${row.product_id}</div><div class="item-meta">${date(row.last_interaction_at)} · ${row.session_count} 个会话</div></div><div class="item-meta">浏览 ${row.view_count}<br>加购 ${row.cart_count} · 买 ${row.purchase_count}</div></div>`).join("") : `<p class="muted">该访客在推荐截点前没有可用历史，模型使用热门补齐。</p>`;
  renderPurchases(); renderModelSwitch(); renderRecommendations();
}

function renderPurchases() {
  const purchases = state.data.purchases[state.user] || [];
  $("purchaseList").innerHTML = purchases.length ? purchases.map((row) => { const cls = row.eligible_for_evaluation ? "purchase-row actual" : "purchase-row"; const tag = row.eligible_for_evaluation ? "测试答案" : row.prior_purchase ? "历史已购，过滤" : "历史已加购，过滤"; const cold = !row.in_train_catalog ? " cold" : ""; return `<div class="${cls}"><div><div class="item-id">商品 ${row.product_id}</div><div class="item-meta">${date(row.first_future_purchase_at)} UTC</div></div><div class="purchase-tag${cold}">${tag}</div></div>`; }).join("") : `<p class="muted">该访客在测试窗口没有后续购买记录。</p>`;
}

function renderModelSwitch() {
  $("modelSwitch").innerHTML = state.data.models.map((model) => `<button type="button" data-model="${model.id}" class="${model.id === state.model ? "active" : ""}">${model.label}</button>`).join("");
  $("modelSwitch").querySelectorAll("button").forEach((button) => button.addEventListener("click", () => { state.model = button.dataset.model; renderModelSwitch(); renderRecommendations(); }));
  const explain = { user_cf_n100: "User-CF：先找行为相似的访客，再看这些访客交互过哪些商品。", item_cf_n100: "改进版 Item-CF：用用户历史商品的共现关系，已启用共现收缩与最低共同用户门槛。", rrf_fusion: "两路融合：把 Item-CF 与 User-CF 的候选按名次合并；热门榜只在个性化候选不足时补齐。" };
  $("modelExplainer").textContent = explain[state.model];
}

function renderRecommendations() {
  const rows = state.data.recommendations[state.user]?.[state.model] || [];
  $("recommendations").innerHTML = rows.map((row) => `<div class="rec-row ${row.is_actual_purchase ? "hit" : ""}"><div class="rank">${String(row.rank).padStart(2, "0")}</div><div><div class="item-id">商品 ${row.product_id}${row.is_actual_purchase ? '<span class="hit-label">后来购买</span>' : ""}</div><div class="item-meta">${row.source === "popularity_fallback" ? "热门补齐" : "个性化候选"}</div></div><div class="reason">${reasonText(row, state.model)}</div><div class="rec-score">${row.score?.toFixed(3) || "-"}<small>排序分数</small></div></div>`).join("") || `<p class="muted">暂无推荐结果。</p>`;
}

function renderMetrics() {
  $("metricCards").innerHTML = state.data.models.map((model) => { const row = state.data.metrics[model.id]?.overall || {}; return `<article class="metric-card"><h3>${model.label}</h3><div class="metric-big">${pct(row.recall_at_k)} <span>Recall@10</span></div><div class="metric-line"><span>NDCG@10</span><strong>${(row.ndcg_at_k || 0).toFixed(4)}</strong></div><div class="metric-line"><span>商品覆盖率</span><strong>${pct(row.catalog_coverage)}</strong></div><div class="metric-line"><span>热门补齐位</span><strong>${pct(row.fallback_slot_share)}</strong></div></article>`; }).join("");
}

function renderSuggestions(value) {
  const query = value.trim(); if (!query) { $("suggestions").hidden = true; return; }
  const matches = state.data.users.filter((row) => row.user_id.includes(query)).slice(0, 6);
  $("suggestions").innerHTML = matches.map((row) => `<button type="button" class="suggestion" data-user="${row.user_id}"><strong>${row.user_id}</strong><small>${row.activity_segment} · ${row.unique_product_count} 件商品</small></button>`).join("");
  $("suggestions").hidden = !matches.length; $("suggestions").querySelectorAll("button").forEach((button) => button.addEventListener("click", () => setUser(button.dataset.user)));
}

async function init() {
  state.data = await (await fetch("data/recommender_data.json")).json();
  $("loadBtn").addEventListener("click", () => setUser($("userInput").value));
  $("userInput").addEventListener("input", (event) => renderSuggestions(event.target.value));
  $("userInput").addEventListener("keydown", (event) => { if (event.key === "Enter") setUser(event.target.value); });
  $("exampleButtons").innerHTML = state.data.examples.map((user) => `<button type="button" data-user="${user}">${user}</button>`).join("");
  $("exampleButtons").querySelectorAll("button").forEach((button) => button.addEventListener("click", () => setUser(button.dataset.user)));
  renderMetrics(); setUser(state.data.examples[0]);
}

init().catch((error) => { $("searchHint").textContent = `数据加载失败：${error.message}`; });
