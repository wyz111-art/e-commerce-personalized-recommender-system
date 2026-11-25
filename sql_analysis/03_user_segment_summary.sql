/*
  用户分层汇总。
  按活跃度层级和是否跨月汇总，购买率才具有比较意义。
  查询不会创建永久表。
*/

;WITH user_metrics AS
(
    SELECT
        user_id,
        COUNT_BIG(*) AS event_count,
        COUNT(DISTINCT session_key) AS session_count,
        COUNT(DISTINCT product_id) AS unique_product_count,
        COUNT(DISTINCT event_date) AS active_day_count,
        COUNT(DISTINCT event_month) AS active_month_count,
        SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS view_count,
        SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) AS cart_count,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchase_count
    FROM dbo.clean_events_2019_10_11_1pct
    GROUP BY user_id
),
segmented_users AS
(
    SELECT
        *,
        CASE
            WHEN session_count = 1 THEN N'单会话用户'
            WHEN session_count BETWEEN 2 AND 3 THEN N'低活跃用户'
            WHEN session_count BETWEEN 4 AND 7 THEN N'中活跃用户'
            ELSE N'高活跃用户'
        END AS activity_segment,
        CASE WHEN active_month_count >= 2 THEN 1 ELSE 0 END AS is_cross_month_user
    FROM user_metrics
)
SELECT
    activity_segment,
    is_cross_month_user,
    COUNT_BIG(*) AS user_count,
    SUM(CASE WHEN purchase_count > 0 THEN 1 ELSE 0 END) AS purchase_user_count,
    SUM(CASE WHEN purchase_count > 1 THEN 1 ELSE 0 END) AS multi_purchase_event_user_count,
    SUM(event_count) AS total_event_count,
    SUM(view_count) AS total_view_count,
    SUM(cart_count) AS total_cart_count,
    SUM(purchase_count) AS total_purchase_count,
    AVG(CAST(event_count AS DECIMAL(18, 4))) AS avg_events_per_user,
    AVG(CAST(session_count AS DECIMAL(18, 4))) AS avg_sessions_per_user,
    AVG(CAST(unique_product_count AS DECIMAL(18, 4))) AS avg_products_per_user,
    AVG(CAST(active_day_count AS DECIMAL(18, 4))) AS avg_active_days,
    CAST(SUM(CASE WHEN purchase_count > 0 THEN 1 ELSE 0 END) AS DECIMAL(18, 6))
        / NULLIF(CAST(COUNT_BIG(*) AS DECIMAL(18, 6)), 0)
        AS user_purchase_rate,
    CAST(SUM(CASE WHEN purchase_count > 1 THEN 1 ELSE 0 END) AS DECIMAL(18, 6))
        / NULLIF(CAST(COUNT_BIG(*) AS DECIMAL(18, 6)), 0)
        AS multi_purchase_event_user_rate
FROM segmented_users
GROUP BY
    activity_segment,
    is_cross_month_user
ORDER BY
    CASE activity_segment
        WHEN N'单会话用户' THEN 1
        WHEN N'低活跃用户' THEN 2
        WHEN N'中活跃用户' THEN 3
        WHEN N'高活跃用户' THEN 4
    END,
    is_cross_month_user;
