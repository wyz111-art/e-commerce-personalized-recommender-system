/*
  用户行为指标与初版用户分层。
  查询结果一行对应一个 user_id，不会创建永久表。

  注意：purchase_count 是购买事件数，不等同于订单数。
*/

;WITH user_metrics AS
(
    SELECT
        user_id,
        COUNT_BIG(*) AS event_count,
        COUNT(DISTINCT session_key) AS session_count,
        COUNT(DISTINCT product_id) AS unique_product_count,
        COUNT(DISTINCT category_id) AS unique_category_count,
        COUNT(DISTINCT event_date) AS active_day_count,
        COUNT(DISTINCT event_month) AS active_month_count,
        SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS view_count,
        SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) AS cart_count,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchase_count,
        MIN(event_at) AS first_event_at,
        MAX(event_at) AS last_event_at
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
        CASE
            WHEN purchase_count = 0 THEN N'未购买用户'
            WHEN purchase_count = 1 THEN N'单次购买事件用户'
            ELSE N'多次购买事件用户'
        END AS purchase_segment,
        CASE WHEN active_month_count >= 2 THEN 1 ELSE 0 END AS is_cross_month_user
    FROM user_metrics
)
SELECT
    user_id,
    event_count,
    session_count,
    unique_product_count,
    unique_category_count,
    active_day_count,
    active_month_count,
    view_count,
    cart_count,
    purchase_count,
    first_event_at,
    last_event_at,
    activity_segment,
    purchase_segment,
    is_cross_month_user,
    CAST(cart_count AS DECIMAL(18, 6))
        / NULLIF(CAST(view_count AS DECIMAL(18, 6)), 0)
        AS cart_events_per_view_event,
    CAST(purchase_count AS DECIMAL(18, 6))
        / NULLIF(CAST(view_count AS DECIMAL(18, 6)), 0)
        AS purchase_events_per_view_event
FROM segmented_users
ORDER BY
    CASE activity_segment
        WHEN N'单会话用户' THEN 1
        WHEN N'低活跃用户' THEN 2
        WHEN N'中活跃用户' THEN 3
        WHEN N'高活跃用户' THEN 4
    END,
    user_id;
