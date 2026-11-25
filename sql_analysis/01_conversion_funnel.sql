/*
  会话级整体转化漏斗。
  统计范围：2019 年 10 月至 11 月整体，不按月份拆分。

  严格漏斗口径：
  浏览会话 -> 浏览后加购会话 -> 加购后购买会话。
*/

;WITH session_first_events AS
(
    SELECT
        session_key,
        user_id,
        MIN(CASE WHEN event_type = 'view' THEN event_at END) AS first_view_at,
        MIN(CASE WHEN event_type = 'purchase' THEN event_at END) AS first_purchase_at
    FROM dbo.clean_events_2019_10_11_1pct
    GROUP BY
        session_key,
        user_id
),
session_after_view AS
(
    SELECT
        s.session_key,
        s.user_id,
        s.first_view_at,
        s.first_purchase_at,
        MIN(CASE
                WHEN e.event_type = 'cart'
                 AND e.event_at > s.first_view_at
                THEN e.event_at
            END) AS first_cart_after_view_at,
        MIN(CASE
                WHEN e.event_type = 'purchase'
                 AND e.event_at > s.first_view_at
                THEN e.event_at
            END) AS first_purchase_after_view_at
    FROM session_first_events AS s
    LEFT JOIN dbo.clean_events_2019_10_11_1pct AS e
        ON e.session_key = s.session_key
    GROUP BY
        s.session_key,
        s.user_id,
        s.first_view_at,
        s.first_purchase_at
),
session_funnel AS
(
    SELECT
        s.session_key,
        s.user_id,
        s.first_view_at,
        s.first_purchase_at,
        s.first_cart_after_view_at,
        s.first_purchase_after_view_at,
        MIN(CASE
                WHEN e.event_type = 'purchase'
                 AND e.event_at > s.first_cart_after_view_at
                THEN e.event_at
            END) AS first_purchase_after_cart_at
    FROM session_after_view AS s
    LEFT JOIN dbo.clean_events_2019_10_11_1pct AS e
        ON e.session_key = s.session_key
    GROUP BY
        s.session_key,
        s.user_id,
        s.first_view_at,
        s.first_purchase_at,
        s.first_cart_after_view_at,
        s.first_purchase_after_view_at
),
funnel_counts AS
(
    SELECT
        COUNT_BIG(*) AS total_sessions,
        SUM(CASE WHEN first_view_at IS NOT NULL THEN 1 ELSE 0 END) AS view_sessions,
        SUM(CASE WHEN first_cart_after_view_at IS NOT NULL THEN 1 ELSE 0 END)
            AS view_to_cart_sessions,
        SUM(CASE WHEN first_purchase_after_cart_at IS NOT NULL THEN 1 ELSE 0 END)
            AS view_cart_purchase_sessions,
        SUM(CASE WHEN first_purchase_at IS NOT NULL THEN 1 ELSE 0 END)
            AS purchase_sessions,
        SUM(CASE WHEN first_purchase_after_view_at IS NOT NULL THEN 1 ELSE 0 END)
            AS view_to_purchase_sessions,
        SUM(CASE
                WHEN first_purchase_after_view_at IS NOT NULL
                 AND
                 (
                     first_cart_after_view_at IS NULL
                     OR first_purchase_after_view_at < first_cart_after_view_at
                 )
                THEN 1 ELSE 0
            END) AS direct_view_to_purchase_sessions
    FROM session_funnel
)
SELECT
    total_sessions,
    view_sessions,
    view_to_cart_sessions,
    view_cart_purchase_sessions,
    purchase_sessions,
    view_to_purchase_sessions,
    direct_view_to_purchase_sessions,
    CAST(view_to_cart_sessions AS DECIMAL(18, 6))
        / NULLIF(CAST(view_sessions AS DECIMAL(18, 6)), 0)
        AS view_to_cart_rate,
    CAST(view_cart_purchase_sessions AS DECIMAL(18, 6))
        / NULLIF(CAST(view_to_cart_sessions AS DECIMAL(18, 6)), 0)
        AS cart_to_purchase_rate,
    CAST(view_to_purchase_sessions AS DECIMAL(18, 6))
        / NULLIF(CAST(view_sessions AS DECIMAL(18, 6)), 0)
        AS view_to_purchase_rate,
    CAST(purchase_sessions AS DECIMAL(18, 6))
        / NULLIF(CAST(total_sessions AS DECIMAL(18, 6)), 0)
        AS session_purchase_rate
FROM funnel_counts;
