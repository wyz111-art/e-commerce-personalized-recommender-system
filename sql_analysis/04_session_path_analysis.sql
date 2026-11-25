/*
  会话行为路径分析。

  目标：将每个 session_key 划入一个且仅一个互斥路径，并比较各路径规模。

  路径优先级：
  1. 浏览 -> 加购 -> 购买
  2. 浏览 -> 直接购买
  3. 浏览 -> 加购但未购买
  4. 只有浏览
  5. 无浏览但购买
  6. 只有加购
  7. 其他异常路径

  注意：这里分析的是事件路径，不代表因果关系。
*/

;WITH session_base AS
(
    SELECT
        session_key,
        user_id,
        MIN(event_at) AS session_start,
        MAX(event_at) AS session_end,
        COUNT_BIG(*) AS event_count,
        COUNT(DISTINCT product_id) AS unique_product_count,
        SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS view_count,
        SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) AS cart_count,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchase_count,
        COUNT(DISTINCT CASE
            WHEN event_type = 'purchase' THEN product_id
        END) AS unique_purchase_product_count,
        MIN(CASE WHEN event_type = 'view' THEN event_at END) AS first_view_at
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
        s.session_start,
        s.session_end,
        s.event_count,
        s.unique_product_count,
        s.view_count,
        s.cart_count,
        s.purchase_count,
        s.unique_purchase_product_count,
        s.first_view_at,
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
    FROM session_base AS s
    LEFT JOIN dbo.clean_events_2019_10_11_1pct AS e
        ON e.session_key = s.session_key
    GROUP BY
        s.session_key,
        s.user_id,
        s.session_start,
        s.session_end,
        s.event_count,
        s.unique_product_count,
        s.view_count,
        s.cart_count,
        s.purchase_count,
        s.unique_purchase_product_count,
        s.first_view_at
),
session_sequence AS
(
    SELECT
        s.session_key,
        s.user_id,
        s.session_start,
        s.session_end,
        s.event_count,
        s.unique_product_count,
        s.view_count,
        s.cart_count,
        s.purchase_count,
        s.unique_purchase_product_count,
        s.first_view_at,
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
        s.session_start,
        s.session_end,
        s.event_count,
        s.unique_product_count,
        s.view_count,
        s.cart_count,
        s.purchase_count,
        s.unique_purchase_product_count,
        s.first_view_at,
        s.first_cart_after_view_at,
        s.first_purchase_after_view_at
),
classified_sessions AS
(
    SELECT
        *,
        DATEDIFF_BIG(SECOND, session_start, session_end) AS session_duration_seconds,
        CASE
            WHEN first_purchase_after_cart_at IS NOT NULL
                THEN N'浏览→加购→购买'

            WHEN first_purchase_after_view_at IS NOT NULL
                THEN N'浏览→直接购买'

            WHEN first_view_at IS NOT NULL
             AND first_cart_after_view_at IS NOT NULL
             AND purchase_count = 0
                THEN N'浏览→加购未购买'

            WHEN view_count > 0
             AND cart_count = 0
             AND purchase_count = 0
                THEN N'只有浏览'

            WHEN view_count = 0
             AND purchase_count > 0
                THEN N'无浏览但购买'

            WHEN view_count = 0
             AND cart_count > 0
             AND purchase_count = 0
                THEN N'只有加购'

            ELSE N'其他异常路径'
        END AS session_path
    FROM session_sequence
),
path_summary AS
(
    SELECT
        session_path,
        COUNT_BIG(*) AS session_count,
        COUNT(DISTINCT user_id) AS user_count,
        SUM(event_count) AS total_event_count,
        SUM(purchase_count) AS total_purchase_event_count,
        SUM(unique_purchase_product_count) AS session_purchase_product_count,
        AVG(CAST(event_count AS DECIMAL(18, 4))) AS avg_events_per_session,
        AVG(CAST(unique_product_count AS DECIMAL(18, 4))) AS avg_products_per_session,
        AVG(CAST(session_duration_seconds AS DECIMAL(18, 4)))
            AS avg_session_duration_seconds
    FROM classified_sessions
    GROUP BY session_path
),
all_sessions AS
(
    SELECT SUM(session_count) AS total_session_count
    FROM path_summary
)
SELECT
    p.session_path,
    p.session_count,
    CAST(p.session_count AS DECIMAL(18, 6))
        / NULLIF(CAST(a.total_session_count AS DECIMAL(18, 6)), 0)
        AS session_share,
    p.user_count,
    p.total_event_count,
    p.total_purchase_event_count,
    p.session_purchase_product_count,
    p.avg_events_per_session,
    p.avg_products_per_session,
    p.avg_session_duration_seconds
FROM path_summary AS p
CROSS JOIN all_sessions AS a
ORDER BY
    CASE p.session_path
        WHEN N'只有浏览' THEN 1
        WHEN N'浏览→加购未购买' THEN 2
        WHEN N'浏览→加购→购买' THEN 3
        WHEN N'浏览→直接购买' THEN 4
        WHEN N'无浏览但购买' THEN 5
        WHEN N'只有加购' THEN 6
        ELSE 7
    END;
