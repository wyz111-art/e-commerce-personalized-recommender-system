/*
  时间切分可用性分析。

  候选切分：
  - 训练集：2019-10-01 00:00:00 至 2019-11-11 00:00:00 之前
  - 验证集：2019-11-11 00:00:00 至 2019-11-21 00:00:00 之前
  - 测试集：2019-11-21 00:00:00 至数据结束

  本脚本返回四个结果集，只做统计，不修改数据：
  1. 各时间段的数据规模；
  2. 验证/测试购买用户与购买商品的训练历史覆盖；
  3. 测试购买事件的用户/商品冷启动组合；
  4. 测试期标准购买会话及购买前信号可用性。
*/

/* 结果集 1：训练、验证、测试规模 */

;WITH split_events AS
(
    SELECT
        *,
        CASE
            WHEN event_at < '2019-11-11T00:00:00' THEN N'训练集'
            WHEN event_at < '2019-11-21T00:00:00' THEN N'验证集'
            ELSE N'测试集'
        END AS data_split
    FROM dbo.clean_events_2019_10_11_1pct
),
split_summary AS
(
    SELECT
        data_split,
        COUNT_BIG(*) AS event_count,
        COUNT(DISTINCT user_id) AS user_count,
        COUNT(DISTINCT product_id) AS product_count,
        COUNT(DISTINCT session_key) AS session_count,
        SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS view_event_count,
        SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) AS cart_event_count,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchase_event_count,
        COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN user_id END)
            AS purchase_user_count,
        COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN product_id END)
            AS purchased_product_count,
        COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN session_key END)
            AS purchase_session_count,
        MIN(event_at) AS min_event_at,
        MAX(event_at) AS max_event_at
    FROM split_events
    GROUP BY data_split
)
SELECT
    data_split,
    event_count,
    user_count,
    product_count,
    session_count,
    view_event_count,
    cart_event_count,
    purchase_event_count,
    purchase_user_count,
    purchased_product_count,
    purchase_session_count,
    min_event_at,
    max_event_at
FROM split_summary
ORDER BY
    CASE data_split
        WHEN N'训练集' THEN 1
        WHEN N'验证集' THEN 2
        WHEN N'测试集' THEN 3
    END;


/*
  结果集 2：验证/测试购买对象的历史覆盖。

  验证集使用 11 月 11 日之前的历史；
  测试集使用 11 月 21 日之前的全部历史（训练集 + 验证集）。
*/

;WITH user_first_seen AS
(
    SELECT
        user_id,
        MIN(event_at) AS first_event_at
    FROM dbo.clean_events_2019_10_11_1pct
    GROUP BY user_id
),
product_first_seen AS
(
    SELECT
        product_id,
        MIN(event_at) AS first_event_at
    FROM dbo.clean_events_2019_10_11_1pct
    GROUP BY product_id
),
future_purchases AS
(
    SELECT
        CASE
            WHEN event_at < '2019-11-21T00:00:00' THEN N'验证集'
            ELSE N'测试集'
        END AS data_split,
        CASE
            WHEN event_at < '2019-11-21T00:00:00'
                THEN CAST('2019-11-11T00:00:00' AS DATETIME2(6))
            ELSE CAST('2019-11-21T00:00:00' AS DATETIME2(6))
        END AS split_start_at,
        user_id,
        product_id
    FROM dbo.clean_events_2019_10_11_1pct
    WHERE event_type = 'purchase'
      AND event_at >= '2019-11-11T00:00:00'
),
purchase_users AS
(
    SELECT DISTINCT data_split, split_start_at, user_id
    FROM future_purchases
),
purchase_products AS
(
    SELECT DISTINCT data_split, split_start_at, product_id
    FROM future_purchases
),
user_coverage AS
(
    SELECT
        u.data_split,
        COUNT_BIG(*) AS purchase_user_count,
        SUM(CASE
                WHEN f.first_event_at < u.split_start_at THEN 1
                ELSE 0
            END) AS known_purchase_user_count
    FROM purchase_users AS u
    LEFT JOIN user_first_seen AS f
        ON f.user_id = u.user_id
    GROUP BY u.data_split
),
product_coverage AS
(
    SELECT
        p.data_split,
        COUNT_BIG(*) AS purchased_product_count,
        SUM(CASE
                WHEN f.first_event_at < p.split_start_at THEN 1
                ELSE 0
            END) AS known_purchased_product_count
    FROM purchase_products AS p
    LEFT JOIN product_first_seen AS f
        ON f.product_id = p.product_id
    GROUP BY p.data_split
)
SELECT
    u.data_split,
    u.purchase_user_count,
    u.known_purchase_user_count,
    u.purchase_user_count - u.known_purchase_user_count AS new_purchase_user_count,
    CAST(u.known_purchase_user_count AS DECIMAL(18, 6))
        / NULLIF(CAST(u.purchase_user_count AS DECIMAL(18, 6)), 0)
        AS purchase_user_history_coverage,
    p.purchased_product_count,
    p.known_purchased_product_count,
    p.purchased_product_count - p.known_purchased_product_count
        AS new_purchased_product_count,
    CAST(p.known_purchased_product_count AS DECIMAL(18, 6))
        / NULLIF(CAST(p.purchased_product_count AS DECIMAL(18, 6)), 0)
        AS purchased_product_history_coverage
FROM user_coverage AS u
INNER JOIN product_coverage AS p
    ON p.data_split = u.data_split
ORDER BY
    CASE u.data_split
        WHEN N'验证集' THEN 1
        WHEN N'测试集' THEN 2
    END;


/*
  结果集 3：测试购买事件的用户/商品冷启动组合。
  测试前历史包括 11 月 21 日之前的训练集和验证集。
*/

;WITH pretest_users AS
(
    SELECT DISTINCT user_id
    FROM dbo.clean_events_2019_10_11_1pct
    WHERE event_at < '2019-11-21T00:00:00'
),
pretest_products AS
(
    SELECT DISTINCT product_id
    FROM dbo.clean_events_2019_10_11_1pct
    WHERE event_at < '2019-11-21T00:00:00'
),
test_purchase_events AS
(
    SELECT
        e.user_id,
        e.product_id,
        e.session_key,
        CASE WHEN u.user_id IS NOT NULL THEN 1 ELSE 0 END AS is_known_user,
        CASE WHEN p.product_id IS NOT NULL THEN 1 ELSE 0 END AS is_known_product
    FROM dbo.clean_events_2019_10_11_1pct AS e
    LEFT JOIN pretest_users AS u
        ON u.user_id = e.user_id
    LEFT JOIN pretest_products AS p
        ON p.product_id = e.product_id
    WHERE e.event_type = 'purchase'
      AND e.event_at >= '2019-11-21T00:00:00'
),
classified_events AS
(
    SELECT
        *,
        CASE
            WHEN is_known_user = 1 AND is_known_product = 1
                THEN N'老用户+老商品'
            WHEN is_known_user = 0 AND is_known_product = 1
                THEN N'新用户+老商品'
            WHEN is_known_user = 1 AND is_known_product = 0
                THEN N'老用户+新商品'
            ELSE N'新用户+新商品'
        END AS cold_start_segment
    FROM test_purchase_events
),
total_events AS
(
    SELECT COUNT_BIG(*) AS total_purchase_events
    FROM classified_events
)
SELECT
    c.cold_start_segment,
    COUNT_BIG(*) AS purchase_event_count,
    COUNT(DISTINCT c.user_id) AS purchase_user_count,
    COUNT(DISTINCT c.product_id) AS purchased_product_count,
    COUNT(DISTINCT c.session_key) AS purchase_session_count,
    CAST(COUNT_BIG(*) AS DECIMAL(18, 6))
        / NULLIF(CAST(t.total_purchase_events AS DECIMAL(18, 6)), 0)
        AS purchase_event_share
FROM classified_events AS c
CROSS JOIN total_events AS t
GROUP BY
    c.cold_start_segment,
    t.total_purchase_events
ORDER BY
    CASE c.cold_start_segment
        WHEN N'老用户+老商品' THEN 1
        WHEN N'新用户+老商品' THEN 2
        WHEN N'老用户+新商品' THEN 3
        WHEN N'新用户+新商品' THEN 4
    END;


/*
  结果集 4：测试购买会话的购买前信号可用性。
  用户历史判断使用测试开始前（11 月 21 日之前）的全部数据。
*/

;WITH pretest_users AS
(
    SELECT DISTINCT user_id
    FROM dbo.clean_events_2019_10_11_1pct
    WHERE event_at < '2019-11-21T00:00:00'
),
test_purchase_sessions AS
(
    SELECT
        session_key,
        user_id,
        MIN(event_at) AS first_purchase_at,
        COUNT_BIG(*) AS purchase_event_count,
        COUNT(DISTINCT product_id) AS purchased_product_count
    FROM dbo.clean_events_2019_10_11_1pct
    WHERE event_type = 'purchase'
      AND event_at >= '2019-11-21T00:00:00'
    GROUP BY
        session_key,
        user_id
),
session_signals AS
(
    SELECT
        p.session_key,
        p.user_id,
        p.first_purchase_at,
        p.purchase_event_count,
        p.purchased_product_count,
        MAX(CASE
                WHEN e.event_type = 'view'
                 AND e.event_at < p.first_purchase_at
                THEN 1 ELSE 0
            END) AS has_view_before_purchase,
        MAX(CASE
                WHEN e.event_type = 'cart'
                 AND e.event_at < p.first_purchase_at
                THEN 1 ELSE 0
            END) AS has_cart_before_purchase,
        SUM(CASE
                WHEN e.event_at < p.first_purchase_at
                THEN 1 ELSE 0
            END) AS pre_purchase_event_count,
        COUNT(DISTINCT CASE
                WHEN e.event_at < p.first_purchase_at
                THEN e.product_id
            END) AS pre_purchase_product_count
    FROM test_purchase_sessions AS p
    LEFT JOIN dbo.clean_events_2019_10_11_1pct AS e
        ON e.session_key = p.session_key
    GROUP BY
        p.session_key,
        p.user_id,
        p.first_purchase_at,
        p.purchase_event_count,
        p.purchased_product_count
),
classified_sessions AS
(
    SELECT
        s.*,
        CASE WHEN t.user_id IS NOT NULL THEN 1 ELSE 0 END AS has_pretest_user_history,
        CASE
            WHEN s.has_view_before_purchase = 1
             AND s.has_cart_before_purchase = 1
                THEN N'购买前有浏览和加购'
            WHEN s.has_view_before_purchase = 1
             AND s.has_cart_before_purchase = 0
                THEN N'购买前只有浏览'
            WHEN s.has_view_before_purchase = 0
             AND s.has_cart_before_purchase = 1
                THEN N'购买前只有加购'
            ELSE N'购买前无浏览和加购'
        END AS pre_purchase_signal_segment
    FROM session_signals AS s
    LEFT JOIN pretest_users AS t
        ON t.user_id = s.user_id
)
SELECT
    pre_purchase_signal_segment,
    has_pretest_user_history,
    COUNT_BIG(*) AS purchase_session_count,
    COUNT(DISTINCT user_id) AS purchase_user_count,
    SUM(purchase_event_count) AS purchase_event_count,
    SUM(purchased_product_count) AS session_purchased_product_count,
    AVG(CAST(pre_purchase_event_count AS DECIMAL(18, 4)))
        AS avg_pre_purchase_events,
    AVG(CAST(pre_purchase_product_count AS DECIMAL(18, 4)))
        AS avg_pre_purchase_products
FROM classified_sessions
GROUP BY
    pre_purchase_signal_segment,
    has_pretest_user_history
ORDER BY
    CASE pre_purchase_signal_segment
        WHEN N'购买前有浏览和加购' THEN 1
        WHEN N'购买前只有浏览' THEN 2
        WHEN N'购买前只有加购' THEN 3
        ELSE 4
    END,
    has_pretest_user_history DESC;
