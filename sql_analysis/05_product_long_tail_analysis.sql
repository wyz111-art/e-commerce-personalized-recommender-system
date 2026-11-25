/*
  商品长尾与跨月商品结构分析。

  本脚本返回三个结果集：
  1. 商品购买覆盖和低频商品规模；
  2. 头部 1% / 5% / 10% 商品的购买事件贡献；
  3. 10 月与 11 月商品的新旧结构。

  注意：purchase_count 是购买事件数，不等同于订单量或购买件数。
*/

/* ============================================================
   结果集 1：商品购买覆盖与低频分布
   ============================================================ */

;WITH product_metrics AS
(
    SELECT
        product_id,
        COUNT_BIG(*) AS event_count,
        SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS view_count,
        SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) AS cart_count,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchase_count,
        COUNT(DISTINCT user_id) AS interaction_user_count,
        COUNT(DISTINCT CASE
            WHEN event_type = 'purchase' THEN user_id
        END) AS purchase_user_count,
        COUNT(DISTINCT event_month) AS active_month_count
    FROM dbo.clean_events_2019_10_11_1pct
    GROUP BY product_id
)
SELECT
    COUNT_BIG(*) AS total_products,
    SUM(CASE WHEN view_count > 0 THEN 1 ELSE 0 END) AS viewed_products,
    SUM(CASE WHEN cart_count > 0 THEN 1 ELSE 0 END) AS carted_products,
    SUM(CASE WHEN purchase_count > 0 THEN 1 ELSE 0 END) AS purchased_products,
    SUM(CASE WHEN purchase_count = 0 THEN 1 ELSE 0 END) AS never_purchased_products,
    SUM(CASE WHEN purchase_count = 1 THEN 1 ELSE 0 END) AS single_purchase_event_products,
    SUM(CASE WHEN purchase_count BETWEEN 2 AND 5 THEN 1 ELSE 0 END)
        AS two_to_five_purchase_event_products,
    SUM(CASE WHEN purchase_count > 5 THEN 1 ELSE 0 END)
        AS over_five_purchase_event_products,
    SUM(CASE WHEN active_month_count >= 2 THEN 1 ELSE 0 END) AS cross_month_products,
    CAST(SUM(CASE WHEN purchase_count > 0 THEN 1 ELSE 0 END) AS DECIMAL(18, 6))
        / NULLIF(CAST(COUNT_BIG(*) AS DECIMAL(18, 6)), 0)
        AS purchased_product_rate,
    CAST(SUM(CASE WHEN purchase_count = 1 THEN 1 ELSE 0 END) AS DECIMAL(18, 6))
        / NULLIF(
            CAST(SUM(CASE WHEN purchase_count > 0 THEN 1 ELSE 0 END) AS DECIMAL(18, 6)),
            0
        ) AS single_purchase_share_among_purchased_products
FROM product_metrics;


/* ============================================================
   结果集 2：头部商品购买贡献

   排名总体商品，以购买事件数从高到低排序。
   没有购买的商品也在总体商品分母中，因此这里衡量的是：
   全部商品中最热门的 1% / 5% / 10% 贡献了多少购买事件。
   ============================================================ */

;WITH product_purchase AS
(
    SELECT
        product_id,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchase_count
    FROM dbo.clean_events_2019_10_11_1pct
    GROUP BY product_id
),
ranked_products AS
(
    SELECT
        product_id,
        purchase_count,
        ROW_NUMBER() OVER
        (
            ORDER BY purchase_count DESC, product_id
        ) AS product_rank,
        COUNT_BIG(*) OVER () AS total_products,
        SUM(purchase_count) OVER () AS total_purchase_events
    FROM product_purchase
),
thresholds AS
(
    SELECT N'Top 1%' AS head_group, CAST(0.01 AS DECIMAL(5, 4)) AS cutoff
    UNION ALL
    SELECT N'Top 5%', CAST(0.05 AS DECIMAL(5, 4))
    UNION ALL
    SELECT N'Top 10%', CAST(0.10 AS DECIMAL(5, 4))
)
SELECT
    t.head_group,
    MAX(r.total_products) AS total_products,
    CEILING(MAX(r.total_products) * t.cutoff) AS head_product_count,
    SUM(CASE
            WHEN r.product_rank <= CEILING(r.total_products * t.cutoff)
            THEN r.purchase_count
            ELSE 0
        END) AS head_purchase_events,
    MAX(r.total_purchase_events) AS total_purchase_events,
    CAST(
        SUM(CASE
                WHEN r.product_rank <= CEILING(r.total_products * t.cutoff)
                THEN r.purchase_count
                ELSE 0
            END)
        AS DECIMAL(18, 6)
    ) / NULLIF(CAST(MAX(r.total_purchase_events) AS DECIMAL(18, 6)), 0)
        AS purchase_event_contribution
FROM ranked_products AS r
CROSS JOIN thresholds AS t
GROUP BY
    t.head_group,
    t.cutoff
ORDER BY t.cutoff;


/* ============================================================
   结果集 3：10 月与 11 月商品结构

   October only：只在 10 月出现；
   Cross month：10 月和 11 月均出现；
   November new：11 月首次出现，10 月没有行为。
   ============================================================ */

;WITH product_month_presence AS
(
    SELECT
        product_id,
        MAX(CASE WHEN event_month = '2019-10' THEN 1 ELSE 0 END) AS appeared_in_october,
        MAX(CASE WHEN event_month = '2019-11' THEN 1 ELSE 0 END) AS appeared_in_november,
        SUM(CASE
                WHEN event_month = '2019-10'
                 AND event_type = 'purchase'
                THEN 1 ELSE 0
            END) AS october_purchase_count,
        SUM(CASE
                WHEN event_month = '2019-11'
                 AND event_type = 'purchase'
                THEN 1 ELSE 0
            END) AS november_purchase_count
    FROM dbo.clean_events_2019_10_11_1pct
    GROUP BY product_id
),
classified_products AS
(
    SELECT
        *,
        CASE
            WHEN appeared_in_october = 1 AND appeared_in_november = 1
                THEN N'跨月商品'
            WHEN appeared_in_october = 1 AND appeared_in_november = 0
                THEN N'仅10月出现'
            WHEN appeared_in_october = 0 AND appeared_in_november = 1
                THEN N'11月新出现'
            ELSE N'异常'
        END AS product_month_segment
    FROM product_month_presence
)
SELECT
    product_month_segment,
    COUNT_BIG(*) AS product_count,
    SUM(october_purchase_count) AS october_purchase_events,
    SUM(november_purchase_count) AS november_purchase_events,
    SUM(october_purchase_count + november_purchase_count) AS total_purchase_events,
    SUM(CASE
            WHEN october_purchase_count + november_purchase_count > 0
            THEN 1 ELSE 0
        END) AS purchased_product_count,
    CAST(
        SUM(CASE
                WHEN october_purchase_count + november_purchase_count > 0
                THEN 1 ELSE 0
            END)
        AS DECIMAL(18, 6)
    ) / NULLIF(CAST(COUNT_BIG(*) AS DECIMAL(18, 6)), 0)
        AS purchased_product_rate
FROM classified_products
GROUP BY product_month_segment
ORDER BY
    CASE product_month_segment
        WHEN N'仅10月出现' THEN 1
        WHEN N'跨月商品' THEN 2
        WHEN N'11月新出现' THEN 3
        ELSE 4
    END;
