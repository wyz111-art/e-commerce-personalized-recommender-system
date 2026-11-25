/*
  CSV 导入完成并核对行数后执行。
  索引用于加速会话漏斗、用户分层和商品分析。
*/

IF OBJECT_ID(N'dbo.clean_events_2019_10_11_1pct', N'U') IS NULL
BEGIN
    THROW 50002, N'找不到 dbo.clean_events_2019_10_11_1pct，请先建表并导入 CSV。', 1;
END;

IF NOT EXISTS
(
    SELECT 1
    FROM sys.indexes
    WHERE object_id = OBJECT_ID(N'dbo.clean_events_2019_10_11_1pct')
      AND name = N'IX_clean_events_session_event'
)
BEGIN
    CREATE INDEX IX_clean_events_session_event
        ON dbo.clean_events_2019_10_11_1pct
        (
            session_key,
            event_at
        )
        INCLUDE
        (
            user_id,
            event_type,
            product_id
        );
END;

IF NOT EXISTS
(
    SELECT 1
    FROM sys.indexes
    WHERE object_id = OBJECT_ID(N'dbo.clean_events_2019_10_11_1pct')
      AND name = N'IX_clean_events_user_event'
)
BEGIN
    CREATE INDEX IX_clean_events_user_event
        ON dbo.clean_events_2019_10_11_1pct
        (
            user_id,
            event_at
        )
        INCLUDE
        (
            session_key,
            event_type,
            product_id,
            category_id
        );
END;

IF NOT EXISTS
(
    SELECT 1
    FROM sys.indexes
    WHERE object_id = OBJECT_ID(N'dbo.clean_events_2019_10_11_1pct')
      AND name = N'IX_clean_events_product_event'
)
BEGIN
    CREATE INDEX IX_clean_events_product_event
        ON dbo.clean_events_2019_10_11_1pct
        (
            product_id,
            event_at
        )
        INCLUDE
        (
            user_id,
            session_key,
            event_type,
            category_id,
            brand_clean,
            price
        );
END;

SELECT
    name AS index_name,
    type_desc
FROM sys.indexes
WHERE object_id = OBJECT_ID(N'dbo.clean_events_2019_10_11_1pct')
  AND name IS NOT NULL
ORDER BY name;
