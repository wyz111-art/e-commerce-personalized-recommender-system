/*
  创建清洗事件明细表。
  在使用 DBeaver 导入 CSV 之前执行一次。

  安全策略：如果同名表已经存在，脚本报错并停止，不会删除或覆盖。
*/

IF OBJECT_ID(N'dbo.clean_events_2019_10_11_1pct', N'U') IS NOT NULL
BEGIN
    THROW 50001, N'表 dbo.clean_events_2019_10_11_1pct 已存在，请勿重复创建或覆盖。', 1;
END;

CREATE TABLE dbo.clean_events_2019_10_11_1pct
(
    event_at                   DATETIME2(6) NOT NULL,
    event_date                 DATE NOT NULL,
    event_month                CHAR(7) NOT NULL,
    event_hour                 TINYINT NOT NULL,
    event_type                 VARCHAR(20) NOT NULL,
    product_id                 BIGINT NOT NULL,
    category_id                BIGINT NOT NULL,
    category_code              VARCHAR(255) NULL,
    category_code_missing      BIT NOT NULL,
    brand_clean                VARCHAR(255) NOT NULL,
    brand_missing_original     BIT NOT NULL,
    price                      DECIMAL(18, 4) NOT NULL,
    price_zero_flag            BIT NOT NULL,
    price_negative_flag        BIT NOT NULL,
    user_id                    BIGINT NOT NULL,
    user_session               VARCHAR(100) NOT NULL,
    session_key                VARCHAR(150) NOT NULL,
    source_file                VARCHAR(100) NOT NULL,
    metadata_conflict_flag     BIT NOT NULL
);

SELECT
    SCHEMA_NAME(schema_id) AS schema_name,
    name AS table_name
FROM sys.tables
WHERE object_id = OBJECT_ID(N'dbo.clean_events_2019_10_11_1pct');