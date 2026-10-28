-- ---------------------------------------------------------------
-- File: sql/01_staging.sql
-- Purpose: Staging layer of the warehouse. Each view reads one
--          cleaned interim table and gives its columns proper types.
-- Input:   data/interim/*.csv ({interim} is replaced by the folder
--          path when 03_build_panel.py runs this file)
-- Output:  views stg_ncfi, stg_brent, stg_gpr, stg_port_calls
-- ---------------------------------------------------------------

-- Weekly NCFI values, one row per week and route. value_source tells
-- whether the value comes from the sheet of that week ("current
-- column") or from the next week's sheet.
CREATE OR REPLACE VIEW stg_ncfi AS
SELECT
    CAST(week_date AS DATE) AS week_date,
    route,
    CAST(value AS DOUBLE) AS rate,
    value_source
FROM read_csv('{interim}/ncfi_weekly.csv', header = true);

-- Daily Brent crude price, USD per barrel (business days only).
CREATE OR REPLACE VIEW stg_brent AS
SELECT
    CAST(date AS DATE) AS day,
    CAST(brent_usd AS DOUBLE) AS brent_usd
FROM read_csv('{interim}/brent.csv', header = true);

-- Daily Geopolitical Risk Index.
CREATE OR REPLACE VIEW stg_gpr AS
SELECT
    CAST(date AS DATE) AS day,
    CAST(gprd AS DOUBLE) AS gprd
FROM read_csv('{interim}/gpr_daily.csv', header = true);

-- Daily container port calls at Ningbo, Jebel Ali and Dammam.
CREATE OR REPLACE VIEW stg_port_calls AS
SELECT
    CAST(date AS DATE) AS day,
    portid,
    portname,
    CAST(portcalls_container AS INTEGER) AS container_calls
FROM read_csv('{interim}/portwatch_ports.csv', header = true);
