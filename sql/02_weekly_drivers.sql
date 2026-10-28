-- ---------------------------------------------------------------
-- File: sql/02_weekly_drivers.sql
-- Purpose: Aggregates the daily drivers to NCFI weeks. An NCFI week
--          runs from Saturday to Friday and is dated by its Friday,
--          the day the index is published.
-- Input:   staging views from sql/01_staging.sql
-- Output:  table weekly_drivers, one row per Friday, with the weekly
--          value of each driver x[k, t] BEFORE any publication lag
--          (the lag is applied in 03_build_panel.py)
-- ---------------------------------------------------------------

-- In DuckDB, dayofweek() returns 0 for Sunday up to 6 for Saturday,
-- so Friday is 5. Adding (5 - day + 7) % 7 days moves every date to
-- the Friday that closes its week: a Saturday moves forward 6 days,
-- a Friday stays where it is.
CREATE OR REPLACE MACRO week_ending_friday(day) AS
    day + CAST((5 - dayofweek(day) + 7) % 7 AS INTEGER);

-- Brent: average of the business-day prices of the week.
CREATE OR REPLACE TABLE weekly_brent AS
SELECT
    week_ending_friday(day) AS week_date,
    AVG(brent_usd) AS brent_usd,
    COUNT(*) AS brent_days
FROM stg_brent
GROUP BY week_date;

-- Geopolitical risk: average of the daily index over the week.
CREATE OR REPLACE TABLE weekly_gpr AS
SELECT
    week_ending_friday(day) AS week_date,
    AVG(gprd) AS gprd,
    COUNT(*) AS gpr_days
FROM stg_gpr
GROUP BY week_date;

-- Port calls: total container port calls over the week, for the two
-- Gulf destination ports together (Jebel Ali port744, Dammam port275)
-- and for the origin port (Ningbo port824). A week counts only when
-- every port has all 7 days (one row per port and day), so that a
-- partial week is never read as a drop in activity.
CREATE OR REPLACE TABLE weekly_port_calls AS
SELECT
    week_ending_friday(day) AS week_date,
    CASE
        WHEN COUNT(CASE WHEN portid IN ('port744', 'port275') THEN 1 END) = 14
        THEN SUM(CASE WHEN portid IN ('port744', 'port275') THEN container_calls END)
    END AS gulf_container_calls,
    CASE
        WHEN COUNT(CASE WHEN portid = 'port824' THEN 1 END) = 7
        THEN SUM(CASE WHEN portid = 'port824' THEN container_calls END)
    END AS ningbo_container_calls
FROM stg_port_calls
GROUP BY week_date;

-- All drivers side by side, one row per Friday.
CREATE OR REPLACE TABLE weekly_drivers AS
SELECT
    COALESCE(b.week_date, g.week_date, p.week_date) AS week_date,
    b.brent_usd,
    g.gprd,
    p.gulf_container_calls,
    p.ningbo_container_calls
FROM weekly_brent AS b
FULL OUTER JOIN weekly_gpr AS g
    ON b.week_date = g.week_date
FULL OUTER JOIN weekly_port_calls AS p
    ON COALESCE(b.week_date, g.week_date) = p.week_date
ORDER BY week_date;
