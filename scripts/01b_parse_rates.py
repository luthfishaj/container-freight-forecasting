# ---------------------------------------------------------------
# Script: 01b_parse_rates.py
# Purpose: Reads every NCFI data sheet, builds one weekly table of
#          index values r[l, t] per route, and checks it.
# Input:   data/raw/ncfi/*.pdf  (written by 01_ingest_rates.py)
# Output:  data/interim/ncfi_weekly.csv
#              week_date, route, value, value_source
#          data/interim/ncfi_check_mismatches.csv
#              weeks where two sheets disagree (if any)
#
# Each sheet gives two values per route: the "current" week t and
# the "previous" week t-1. The weekly table uses, for every week:
#   - the current value of the sheet for that week, which is the
#     value first published in week t (small later revisions are
#     ignored, so no later information is used);
#   - but if the next sheet prints a value more than 5% different,
#     the current value is a publishing error (two sheets in 2024
#     printed every route at about twice its level) and the corrected
#     value from the next sheet is used instead;
#   - otherwise, the previous value printed on the next week's sheet,
#     which fills single weeks whose sheet is missing.
# The value_source column records which case applies to each week.
# ---------------------------------------------------------------

import pandas as pd
import pandera.pandas as pa

from hff.config import PROJECT_ROOT, load_config
from hff.ncfi import extract_sheet_text, parse_sheet_text

# Two values are treated as equal when they differ by less than this
# many index points (values are printed with two decimals).
MATCH_TOLERANCE = 0.011
# Allowed gap, in percentage points, between the printed weekly growth
# and the growth computed from the two printed values.
GROWTH_TOLERANCE = 0.02
# A current value that differs from the next sheet's value by more
# than this share is treated as a publishing error and corrected.
ERROR_THRESHOLD = 0.05

SOURCE_CURRENT = "current column"
SOURCE_CORRECTED = "corrected by next sheet"
SOURCE_FILLED = "previous column of next sheet"

config = load_config()
sheet_folder = PROJECT_ROOT / config["paths"]["raw_data"] / "ncfi"
interim_folder = PROJECT_ROOT / config["paths"]["interim_data"]
interim_folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# 1. Read every sheet
# ---------------------------------------------------------------
sheet_paths = sorted(sheet_folder.glob("*.pdf"))
print(f"Reading {len(sheet_paths)} data sheets...")
sheet_rows = []
unreadable_sheets = []
for sheet_path in sheet_paths:
    try:
        rows = parse_sheet_text(extract_sheet_text(sheet_path))
    except Exception as error:
        unreadable_sheets.append(sheet_path.name)
        print(f"  could not read {sheet_path.name}: {error}")
        continue
    for row in rows:
        row["sheet_file"] = sheet_path.name
        sheet_rows.append(row)
sheets = pd.DataFrame(sheet_rows)
print(f"  {len(sheets)} route values read, {len(unreadable_sheets)} sheets unreadable")

# ---------------------------------------------------------------
# 2. Check the printed weekly growth against the printed values
# ---------------------------------------------------------------
computed_growth = 100 * (sheets["current_value"] / sheets["previous_value"] - 1)
growth_gap = (computed_growth - sheets["weekly_growth_pct"]).abs()
growth_problems = sheets[growth_gap > GROWTH_TOLERANCE]
print(f"  growth check: {len(growth_problems)} rows outside tolerance")

# ---------------------------------------------------------------
# 3. Current values (first publication) and previous values
# ---------------------------------------------------------------
current = sheets[["current_date", "route", "current_value", "sheet_file"]].copy()
current.columns = ["week_date", "route", "value", "sheet_file"]
previous = sheets[["previous_date", "route", "previous_value", "sheet_file"]].copy()
previous.columns = ["week_date", "route", "value", "sheet_file"]

# An index value of 0.00 is a publishing error (one 2020 sheet prints
# it for a route not used by the project); the week is left missing.
zero_current = current[current["value"] <= 0]
zero_previous = previous[previous["value"] <= 0]
for zero_row in pd.concat([zero_current, zero_previous]).itertuples():
    print(f"  dropped value 0.00: {zero_row.route}, week {zero_row.week_date}")
current = current[current["value"] > 0]
previous = previous[previous["value"] > 0]

# The same week can appear in two sheet files (a report posted twice).
# Copies with the same value are merged; different values are reported.
current_copies = current.groupby(["week_date", "route"])["value"].nunique()
conflicting_copies = current_copies[current_copies > 1]
print(f"  duplicate check: {len(conflicting_copies)} weeks with conflicting copies")
current = current.drop_duplicates(subset=["week_date", "route"], keep="first")
previous = previous.drop_duplicates(subset=["week_date", "route"], keep="first")

# ---------------------------------------------------------------
# 4. Compare overlapping values and build the weekly table
# ---------------------------------------------------------------
both = current.merge(
    previous, on=["week_date", "route"], how="inner", suffixes=("", "_next_sheet")
)
difference = (both["value"] - both["value_next_sheet"]).abs()
mismatches = both[difference > MATCH_TOLERANCE]
print(
    f"  overlap check: {len(both)} values printed twice, "
    f"{len(mismatches)} differ between sheets"
)
mismatches.to_csv(interim_folder / "ncfi_check_mismatches.csv", index=False)

# Large differences are publishing errors: the value from the next
# sheet replaces the current value for that week and route.
relative_difference = difference / both["value_next_sheet"]
errors = both[relative_difference > ERROR_THRESHOLD]
print(f"  publishing errors corrected: {len(errors)} values")
for week_date in sorted(errors["week_date"].unique()):
    error_routes = errors[errors["week_date"] == week_date]
    print(f"    week {week_date}: {len(error_routes)} routes")

current["value_source"] = SOURCE_CURRENT
previous["value_source"] = SOURCE_FILLED
for error_row in errors.itertuples():
    is_error = (current["week_date"] == error_row.week_date) & (
        current["route"] == error_row.route
    )
    current.loc[is_error, "value"] = error_row.value_next_sheet
    current.loc[is_error, "value_source"] = SOURCE_CORRECTED

# Weeks already covered by a current value are removed from the
# previous values before the two are combined.
covered_weeks = current[["week_date", "route"]]
previous_only = previous.merge(
    covered_weeks, on=["week_date", "route"], how="left", indicator=True
)
previous_only = previous_only[previous_only["_merge"] == "left_only"]
previous_only = previous_only.drop(columns="_merge")

weekly = pd.concat([current, previous_only], ignore_index=True)
weekly = weekly[["week_date", "route", "value", "value_source"]]
weekly = weekly.sort_values(["route", "week_date"]).reset_index(drop=True)

# ---------------------------------------------------------------
# 5. Validate and save
# ---------------------------------------------------------------
weekly_schema = pa.DataFrameSchema(
    {
        "week_date": pa.Column(str, pa.Check.str_matches(r"^\d{4}-\d{2}-\d{2}$")),
        "route": pa.Column(str),
        "value": pa.Column(float, pa.Check.gt(0)),
        "value_source": pa.Column(
            str, pa.Check.isin([SOURCE_CURRENT, SOURCE_CORRECTED, SOURCE_FILLED])
        ),
    },
    unique=["week_date", "route"],
)
weekly = weekly_schema.validate(weekly)
weekly.to_csv(interim_folder / "ncfi_weekly.csv", index=False)

# ---------------------------------------------------------------
# 6. Coverage of the project lanes
# ---------------------------------------------------------------
print("Coverage of the project lanes:")
for lane in config["lanes"]:
    lane_weeks = weekly[weekly["route"] == lane["ncfi_route"]]
    if len(lane_weeks) == 0:
        print(f"  {lane['id']}: NO DATA for route '{lane['ncfi_route']}'")
        continue
    lane_dates = pd.to_datetime(lane_weeks["week_date"])
    gaps_in_days = lane_dates.diff().dt.days
    missing_weeks = int(((gaps_in_days // 7) - 1).clip(lower=0).sum())
    filled_weeks = int((lane_weeks["value_source"] != "current column").sum())
    print(
        f"  {lane['id']}: {len(lane_weeks)} weeks, "
        f"{lane_dates.min().date()} to {lane_dates.max().date()}, "
        f"{filled_weeks} taken from the next sheet, {missing_weeks} still missing"
    )
print(f"Saved {len(weekly)} rows to data/interim/ncfi_weekly.csv")
