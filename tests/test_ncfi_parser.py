# ---------------------------------------------------------------
# Test: test_ncfi_parser.py
# Purpose: Checks that the NCFI data sheet parser reads dates, route
#          names and values correctly.
#
# The sheet text below is SYNTHETIC: it copies the layout of a real
# NCFI data sheet, but every number is invented for the test.
# ---------------------------------------------------------------

import math

import pytest

from hff.ncfi import parse_sheet_text

SYNTHETIC_SHEET_TEXT = """Previous Index Current Index
(2020-01-03) (2020-01-10)
Composite Index 1000.00 1100.00 10.00%
Europe 500.00 450.00 -10.00%
W. Mediterranean 200.50 210.25 4.86%
Middle East 800.00 800.00 0.00%
India& Pakistan 300.00 330.00 10.00%
Australia& New
Zealand 400.00 420.00 5.00%
Route WeeklyGrowth(%)
"""


def test_parser_reads_both_dates():
    rows = parse_sheet_text(SYNTHETIC_SHEET_TEXT)
    for row in rows:
        assert row["previous_date"] == "2020-01-03"
        assert row["current_date"] == "2020-01-10"


def test_parser_finds_every_route():
    rows = parse_sheet_text(SYNTHETIC_SHEET_TEXT)
    route_names = []
    for row in rows:
        route_names.append(row["route"])
    assert route_names == [
        "Composite Index",
        "Europe",
        "W. Mediterranean",
        "Middle East",
        "India& Pakistan",
        "Australia& New Zealand",
    ]


def test_parser_reads_values_and_negative_growth():
    rows = parse_sheet_text(SYNTHETIC_SHEET_TEXT)
    europe = rows[1]
    assert europe["previous_value"] == 500.00
    assert europe["current_value"] == 450.00
    assert europe["weekly_growth_pct"] == -10.00


# Newer sheets put the header on one line, add an arrow symbol line
# after each route and can place the values on their own line.
SYNTHETIC_NEW_LAYOUT_TEXT = (
    "Route Previous Index Current Index Weekly Growth(%)(2026-01-02) (2026-01-09)\n"
    "Composite Index 1000.00 1100.00 10.00%\n"
    " #\n"
    "Middle East 800.00 760.00 -5.00%\n"
    " #\n"
    "Australia& New \n"
    "Zealand\n"
    "400.00 420.00 5.00%\n"
    " #\n"
)


def test_parser_reads_new_layout():
    rows = parse_sheet_text(SYNTHETIC_NEW_LAYOUT_TEXT)
    route_names = []
    for row in rows:
        route_names.append(row["route"])
    assert route_names == ["Composite Index", "Middle East", "Australia& New Zealand"]
    assert rows[0]["previous_date"] == "2026-01-02"
    assert rows[0]["current_date"] == "2026-01-09"
    assert rows[1]["current_value"] == 760.00
    assert rows[2]["weekly_growth_pct"] == 5.00


# Some 2020 sheets end route lines with "##" and print "#DIV/0!" as
# the growth when the previous value is 0.00.
SYNTHETIC_MARKED_TEXT = """Previous Index Current Index
(2020-01-03) (2020-01-10)
Middle East 800.00 760.00 -5.00% ##
Australia& New
Zealand 0.00 420.00 #DIV/0! ##
W. Japan 200.00 210.00 5.00%
Route WeeklyGrowth(%)
"""


def test_parser_reads_marked_lines_and_division_errors():
    rows = parse_sheet_text(SYNTHETIC_MARKED_TEXT)
    route_names = []
    for row in rows:
        route_names.append(row["route"])
    assert route_names == ["Middle East", "Australia& New Zealand", "W. Japan"]
    assert rows[0]["weekly_growth_pct"] == -5.00
    assert rows[1]["previous_value"] == 0.00
    assert math.isnan(rows[1]["weekly_growth_pct"])
    assert rows[2]["current_value"] == 210.00


def test_parser_rejects_text_without_two_dates():
    with pytest.raises(ValueError):
        parse_sheet_text("Composite Index 1000.00 1100.00 10.00%")
