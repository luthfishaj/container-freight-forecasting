# ---------------------------------------------------------------
# Module: hff/ncfi.py
# Purpose: Reads one weekly data sheet of the Ningbo Containerized
#          Freight Index (NCFI) and returns its values as rows.
#
# Each data sheet is a one-page PDF published by the Ningbo Shipping
# Exchange. It lists 21 routes plus a composite index, with two
# columns of values in index points:
#   Previous Index  r[l, t-1]  value of the previous week
#   Current Index   r[l, t]    value of the week the sheet reports
#   Weekly Growth   100 * (r[l, t] / r[l, t-1] - 1), in percent
# The column dates appear in the header as (YYYY-MM-DD).
#
# Two layouts exist. In older sheets each route is one text line:
#   "Middle East 806.12 920.64 14.21%"
# In newer sheets each route line is followed by a line holding an
# arrow symbol ("#"), and a long route name can be split over several
# lines with its values on a line of their own:
#   "Australia& New" / "Zealand" / "4064.13 3851.12 -5.24%"
# The parser handles both by collecting name fragments until a line
# with the three values appears.
#
# Two spreadsheet artefacts also appear in some 2020 sheets: route
# lines ending with "##" marks, and a growth cell printed as
# "#DIV/0!" when the previous value is 0.00 (a publishing error).
# The marks are ignored and the growth is then returned as missing.
# ---------------------------------------------------------------

import re
from pathlib import Path

from pypdf import PdfReader

# The three values at the end of a route line: previous value,
# current value and weekly growth in percent (or "#DIV/0!"),
# optionally followed by "#" marks.
VALUES_PATTERN = re.compile(
    r"(?P<previous>\d+\.\d+)\s+(?P<current>\d+\.\d+)\s+"
    r"(?:(?P<growth>-?\d+\.\d+)%|#DIV/0!)[\s#]*$"
)
DATE_PATTERN = re.compile(r"\((\d{4}-\d{2}-\d{2})\)")
# Words that only appear in the column headers.
HEADER_WORDS = ["Index", "Growth", "Route"]


def extract_sheet_text(pdf_path: Path) -> str:
    """Returns the text of the first page of an NCFI data sheet."""
    reader = PdfReader(pdf_path)
    return reader.pages[0].extract_text()


def parse_sheet_text(sheet_text: str) -> list[dict]:
    """Parses the text of one data sheet into one dictionary per route."""
    # The two dates in the header give the previous and current week.
    dates = DATE_PATTERN.findall(sheet_text)
    if len(dates) != 2:
        raise ValueError(f"Expected 2 dates in the sheet header, found {len(dates)}")
    previous_date = dates[0]
    current_date = dates[1]

    rows = []
    name_fragments = []
    for line in sheet_text.splitlines():
        line = line.strip()
        values = VALUES_PATTERN.search(line)

        if values is None:
            # A line without values is either noise (arrow symbols,
            # headers, dates) or part of a route name split over lines.
            has_letters = re.search(r"[A-Za-z]", line) is not None
            is_header = False
            for word in HEADER_WORDS:
                if word in line:
                    is_header = True
            if has_letters and not is_header:
                name_fragments.append(line)
            continue

        # The route name is the text before the values, preceded by any
        # fragments collected from earlier lines.
        name_fragments.append(line[: values.start()].strip())
        route_name = " ".join(name_fragments)
        route_name = re.sub(r"\s+", " ", route_name).strip()
        name_fragments = []

        if values.group("growth") is None:
            weekly_growth_pct = float("nan")
        else:
            weekly_growth_pct = float(values.group("growth"))

        rows.append(
            {
                "route": route_name,
                "previous_date": previous_date,
                "current_date": current_date,
                "previous_value": float(values.group("previous")),
                "current_value": float(values.group("current")),
                "weekly_growth_pct": weekly_growth_pct,
            }
        )

    return rows
