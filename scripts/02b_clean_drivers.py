# ---------------------------------------------------------------
# Script: 02b_clean_drivers.py
# Purpose: Converts the three raw driver files into tidy daily tables,
#          checks them, and saves them for the weekly panel.
# Input:   data/raw/drivers/*  (written by 02_ingest_drivers.py)
# Output:  data/interim/brent.csv            date, brent_usd
#          data/interim/gpr_daily.csv        date, gprd, gprd_act,
#                                            gprd_threat
#          data/interim/portwatch_ports.csv  date, portid, portname,
#                                            port calls and volumes
#
# Dates are stored as text in YYYY-MM-DD format. Days without a
# published value (weekends, holidays) are simply absent; the weekly
# aggregation in Phase 3 handles them.
# ---------------------------------------------------------------

import pandas as pd
import pandera.pandas as pa

from hff.config import PROJECT_ROOT, load_config

config = load_config()
raw_folder = PROJECT_ROOT / config["paths"]["raw_data"] / "drivers"
interim_folder = PROJECT_ROOT / config["paths"]["interim_data"]
interim_folder.mkdir(parents=True, exist_ok=True)

# A date stored as text must look like 2024-01-31.
DATE_TEXT = pa.Check.str_matches(r"^\d{4}-\d{2}-\d{2}$")


def save_table(table: pd.DataFrame, file_name: str) -> None:
    """Saves a cleaned table and prints its size and date range."""
    table.to_csv(interim_folder / file_name, index=False)
    first_date = table["date"].min()
    last_date = table["date"].max()
    print(f"  {file_name}: {len(table)} rows, {first_date} to {last_date}")


# ---------------------------------------------------------------
# 1. Brent crude oil price
# ---------------------------------------------------------------
print("Cleaning Brent...")
raw_brent = pd.read_csv(raw_folder / "fred_DCOILBRENTEU.csv")
brent = pd.DataFrame()
brent["date"] = raw_brent["observation_date"]
# Days without a published price are empty in the file and are
# dropped rather than filled.
brent["brent_usd"] = pd.to_numeric(raw_brent["DCOILBRENTEU"], errors="coerce")
brent = brent.dropna()

brent_schema = pa.DataFrameSchema(
    {
        "date": pa.Column(str, DATE_TEXT, unique=True),
        "brent_usd": pa.Column(float, pa.Check.gt(0)),
    }
)
brent = brent_schema.validate(brent)
save_table(brent, "brent.csv")

# ---------------------------------------------------------------
# 2. Daily Geopolitical Risk Index
# ---------------------------------------------------------------
print("Cleaning GPR...")
raw_gpr = pd.read_excel(raw_folder / "gpr_daily.xls", engine="calamine")
gpr_daily = pd.DataFrame()
gpr_daily["date"] = pd.to_datetime(raw_gpr["date"]).dt.strftime("%Y-%m-%d")
# GPRD is the overall daily index; ACT and THREAT split it into
# newspaper articles about actual events and about threats.
gpr_daily["gprd"] = raw_gpr["GPRD"].astype(float)
gpr_daily["gprd_act"] = raw_gpr["GPRD_ACT"].astype(float)
gpr_daily["gprd_threat"] = raw_gpr["GPRD_THREAT"].astype(float)
gpr_daily = gpr_daily.dropna()

gpr_schema = pa.DataFrameSchema(
    {
        "date": pa.Column(str, DATE_TEXT, unique=True),
        "gprd": pa.Column(float, pa.Check.ge(0)),
        "gprd_act": pa.Column(float, pa.Check.ge(0)),
        "gprd_threat": pa.Column(float, pa.Check.ge(0)),
    }
)
gpr_daily = gpr_schema.validate(gpr_daily)
save_table(gpr_daily, "gpr_daily.csv")

# ---------------------------------------------------------------
# 3. PortWatch port calls
# ---------------------------------------------------------------
print("Cleaning PortWatch port calls...")
ports = pd.read_csv(raw_folder / "portwatch_ports.csv")
ports = ports.sort_values(["portid", "date"]).reset_index(drop=True)
ports_schema = pa.DataFrameSchema(
    {
        "date": pa.Column(str, DATE_TEXT),
        "portid": pa.Column(str),
        "portname": pa.Column(str),
        "portcalls_container": pa.Column(int, pa.Check.ge(0)),
        "portcalls": pa.Column(int, pa.Check.ge(0)),
        # Import and export volumes are estimates in metric tons.
        "import": pa.Column(float, pa.Check.ge(0), coerce=True),
        "export": pa.Column(float, pa.Check.ge(0), coerce=True),
    },
    unique=["date", "portid"],
)
ports = ports_schema.validate(ports)
save_table(ports, "portwatch_ports.csv")

print("Done.")
