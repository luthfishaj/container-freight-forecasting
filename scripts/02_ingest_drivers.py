# ---------------------------------------------------------------
# Script: 02_ingest_drivers.py
# Purpose: Downloads the three driver series x[k, t] and records each
#          download in the raw-data manifest:
#            - Brent crude oil price (fuel cost channel)
#            - daily Geopolitical Risk Index (risk channel)
#            - daily port calls at Ningbo, Jebel Ali and Dammam
#              (Gulf disruption and port activity channel)
#          All three are daily, so no monthly publication lag has to
#          be handled when the weekly panel is built.
# Input:   FRED, Caldara-Iacoviello GPR site, IMF PortWatch
# Output:  data/raw/drivers/fred_DCOILBRENTEU.csv
#          data/raw/drivers/gpr_daily.xls
#          data/raw/drivers/portwatch_ports.csv
#          data/raw/manifest.csv
#
# Every file is downloaded again on each run, because the sources
# revise past values. The manifest keeps the retrieval time and
# checksum of every download, so each run can be traced.
# ---------------------------------------------------------------

import csv
import time

import requests

from hff.config import PROJECT_ROOT, load_config
from hff.io import USER_AGENT, append_to_manifest, download_file

# ---------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------
# Brent crude oil spot price, USD per barrel, from the public FRED
# CSV export (no API key needed).
BRENT_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"

# Daily Geopolitical Risk Index (Caldara and Iacoviello).
GPR_DAILY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"

# IMF PortWatch daily port data, queried through its public ArcGIS
# service. Ningbo is the origin port of the NCFI; Jebel Ali and Dammam
# are the two destination ports of its Middle East route.
PORTWATCH_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "Daily_Ports_Data/FeatureServer/0/query"
)
PORTWATCH_PORTS = {"port824": "Ningbo", "port744": "Jebel Ali", "port275": "Dammam"}
PORTWATCH_FIELDS = "date,portid,portname,portcalls_container,portcalls,import,export"
PORTWATCH_PAGE_SIZE = 1000  # maximum number of rows the service returns
PORTWATCH_ATTEMPTS = 5
PORTWATCH_RETRY_SECONDS = 30

config = load_config()
driver_folder = PROJECT_ROOT / config["paths"]["raw_data"] / "drivers"
manifest_path = PROJECT_ROOT / config["paths"]["manifest"]
driver_folder.mkdir(parents=True, exist_ok=True)


def download_and_record(url: str, file_name: str) -> bool:
    """Downloads one file into the driver folder; returns False on failure."""
    destination = driver_folder / file_name
    try:
        download_file(url, destination)
    except requests.RequestException as error:
        print(f"  FAILED {file_name}: {error}")
        return False
    append_to_manifest(manifest_path, destination, url)
    print(f"  saved {file_name}")
    return True


def query_port_page(port_id: str, offset: int) -> dict:
    """Requests one page of daily rows for one port, with retries."""
    # The service fails intermittently with generic errors, so each
    # page is tried several times before the download is abandoned.
    last_error = ""
    for attempt in range(1, PORTWATCH_ATTEMPTS + 1):
        try:
            response = requests.get(
                PORTWATCH_URL,
                params={
                    "where": f"portid = '{port_id}'",
                    "outFields": PORTWATCH_FIELDS,
                    "orderByFields": "ObjectId ASC",
                    "resultOffset": offset,
                    "resultRecordCount": PORTWATCH_PAGE_SIZE,
                    "returnGeometry": "false",
                    "f": "json",
                },
                headers={"User-Agent": USER_AGENT},
                timeout=120,
            )
            response.raise_for_status()
            answer = response.json()
            if "error" not in answer:
                return answer
            last_error = str(answer["error"])
        except requests.RequestException as error:
            last_error = str(error)
        print(f"    attempt {attempt} failed for {port_id}: {last_error}")
        time.sleep(PORTWATCH_RETRY_SECONDS)
    raise RuntimeError(f"PortWatch gave no answer for {port_id}: {last_error}")


def download_port_calls(file_name: str) -> bool:
    """Downloads every daily row for the selected ports into one CSV file."""
    rows = []
    try:
        for port_id in PORTWATCH_PORTS:
            offset = 0
            while True:
                # The service returns a limited number of rows per
                # request, so results are read page by page.
                answer = query_port_page(port_id, offset)
                for feature in answer["features"]:
                    row = feature["attributes"]
                    # Dates come as text such as "2019-01-01".
                    row["date"] = str(row["date"])[:10]
                    rows.append(row)

                # The next page starts after the rows actually returned,
                # which can be fewer than requested.
                more_rows_available = answer.get("exceededTransferLimit", False)
                if not more_rows_available or len(answer["features"]) == 0:
                    break
                offset += len(answer["features"])
    except RuntimeError as error:
        print(f"  FAILED {file_name}: {error}")
        return False

    destination = driver_folder / file_name
    with open(destination, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=PORTWATCH_FIELDS.split(","))
        writer.writeheader()
        writer.writerows(rows)
    source = PORTWATCH_URL + " (portid IN " + ", ".join(PORTWATCH_PORTS) + ")"
    append_to_manifest(manifest_path, destination, source)
    print(f"  saved {file_name} ({len(rows)} rows)")
    return True


# ---------------------------------------------------------------
# 2. Downloads
# ---------------------------------------------------------------
failures = []

print("Brent crude oil price (FRED)...")
if not download_and_record(BRENT_URL, "fred_DCOILBRENTEU.csv"):
    failures.append("Brent")

print("Daily Geopolitical Risk Index...")
if not download_and_record(GPR_DAILY_URL, "gpr_daily.xls"):
    failures.append("GPR")

print("PortWatch port calls...")
if not download_port_calls("portwatch_ports.csv"):
    failures.append("PortWatch ports")

if failures:
    print("Finished with failures: " + ", ".join(failures))
else:
    print("Finished: all driver files downloaded")
