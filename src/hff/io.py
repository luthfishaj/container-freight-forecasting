# ---------------------------------------------------------------
# Module: hff/io.py
# Purpose: Downloads files and records each download in the raw-data
#          manifest, so that every raw file can be traced back to its
#          source URL, retrieval time and content checksum.
# Output:  files in data/raw/ and rows in data/raw/manifest.csv
# ---------------------------------------------------------------

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import requests

from hff.config import PROJECT_ROOT

# Requests identify the project so that site owners can see who is
# downloading and why.
USER_AGENT = (
    "container-freight-forecasting research script "
    "(github.com/luthfishaj/container-freight-forecasting)"
)
TIMEOUT_SECONDS = 60

MANIFEST_COLUMNS = ["file", "source_url", "retrieved_at_utc", "sha256", "size_bytes"]


def download_file(url: str, destination: Path) -> None:
    """Downloads one URL to a local file; raises an error if the request fails."""
    response = requests.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS
    )
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)


def sha256_of_file(file_path: Path) -> str:
    """Returns the SHA-256 checksum of a file as a hexadecimal string."""
    file_hash = hashlib.sha256()
    file_hash.update(file_path.read_bytes())
    return file_hash.hexdigest()


def append_to_manifest(manifest_path: Path, file_path: Path, source_url: str) -> None:
    """Adds one row describing a downloaded file to the manifest CSV."""
    is_new_manifest = not manifest_path.exists()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Paths are stored relative to the project root so that the
    # manifest stays valid when the project folder is moved.
    relative_path = file_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    row = {
        "file": relative_path,
        "source_url": source_url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": sha256_of_file(file_path),
        "size_bytes": file_path.stat().st_size,
    }

    with open(manifest_path, "a", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
        if is_new_manifest:
            writer.writeheader()
        writer.writerow(row)


def load_panel(config: dict) -> pd.DataFrame:
    """Reads the weekly panel built by 03_build_panel.py."""
    panel_path = (PROJECT_ROOT / config["paths"]["panel"]).as_posix()
    panel = duckdb.sql(f"SELECT * FROM '{panel_path}'").df()
    panel["week_date"] = pd.to_datetime(panel["week_date"])
    return panel.sort_values(["lane_id", "week_date"]).reset_index(drop=True)
