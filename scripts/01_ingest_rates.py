# ---------------------------------------------------------------
# Script: 01_ingest_rates.py
# Purpose: Downloads the weekly data sheets of the Ningbo
#          Containerized Freight Index (NCFI), one PDF per week, and
#          records each download in the raw-data manifest.
# Input:   report pages of Hellenic Shipping News, which republishes
#          the weekly NCFI report of the Ningbo Shipping Exchange
# Output:  data/raw/ncfi/*.pdf             one data sheet per week
#          data/raw/ncfi/sheet_urls.csv     where each sheet was found
#          data/raw/ncfi/failed_downloads.csv  sheets that could not
#                                           be downloaded (if any)
#          data/raw/manifest.csv            one row per downloaded file
#
# How the sheets are found:
#   1. Reports from mid-2021 onward are listed through the news
#      site's public WordPress API.
#   2. Older reports (2019 to 2021) were removed from the site but
#      are kept by the Internet Archive. Each archived report page
#      gives the address of its data sheet; the sheets themselves are
#      still served by the news site.
#
# The news site's robots.txt asks for 30 seconds between requests,
# so the first full run takes several hours. Later runs skip the
# sheets already downloaded and only fetch the new weeks.
# ---------------------------------------------------------------

import csv
import re
import time

import requests

from hff.config import PROJECT_ROOT, load_config
from hff.io import USER_AGENT, append_to_manifest, download_file

# ---------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------
NEWS_SITE = "https://www.hellenicshippingnews.com"
WORDPRESS_API = NEWS_SITE + "/wp-json/wp/v2/posts"
SEARCH_PHRASE = "Ningbo Containerized Freight Index Report"

ARCHIVE_INDEX_API = "https://web.archive.org/cdx/search/cdx"
ARCHIVE_PAGE_PREFIX = "https://web.archive.org/web/"
ARCHIVED_REPORT_PAGES = (
    "hellenicshippingnews.com/ningbo-containerized-freight-index-report*"
)
ARCHIVE_FIRST_YEAR = "2019"
ARCHIVE_LAST_YEAR = "2021"

NEWS_SITE_DELAY_SECONDS = 30
ARCHIVE_DELAY_SECONDS = 5

SHEET_LIST_COLUMNS = ["report_url", "sheet_url", "found_via"]

# A data sheet link, for example
# .../wp-content/uploads/2019/06/NCFI-Weekly-Index-Data-2019-5.25-5.31.pdf
SHEET_LINK_PATTERN = re.compile(
    r"wp-content/uploads/[^\"'\s<>]*NCFI-Weekly-Index-Data[^\"'\s<>]*\.pdf",
    re.IGNORECASE,
)

config = load_config()
sheet_folder = PROJECT_ROOT / config["paths"]["raw_data"] / "ncfi"
manifest_path = PROJECT_ROOT / config["paths"]["manifest"]
archived_list_path = sheet_folder / "archived_sheet_urls.csv"
sheet_list_path = sheet_folder / "sheet_urls.csv"
failed_list_path = sheet_folder / "failed_downloads.csv"
sheet_folder.mkdir(parents=True, exist_ok=True)


def fetch(url: str, params: dict | None = None) -> requests.Response:
    """Sends one GET request with the project's user agent."""
    response = requests.get(
        url, params=params, headers={"User-Agent": USER_AGENT}, timeout=60
    )
    response.raise_for_status()
    return response


def find_sheet_urls(page_html: str) -> list[str]:
    """Returns the data sheet addresses on the news site found in a page."""
    sheet_urls = []
    for match in SHEET_LINK_PATTERN.findall(page_html):
        # Archived pages may rewrite links; only the path on the news
        # site is kept, so every address points to the original file.
        sheet_url = NEWS_SITE + "/" + match
        if sheet_url not in sheet_urls:
            sheet_urls.append(sheet_url)
    return sheet_urls


def write_rows(path, rows: list[dict], columns: list[str]) -> None:
    """Writes a list of dictionaries to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------
# 2. Sheets listed on the news site (mid-2021 onward)
# ---------------------------------------------------------------
print("Step 1: listing NCFI reports through the news site's API...")
live_sheets = []
page_number = 1
total_pages = 1
while page_number <= total_pages:
    time.sleep(NEWS_SITE_DELAY_SECONDS)
    response = fetch(
        WORDPRESS_API,
        params={
            "search": SEARCH_PHRASE,
            "per_page": 100,
            "page": page_number,
            "_fields": "date,link,content",
        },
    )
    total_pages = int(response.headers.get("X-WP-TotalPages", 1))
    for post in response.json():
        for sheet_url in find_sheet_urls(post["content"]["rendered"]):
            live_sheets.append(
                {
                    "report_url": post["link"],
                    "sheet_url": sheet_url,
                    "found_via": "news site",
                }
            )
    print(f"  page {page_number}/{total_pages}: {len(live_sheets)} sheets so far")
    page_number += 1

# ---------------------------------------------------------------
# 3. Sheets found through archived report pages (2019 to 2021)
# ---------------------------------------------------------------
# The archived pages never change, so they are read only once and
# the result is kept in archived_sheet_urls.csv for later runs.
archived_sheets = []
if archived_list_path.exists():
    print("Step 2: reading the saved list of archived sheets...")
    with open(archived_list_path, encoding="utf-8") as archived_file:
        archived_sheets = list(csv.DictReader(archived_file))
else:
    print("Step 2: looking up archived report pages in the Internet Archive...")
    index_rows = fetch(
        ARCHIVE_INDEX_API,
        params={
            "url": ARCHIVED_REPORT_PAGES,
            "from": ARCHIVE_FIRST_YEAR,
            "to": ARCHIVE_LAST_YEAR,
            "output": "json",
            "fl": "timestamp,original",
            "filter": "statuscode:200",
            "collapse": "urlkey",
        },
    ).json()

    # The first row holds column names. Copies of the same page in
    # other formats (/amp/, /feed/) are skipped.
    report_pages = []
    for timestamp, original_url in index_rows[1:]:
        if "/amp" in original_url or "/feed" in original_url:
            continue
        report_pages.append((timestamp, original_url))
    print(f"  {len(report_pages)} archived report pages to read")

    for page_index, (timestamp, original_url) in enumerate(report_pages, start=1):
        time.sleep(ARCHIVE_DELAY_SECONDS)
        # The "id_" flag returns the page as originally published,
        # without the archive's navigation bar.
        archived_page_url = ARCHIVE_PAGE_PREFIX + timestamp + "id_/" + original_url
        try:
            page_html = fetch(archived_page_url).text
        except requests.RequestException as error:
            print(f"  could not read {original_url}: {error}")
            continue
        for sheet_url in find_sheet_urls(page_html):
            archived_sheets.append(
                {
                    "report_url": original_url,
                    "sheet_url": sheet_url,
                    "found_via": "internet archive",
                }
            )
        if page_index % 20 == 0:
            print(f"  {page_index}/{len(report_pages)} pages read")

    write_rows(archived_list_path, archived_sheets, SHEET_LIST_COLUMNS)

# ---------------------------------------------------------------
# 4. Combined list of sheets (one entry per file name)
# ---------------------------------------------------------------
all_sheets = []
seen_file_names = set()
for sheet in live_sheets + archived_sheets:
    file_name = sheet["sheet_url"].rsplit("/", 1)[-1]
    if file_name in seen_file_names:
        continue
    seen_file_names.add(file_name)
    all_sheets.append(sheet)

write_rows(sheet_list_path, all_sheets, SHEET_LIST_COLUMNS)
print(
    f"Found {len(all_sheets)} distinct data sheets "
    f"({len(live_sheets)} links on the news site, "
    f"{len(archived_sheets)} via the archive)"
)

# ---------------------------------------------------------------
# 5. Download the sheets not yet on disk
# ---------------------------------------------------------------
print("Step 3: downloading new data sheets...")
downloaded_count = 0
skipped_count = 0
failed_sheets = []
for sheet_index, sheet in enumerate(all_sheets, start=1):
    file_name = sheet["sheet_url"].rsplit("/", 1)[-1]
    destination = sheet_folder / file_name
    if destination.exists():
        skipped_count += 1
        continue

    # The news site is tried first. If the file has been removed
    # there, the Internet Archive's copy is used instead.
    archive_copy_url = ARCHIVE_PAGE_PREFIX + "2id_/" + sheet["sheet_url"]
    source_urls = [sheet["sheet_url"], archive_copy_url]
    saved_from = None
    for source_url in source_urls:
        if source_url.startswith(NEWS_SITE):
            time.sleep(NEWS_SITE_DELAY_SECONDS)
        else:
            time.sleep(ARCHIVE_DELAY_SECONDS)
        try:
            download_file(source_url, destination)
            saved_from = source_url
            break
        except requests.RequestException as error:
            print(f"  {file_name}: not available from {source_url} ({error})")

    if saved_from is None:
        failed_sheets.append(sheet)
        continue

    append_to_manifest(manifest_path, destination, saved_from)
    downloaded_count += 1
    print(f"  [{sheet_index}/{len(all_sheets)}] saved {file_name}")

# Rewritten on every run, so the file never lists failures that a
# later run has already fetched (no rows means nothing failed).
write_rows(failed_list_path, failed_sheets, SHEET_LIST_COLUMNS)

print(
    f"Done: {downloaded_count} downloaded, {skipped_count} already on disk, "
    f"{len(failed_sheets)} failed"
)
