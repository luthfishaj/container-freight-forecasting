# ---------------------------------------------------------------
# Script: app/dashboard.py
# Purpose: Entry point of the Streamlit dashboard ("make dashboard",
#          then http://localhost:8501). Each page is a file in
#          app/views/; shared loaders and chart helpers are in
#          app/common.py.
# Input:   the public result files in reports/, written by "make all"
#          and "make weekly" (the dashboard never reads data/)
#
# Deployment: on Streamlit Community Cloud the project package (hff) is
# not installed, only app/requirements.txt. The src/ folder is
# therefore added to Python's search path here, before any page
# imports hff. Locally this changes nothing: the same code is found.
# ---------------------------------------------------------------

import sys
from pathlib import Path

import streamlit as st

SOURCE_FOLDER = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_FOLDER) not in sys.path:
    sys.path.insert(0, str(SOURCE_FOLDER))

st.set_page_config(
    page_title="Container freight rate forecasting",
    layout="wide",
)

pages = [
    st.Page("views/overview.py", title="Overview", default=True),
    st.Page("views/forecast.py", title="Latest forecast"),
    st.Page("views/accuracy.py", title="Model accuracy"),
    st.Page("views/track_record.py", title="Forecasts vs actual"),
    st.Page("views/stress_test.py", title="Hormuz stress test"),
    st.Page("views/scenarios.py", title="Scenario explorer"),
    st.Page("views/pricing.py", title="Pricing illustration"),
    st.Page("views/monitoring.py", title="Monitoring"),
]
st.navigation(pages).run()
