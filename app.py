import streamlit as st
import pandas as pd

from tabs import upload_and_overview
from tabs import cleaning_and_prep
from tabs import visualization
from tabs import export_and_report

# Page config
st.set_page_config(
    page_title="Data Wrangler and Visualizer",
    layout="wide",
    page_icon="👋"
)

# Sidebar title
st.sidebar.header("Upload Dataset", help="Upload any dataset of type CSV, JSON or EXCEL")


@st.cache_data
def load_file(uploaded_file):
    """Read an uploaded file into a DataFrame.

    Returns (dataframe, error_message). Exactly one of the two is not None,
    so the caller can show a friendly message instead of crashing.
    """
    # Match the extension case-insensitively so "DATA.CSV" works too.
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file), None
        elif name.endswith(".json"):
            return pd.read_json(uploaded_file), None
        elif name.endswith(".csv"):
            return pd.read_csv(uploaded_file), None
        else:
            return None, "Unsupported file format. Please upload a CSV, XLSX or JSON file."
    except Exception as e:
        return None, f"Could not read this file. It may be corrupted or in an unexpected format.\n\nDetails: {e}"


# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload your file",
    type=["csv", "xlsx", "json"],
    label_visibility="collapsed"
)

# Load file (only when a genuinely new file is uploaded, so cleaning progress
# is preserved across the reruns that happen on every interaction)
if uploaded_file is not None:
    if "current_file" not in st.session_state or st.session_state["current_file"] != uploaded_file.name:
        df, error = load_file(uploaded_file)

        if df is None:
            # Show the problem and leave any previous dataset untouched.
            st.sidebar.error(error)
        else:
            st.session_state["data"] = df
            # Keep an untouched copy for the Reset button
            st.session_state["original_data"] = df.copy()
            # Remember the file name so Streamlit does not reload it on every rerun
            st.session_state["current_file"] = uploaded_file.name
            # Reset the undo history and transformation log for the new dataset
            st.session_state["history"] = []
            st.session_state["transform_log"] = []

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Upload and Overview",
    "Cleaning and Prep",
    "Visualization",
    "Export and Report"
])

with tab1:
    upload_and_overview.upload_overview()

with tab2:
    cleaning_and_prep.clean_prep()

with tab3:
    visualization.visualization()

with tab4:
    export_and_report.export_report()
