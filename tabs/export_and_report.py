import io
import json
from datetime import datetime

import streamlit as st
import pandas as pd


@st.cache_data
def _dataframe_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data
def _dataframe_to_excel(df):
    """Build the Excel file in memory (not on the server's disk).

    Cached on the dataframe, so repeated reruns do not rebuild it, and nothing
    is written to a shared file on disk.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buffer.getvalue()


def export_report():
    st.title("Export & Report")

    # Check dataset
    if "data" not in st.session_state or st.session_state["data"] is None:
        st.warning("Please upload and process a dataset first.")
        return

    df = st.session_state["data"]

    if "transform_log" not in st.session_state:
        st.session_state["transform_log"] = []

    st.subheader("1. Export Cleaned Dataset")

    # CSV export
    st.download_button(
        label="Download CSV",
        data=_dataframe_to_csv(df),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

    # Excel export - built on demand so it does not run on every rerun
    # (building a large workbook on each interaction made the whole app slow).
    st.caption("Excel files take longer to build for large datasets.")
    if st.button("Prepare Excel file"):
        try:
            st.session_state["_excel_bytes"] = _dataframe_to_excel(df)
        except Exception as e:
            st.session_state.pop("_excel_bytes", None)
            st.error(f"Excel export not available: {e}")

    if "_excel_bytes" in st.session_state:
        st.download_button(
            label="Download Excel",
            data=st.session_state["_excel_bytes"],
            file_name="cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.divider()

    # ------------------- REPORT -------------------
    st.subheader("2. Transformation Report")

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "steps": st.session_state["transform_log"]
    }

    st.json(report)

    # default=str is a safety net so the download never crashes on an
    # unexpected value type.
    report_json = json.dumps(report, indent=4, default=str)

    st.download_button(
        label="Download Report (JSON)",
        data=report_json,
        file_name="transformation_report.json",
        mime="application/json"
    )

    st.divider()

    # ------------------- RECIPE -------------------
    st.subheader("3. Pipeline Recipe")

    recipe = {
        "steps": st.session_state["transform_log"]
    }

    recipe_json = json.dumps(recipe, indent=4, default=str)

    st.download_button(
        label="Download Recipe (JSON)",
        data=recipe_json,
        file_name="pipeline_recipe.json",
        mime="application/json"
    )

    st.divider()
