import streamlit as st
import pandas as pd


def upload_overview():

    # Title and welcome message
    st.title("Your Data Wrangler and Visualizer")
    st.write("Welcome to your Data Wrangler and Visualizer! Upload your data and explore it with ease!")

    # Get dataset from session state
    df = st.session_state.get("data")
    if df is None:
        st.warning("Please upload a dataset first in the sidebar.")
        return

    # Display dataset overview
    st.header("Data Preview:")

    @st.fragment
    def show_full_dataset():
        return st.dataframe(df)
    with st.expander("Show full dataset"):
        show_full_dataset()

    # Data shape
    with st.container(border=True):
        st.markdown("**Your dataset has:**")
        col1, col2 = st.columns(2)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])

    # Duplicate rows
    with st.expander("Duplicate Rows"):
        if df.duplicated().sum() > 0:
            st.markdown("**Duplicate Rows:**")
            st.dataframe(df[df.duplicated()])
        else:
            st.info("No duplicate rows found")

    with st.container(horizontal=True):
        # Column names
        with st.container(border=True):
            st.markdown("**Column Names:**")
            st.dataframe(
                pd.DataFrame({"Column": df.columns.astype(str)}),
                hide_index=True
            )
        # Data types
        with st.container(border=True):
            st.markdown("**Data Types:**")
            st.dataframe(
                pd.DataFrame({
                    "Column": df.columns.astype(str),
                    "Data type": df.dtypes.astype(str).values
                }),
                hide_index=True
            )
        # Missing values
        with st.container(border=True):
            missing = pd.DataFrame({
                "Column": df.columns.astype(str),
                "Missing Count": df.isnull().sum().values,
                "Missing Percentage (%)": (df.isnull().sum().values / len(df) * 100).round(2)
            })
            st.markdown("**Missing Values:**")
            st.dataframe(missing, hide_index=True)

    # Summary stats
    with st.container(border=True):
        st.markdown("**Summary Statistics:**")
        # Cast to string for display so mixed-type columns always render.
        st.dataframe(df.describe(include="all").astype(str))
