import re

import streamlit as st
import numpy as np
import pandas as pd


# ---------- TRANSFORMATION ENGINE ----------

def init_transform_system():
    if "transform_log" not in st.session_state:
        st.session_state["transform_log"] = []

    if "history" not in st.session_state:
        st.session_state["history"] = []


def _json_safe(value):
    """Recursively convert a value into something json.dumps can handle.

    The transformation log is exported as JSON on the Export tab, so numpy
    scalars, pandas Index objects, sets, etc. must be turned into plain
    Python types first. This keeps the export from ever crashing.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Index):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def log_step(operation, parameters, columns):
    st.session_state["transform_log"].append({
        "operation": operation,
        "parameters": _json_safe(parameters),
        "columns": _json_safe(columns)
    })


def save_state():
    """Snapshot the current data so the last action can be undone.

    Always called immediately before a real mutation, so one history entry
    corresponds to exactly one transformation log entry. That keeps Undo and
    the log in step with each other.
    """
    st.session_state["history"].append(
        st.session_state["data"].copy()
    )


def undo_last():
    if st.session_state["history"]:
        st.session_state["data"] = st.session_state["history"].pop()
        if st.session_state["transform_log"]:
            st.session_state["transform_log"].pop()


def reset_all():
    st.session_state["data"] = st.session_state["original_data"].copy()
    st.session_state["transform_log"] = []
    st.session_state["history"] = []


def _formula_is_allowed(expr):
    """Only allow safe column arithmetic in the formula box.

    DataFrame.eval can otherwise reach Python internals through the `@` prefix
    and dunder attributes. This app only needs arithmetic on column names, so
    anything that could escape that is rejected outright.
    """
    lowered = expr.lower()
    if "@" in expr or "__" in expr:
        return False
    banned = ("import", "lambda", "exec(", "eval(", "open(", "os.",
              "sys.", "subprocess", "globals", "locals", "getattr", "setattr")
    return not any(token in lowered for token in banned)


def _clean_to_numeric(series):
    """Convert a messy text column into numbers without mangling values.

    Handles currency symbols, thousands separators and accounting-style
    negatives like "(123)", while leaving scientific notation and decimals
    intact. Anything that is not really a number becomes NaN.
    """
    # If it is already numeric there is nothing to clean.
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip()

    # Accounting negatives: "(123)" -> "-123"
    negatives = s.str.match(r"^\(.*\)$")
    s = s.str.replace(r"^\((.*)\)$", r"\1", regex=True)

    # Remove currency symbols, percent signs, spaces and thousands separators,
    # but keep digits, a single decimal point, sign and the exponent marker.
    s = s.str.replace(r"[,$€£%\s]", "", regex=True)

    out = pd.to_numeric(s, errors="coerce")
    out = out.mask(negatives.fillna(False), -out.abs())
    return out


@st.cache_data
def compute_missing(df):
    df_missing = df.loc[:, df.isnull().any()]
    if df_missing.shape[1] == 0:
        return None

    return pd.DataFrame({
        "Column": df_missing.columns,
        "Missing Count": df_missing.isnull().sum().values,
        "Missing Percentage (%)": (df_missing.isnull().sum().values / len(df) * 100).round(2),
        "Data type": df.dtypes[df_missing.columns].astype(str).values
    })


@st.cache_data
def compute_missin_percent(df):
    return df.isnull().mean() * 100


@st.cache_data
def compute_duplicates(df, cols, mode):
    if cols:
        if mode == "Including selected columns":
            subset_cols = cols
        else:
            subset_cols = [c for c in df.columns if c not in cols]

        if not subset_cols:
            return pd.DataFrame()

        return df[df.duplicated(subset=subset_cols, keep=False)]
    else:
        return df[df.duplicated(keep=False)]


@st.cache_data
def compute_binning_preview(df, col, method, num_bins, labels):
    if method == "Equal-width (Standard)":
        preview_series = pd.cut(
            df[col],
            bins=num_bins,
            labels=labels if labels else None
        )
    else:
        preview_series = pd.qcut(
            df[col],
            q=num_bins,
            labels=labels if labels else None,
            duplicates="drop"
        )

    # Count on the (ordered) categorical series so bins stay in numeric order,
    # then relabel the index as text for display.
    distribution = preview_series.value_counts().sort_index()
    distribution.index = distribution.index.astype(str)

    return preview_series, distribution


# ------------------------------------------------------------ MAIN LOGIC - CLEANING AND PREP --------------------------------------------------

def clean_prep():
    st.title("Data Cleaning and Preparation")

    st.subheader("Page overview")
    st.write("Here you can clean up your data. Do not reload your page, otherwise you'll lose your progress.")
    st.write("Page goes in the following order: \n\n")
    st.code("1. Handle Missing Values \n\n"
            "2. Duplicate Detection \n\n"
            "3. Column Operations (drop, rename, create new, binning) \n\n"
            "4. Data Type Conversion and Parsing \n\n"
            "5. Categorical Data Tools \n\n"
            "6. Numeric Cleaning \n\n"
            "7. Normalization and Scaling \n\n"
            "8. Data Validation Rules \n\n")
    st.write("**Dataset Preview** is displayed at the end of the page. All your changes will be reflected there.")

    # Show the message from the previous action (set just before st.rerun)
    if "last_action" in st.session_state:
        st.success(st.session_state["last_action"])
        del st.session_state["last_action"]

    # Check dataset
    if "data" not in st.session_state or st.session_state["data"] is None:
        st.warning("Please upload a dataset first in the sidebar.")
        return

    init_transform_system()

    st.space("medium")

    # ----------------------------- 1 MISSING VALUES (NULL HANDLING) -------------------

    with st.container(border=True):

        st.subheader("1 Handle Missing Values", text_alignment="center")

        # Tracks whether a real change happened this run (drives the preview below)
        action_triggered = False
        df_before = st.session_state["data"].copy()

        # Drop all rows with missing values
        with st.container(horizontal_alignment="right"):
            if st.button("Drop all rows with missing values", key="drop_missing_btn", type="primary"):
                save_state()

                new_df = st.session_state["data"].copy()
                rows_to_drop = int(new_df.isna().any(axis=1).sum())
                new_df = new_df.dropna()

                st.session_state["data"] = new_df.reset_index(drop=True)

                log_step(
                    "Drop missing rows",
                    {"rows_removed": rows_to_drop},
                    "ALL"
                )

                st.success(f"Dropped {rows_to_drop} rows successfully!")
                action_triggered = True

        # Per-column missing value handling - column selector
        col_select = st.selectbox(
            "Select columns to handle missing values",
            [None] + st.session_state["data"].columns.tolist(),
            key="fill_column_select"
        )

        st.space("xsmall")

        left, right = st.columns([1, 2])
        # Drop rows in selected column
        with left:
            with st.container(horizontal=True, horizontal_alignment="right", gap=None):
                with st.container():
                    if st.button("Drop selected", key="drop_missing_col_btn", type="primary"):
                        if not col_select:
                            st.warning("Please select a column first.")
                        else:
                            save_state()

                            new_df = st.session_state["data"].copy()
                            rows_to_drop = int(new_df[col_select].isna().sum())
                            new_df = new_df.dropna(subset=[col_select])

                            st.session_state["data"] = new_df.reset_index(drop=True)

                            log_step(
                                "Drop selected rows",
                                {"rows_removed": rows_to_drop},
                                [col_select]
                            )
                            st.success(f"Dropped {rows_to_drop} rows with missing values in '{col_select}' successfully!")
                            action_triggered = True
                with st.container(width="content"):
                    # Fill selected values with most frequent categorical value
                    if st.button("Fill with Most Frequent Category", key="fill_frequent_cat_btn"):
                        if not col_select:
                            st.warning("Please select a column first.")
                        else:
                            mode_vals = st.session_state["data"][col_select].mode(dropna=True)
                            if mode_vals.empty:
                                st.warning(f"'{col_select}' has no non-missing value to use.")
                            else:
                                save_state()

                                new_df = st.session_state["data"].copy()
                                new_df[col_select] = new_df[col_select].fillna(mode_vals.iloc[0])

                                st.session_state["data"] = new_df

                                log_step(
                                    "Replace selected with most frequent category",
                                    {"method": "most_frequent"},
                                    [col_select]
                                )

                                st.success(f"Filled missing values in '{col_select}' with most frequent category")
                                action_triggered = True

        # Drop columns above a threshold
        with right:
            with st.container(horizontal=True, horizontal_alignment="right"):
                with st.container(width="content"):
                    with st.popover("Threshold"):
                        threshold = st.slider(
                            "Drop columns with more than % missing values",
                            0, 100, 50,
                            key="missing_threshold_slider"
                        )
                with st.container(width="content"):
                    if st.button("Drop Columns Above Threshold", key="drop_missing_threshold_btn", type="primary"):
                        save_state()

                        new_df = st.session_state["data"].copy()

                        missing_percent = compute_missin_percent(new_df)

                        cols_to_drop = list(missing_percent[missing_percent > threshold].index)

                        new_df = new_df.drop(columns=cols_to_drop)

                        st.session_state["data"] = new_df.reset_index(drop=True)

                        log_step(
                            "Drop columns above missing threshold",
                            {"threshold_percent": threshold},
                            cols_to_drop
                        )

                        st.success(f"Dropped {len(cols_to_drop)} columns with more than {threshold}% missing values")
                        action_triggered = True

        st.space("xsmall")

        # Mean, Median, Mode
        st.write("Replace the selected numeric column with:")
        btn1, btn2, btn3 = st.columns(3)
        if btn1.button("Mean", key="fill_mean_btn", width="stretch"):
            if not col_select:
                st.warning("Please select a column first.")
            elif not pd.api.types.is_numeric_dtype(st.session_state["data"][col_select]):
                st.warning(f"'{col_select}' is not numeric, so mean cannot be used.")
            else:
                save_state()

                new_df = st.session_state["data"].copy()
                new_df[col_select] = new_df[col_select].fillna(new_df[col_select].mean())

                st.session_state["data"] = new_df

                log_step("Fill with Mean", {"method": "mean"}, [col_select])
                st.success(f"Filled missing values in '{col_select}' with mean")
                action_triggered = True
        if btn2.button("Median", key="fill_median_btn", width="stretch"):
            if not col_select:
                st.warning("Please select a column first.")
            elif not pd.api.types.is_numeric_dtype(st.session_state["data"][col_select]):
                st.warning(f"'{col_select}' is not numeric, so median cannot be used.")
            else:
                save_state()

                new_df = st.session_state["data"].copy()
                new_df[col_select] = new_df[col_select].fillna(new_df[col_select].median())

                st.session_state["data"] = new_df

                log_step("Fill with Median", {"method": "median"}, [col_select])
                st.success(f"Filled missing values in '{col_select}' with median")
                action_triggered = True
        if btn3.button("Mode", key="fill_mode_btn", width="stretch"):
            if not col_select:
                st.warning("Please select a column first.")
            else:
                mode_vals = st.session_state["data"][col_select].mode(dropna=True)
                if mode_vals.empty:
                    st.warning(f"'{col_select}' has no non-missing value to use.")
                else:
                    save_state()

                    new_df = st.session_state["data"].copy()
                    new_df[col_select] = new_df[col_select].fillna(mode_vals.iloc[0])

                    st.session_state["data"] = new_df

                    log_step("Fill with Mode", {"method": "mode"}, [col_select])
                    st.success(f"Filled missing values in '{col_select}' with mode")
                    action_triggered = True

        st.space("xsmall")

        with st.container(horizontal=True):
            # Fill selected values with custom value
            with st.container():
                constant = st.text_input("Fill selected missing values with:", value="0", key="fill_constant_input")
                if st.button("Apply Custom Fill", key="apply_custom_fill_btn", type="primary"):
                    if not col_select:
                        st.warning("Please select a column first.")
                    else:
                        save_state()

                        new_df = st.session_state["data"].copy()
                        try:
                            value = float(constant)
                        except (TypeError, ValueError):
                            value = constant
                        new_df[col_select] = new_df[col_select].fillna(value)

                        st.session_state["data"] = new_df

                        log_step(
                            "Custom replacement for missing",
                            {"sample_input": constant},
                            [col_select]
                        )

                        st.success(f"Filled missing values in '{col_select}' with '{value}'")
                        action_triggered = True

            st.space("medium")

            # Forward / backward fill
            with st.container():
                forward_backward = st.radio(
                    "Select option",
                    ["Forward fill", "Backward fill"],
                    horizontal=True
                )
                if st.button("Apply Forward / Backward fill", type="primary"):
                    if not col_select:
                        st.warning("Please select a column first.")
                    else:
                        save_state()

                        new_df = st.session_state["data"].copy()
                        if forward_backward == "Forward fill":
                            new_df[col_select] = new_df[col_select].ffill()
                        elif forward_backward == "Backward fill":
                            new_df[col_select] = new_df[col_select].bfill()

                        st.session_state["data"] = new_df

                        log_step(
                            "Fill missing - Forward/Backward",
                            {"setting": forward_backward},
                            [col_select]
                        )

                        st.success("Forward/Backward setting applied")
                        action_triggered = True

        # ------ BEFORE/AFTER PREVIEW -----
        if action_triggered:
            st.divider()
            st.subheader("Action Preview")
            df_after = st.session_state["data"]

            m1, m2, m3 = st.columns(3)

            # Row count change
            rows_b, rows_a = len(df_before), len(df_after)
            m1.metric("Total Rows", rows_a, delta=rows_a - rows_b)

            # Null count change (inverse color: fewer nulls shows green)
            nulls_b = int(df_before.isnull().sum().sum())
            nulls_a = int(df_after.isnull().sum().sum())
            m2.metric("Total Nulls", nulls_a, delta=nulls_a - nulls_b, delta_color="inverse")

            # Columns remaining
            cols_b, cols_a = len(df_before.columns), len(df_after.columns)
            m3.metric("Columns", cols_a, delta=cols_a - cols_b)

            st.toast("Data updated!")

        # Missing values overview, computed from the current data so it always
        # reflects the action that was just applied (no stale table).
        st.divider()
        if st.checkbox("Show missing values", key="show_missing_checkbox", value=True):
            missing = compute_missing(st.session_state["data"])
            if missing is not None:
                st.markdown("**Missing Values Overview:**")
                st.dataframe(missing, hide_index=True, width="content")
            else:
                st.success("No missing values remaining!")

    st.space("xlarge")

    # ------------------- 2 DUPLICATE DETECTION -------------------

    with st.container(border=True):
        st.subheader("2 Duplicate Detection", text_alignment="center")

        # Include / exclude toggle
        mode = st.radio(
            "Show duplicates:",
            ["Including selected columns", "Excluding selected columns"],
            key="column_mode",
            horizontal=True
        )

        # Column selector
        cols = st.multiselect(
            "Select columns to check duplicates (leave empty for full-row duplicates)",
            st.session_state["data"].columns,
            key="duplicate_columns"
        )

        # Detect duplicates (read-only: no history or log entry)
        if st.button("Detect Duplicates", key="detect_duplicates_btn"):

            new_df = st.session_state["data"]

            duplicates = compute_duplicates(new_df, cols, mode)

            if duplicates.empty:
                st.warning("No duplicates found for the current selection.")
            else:
                st.dataframe(duplicates.head())
                st.info(f"{len(duplicates)} duplicate rows found")

        st.divider()

        left, right = st.columns([2, 1], vertical_alignment="bottom")

        # Keep option
        keep_option = left.radio(
            "Keep which duplicate?",
            ["first", "last"],
            key="keep_option",
            horizontal=True
        )

        st.space("stretch")

        # Remove duplicates
        if right.button("Remove Duplicates", key="remove_duplicates_btn", type="primary", width="stretch"):
            save_state()

            new_df = st.session_state["data"].copy()

            if cols:
                if mode == "Including selected columns":
                    subset_cols = cols
                else:
                    subset_cols = [c for c in new_df.columns if c not in cols]

                new_df = new_df.drop_duplicates(subset=subset_cols, keep=keep_option).reset_index(drop=True)
            else:
                new_df = new_df.drop_duplicates(keep=keep_option).reset_index(drop=True)

            removed = len(st.session_state["data"]) - len(new_df)
            st.session_state["data"] = new_df

            # Logging
            if not cols:
                col_selected_or_not = "Full-row duplicates"
            else:
                col_selected_or_not = cols

            log_step(
                "Remove selected duplicates",
                {"keep": keep_option, "mode": mode, "rows_removed": removed},
                [col_selected_or_not]
            )

            st.success(f"{removed} duplicate rows removed")

    st.space("xlarge")

    # ------------------- 3 COLUMN OPERATIONS -------------------

    with st.container(border=True):
        st.subheader("3 Column Operations", text_alignment="center")

        st.space("medium")

        with st.container(horizontal=True):
            with st.container():
                # Drop columns
                st.write("**Drop Columns**")
                columns_to_drop = st.multiselect(
                    "Select columns to drop",
                    st.session_state["data"].columns,
                    key="columns_to_drop"
                )
                if st.button("Drop Columns", key="drop_selected_columns", type="primary"):
                    if not columns_to_drop:
                        st.warning("Please select at least one column to drop.")
                    else:
                        save_state()

                        new_df = st.session_state["data"].drop(columns=columns_to_drop)
                        st.session_state["data"] = new_df

                        log_step(
                            "Drop columns",
                            {"count": len(columns_to_drop)},
                            list(columns_to_drop)
                        )
                        st.success("Selected columns dropped")

            st.space("medium")

            with st.container():
                # Rename columns
                st.write("**Rename Columns**")

                old_name = st.selectbox(
                    "Select column to rename",
                    st.session_state["data"].columns,
                    key="rename_column"
                )
                new_name = st.text_input(
                    "Enter new column name",
                    key="new_column_name"
                )
                if st.button("Rename Column", key="rename_column_btn", type="primary"):
                    clean_new_name = new_name.strip() if new_name else ""
                    if not clean_new_name:
                        st.warning("Please enter a new column name.")
                    elif clean_new_name != old_name and clean_new_name in st.session_state["data"].columns:
                        st.warning(f"A column named '{clean_new_name}' already exists. Choose a different name.")
                    else:
                        save_state()

                        new_df = st.session_state["data"].rename(columns={old_name: clean_new_name})
                        st.session_state["data"] = new_df

                        log_step(
                            "Rename columns",
                            {"New column name": clean_new_name},
                            [old_name]
                        )
                        st.success("Column renamed successfully")

        st.divider()

        with st.container(horizontal=True):
            with st.container():

                # Create new columns
                st.write("**Create New Columns**")

                with st.popover("Helper! / Rules", type="tertiary"):
                    st.write("Write formulas using your exact column names. \n\n"
                             "Before applying a new formula, convert the columns into *String* or *Integer*. \n\n"
                             "*Examples:* `Price / Quantity`, `Age * 1.2`, `Salary - Salary.mean()` \n\n"
                             "It supports standard arithmetic (+, -, *, /) and boolean operators (&, |, ~). \n\n"
                             "Use `` ` `` backticks if column names have spaces.")

                with st.container():
                    new_formula_col = st.text_input("New Column Name", key="new_formula_col")
                with st.container():
                    formula_input = st.text_input("Formula", key="formula_input")

                if st.button("Apply Formula", key="btn_apply_formula", type="primary"):

                    if new_formula_col and formula_input:
                        if not _formula_is_allowed(formula_input):
                            st.error("This formula uses names or symbols that are not allowed. "
                                     "Use only your column names, numbers and arithmetic operators.")
                        else:
                            new_df = st.session_state["data"].copy()
                            try:
                                # Evaluate the arithmetic expression over the columns.
                                result = new_df.eval(formula_input, engine="python")

                                save_state()
                                new_df[new_formula_col] = result
                                st.session_state["data"] = new_df

                                log_step(
                                    "Custom formula",
                                    {"New column": new_formula_col, "Formula": formula_input},
                                    []
                                )
                                st.success(f"Successfully created '{new_formula_col}'.")

                            except Exception as e:
                                st.error("Error evaluating formula. Check your column names and syntax. \n\n"
                                         "See *Helper!/Rules* above. \n\n"
                                         f"Details: {e}")
                    else:
                        st.warning("Please provide both a new column name and a formula.")

            st.space("medium")

            with st.container():
                # Binning Numeric Columns
                st.write("**Binning Numeric Columns**")

                numeric_columns = st.session_state["data"].select_dtypes(include=["number"]).columns.tolist()

                if numeric_columns:
                    bin_target_col = st.selectbox("Select column to bin", numeric_columns, key="bin_target_col")
                    new_binned_col = st.text_input("New Binned Column Name", key="new_binned_col")

                    bin_col1, bin_col2 = st.columns(2)
                    with bin_col1:
                        bin_method = st.radio(
                            "Binning Method",
                            ["Equal-width (Standard)", "Quantile (Equal-sized groups)"],
                            key="bin_method"
                        )
                    with bin_col2:
                        num_bins = st.number_input("Number of bins", min_value=2, max_value=50, value=4, key="num_bins")

                    with st.container(horizontal=True, horizontal_alignment="right"):
                        with st.container():
                            use_labels = st.checkbox("Use custom labels")
                        with st.container():
                            n_rows = len(st.session_state["data"])
                            if n_rows > 1:
                                suggested_bins = max(2, int(np.ceil(np.log2(n_rows) + 1)))
                            else:
                                suggested_bins = 2
                            st.caption(f"Suggested bins: {suggested_bins}", help="Suggestion uses Sturges' rule - a simple, "
                                       "commonly used statistical formula to determine the optimal number of bins for a histogram, "
                                       "based on the number of observations. It works best for relatively small, normal datasets.")

                    labels = None
                    if use_labels:
                        labels_input = st.text_input("Enter labels (comma-separated)", placeholder="Low, Medium, High, Very High")
                        if labels_input:
                            labels = [l.strip() for l in labels_input.split(",")]

                    if labels and len(labels) != num_bins:
                        st.error("Number of labels must match number of bins.")

                    btn1, btn2 = st.columns([1, 3])

                    with btn1:
                        if st.button("Apply Binning", key="btn_apply_binning", type="primary"):
                            if not new_binned_col:
                                st.warning("Please provide a name for the new binned column.")
                            elif labels and len(labels) != num_bins:
                                st.error("Number of labels must match number of bins.")
                            else:
                                new_df = st.session_state["data"].copy()
                                try:
                                    if bin_method == "Equal-width (Standard)":
                                        # pd.cut divides the range into equal-width intervals
                                        binned = pd.cut(
                                            new_df[bin_target_col],
                                            bins=num_bins,
                                            labels=labels if labels else None
                                        )
                                    else:
                                        # pd.qcut divides the data so each bin has roughly the same number of records.
                                        # duplicates="drop" prevents errors when many identical values fall on a bin edge.
                                        binned = pd.qcut(
                                            new_df[bin_target_col],
                                            q=num_bins,
                                            labels=labels if labels else None,
                                            duplicates="drop"
                                        )

                                    save_state()
                                    new_df[new_binned_col] = binned
                                    st.session_state["data"] = new_df

                                    log_step(
                                        "Binning numeric columns",
                                        {
                                            "New column name": new_binned_col,
                                            "Binning method": bin_method,
                                            "No. of bins": int(num_bins)
                                        },
                                        [bin_target_col]
                                    )
                                    st.success(f"Successfully created binned column '{new_binned_col}'.")
                                except Exception as e:
                                    st.error(f"Error during binning: {e}")

                    with btn2:
                        with st.popover("Preview bins before applying", width="stretch"):
                            try:
                                preview_series, distribution = compute_binning_preview(
                                    st.session_state["data"],
                                    bin_target_col,
                                    bin_method,
                                    num_bins,
                                    labels
                                )

                                st.write("### Bin Preview")

                                preview_df = pd.DataFrame({
                                    bin_target_col: st.session_state["data"][bin_target_col],
                                    "Bin": preview_series.astype(str)
                                })
                                st.dataframe(preview_df.head(10))
                                st.write("### Bin Distribution")
                                st.bar_chart(distribution)

                            except Exception as e:
                                st.error(f"Preview error: {e}")

                else:
                    st.info("No numeric columns found in the dataset to bin.")

    st.space("xlarge")

    # ------------------------ 4 DATA TYPES AND PARSING ------------------------

    with st.container(border=True):
        st.subheader("4 Data Types and Parsing", text_alignment="center")
        st.space("small")

        with st.container(horizontal=True):
            with st.container():
                # Datatype conversion
                st.write("**Data Type Conversion**")
                with st.container(horizontal=True):
                    with st.container():
                        column_to_convert = st.selectbox(
                            "Select column to convert",
                            st.session_state["data"].columns,
                            key="column_to_convert"
                        )
                    with st.container():
                        new_type = st.selectbox(
                            "Convert to",
                            ["Numeric", "Categorical", "Datetime", "String", "Boolean"],
                            key="new_type"
                        )
                with st.container(horizontal_alignment="right"):
                    if st.button("Convert Column Type", key="convert_column_type", type="primary"):

                        new_df = st.session_state["data"].copy()

                        try:
                            # Convert to numeric
                            if new_type == "Numeric":
                                new_df[column_to_convert] = _clean_to_numeric(new_df[column_to_convert])

                            # Convert to categorical
                            elif new_type == "Categorical":
                                new_df[column_to_convert] = new_df[column_to_convert].astype("category")

                            # Convert to datetime
                            elif new_type == "Datetime":
                                new_df[column_to_convert] = pd.to_datetime(
                                    new_df[column_to_convert],
                                    errors="coerce"
                                )

                            # Convert to string
                            elif new_type == "String":
                                new_df[column_to_convert] = new_df[column_to_convert].astype("string")

                            # Convert to boolean
                            elif new_type == "Boolean":
                                series = new_df[column_to_convert].astype(str).str.lower().str.strip()
                                # Treat "1.0"/"0.0" the same as "1"/"0"
                                series = series.str.replace(r"\.0$", "", regex=True)

                                mapping = {
                                    'true': True, '1': True, 'yes': True, 'y': True,
                                    'false': False, '0': False, 'no': False, 'n': False
                                }
                                bool_series = series.map(mapping)

                                unmapped = int((bool_series.isna() & new_df[column_to_convert].notna()).sum())
                                if unmapped > 0:
                                    st.warning(f"{unmapped} values could not be converted and are set as NaN")

                                new_df[column_to_convert] = bool_series

                            save_state()
                            st.session_state["data"] = new_df
                            log_step(
                                "Convert column type",
                                {"Convert to": new_type},
                                [column_to_convert]
                            )
                            st.success(f"Column converted to {new_type} successfully")

                        except Exception as e:
                            st.error("Conversion failed")
                            st.write(e)

            st.space("large")

            with st.container():

                # Datetime parsing
                st.markdown("**Datetime Parsing**")

                with st.container(horizontal=True):
                    with st.container():
                        datetime_column = st.selectbox(
                            "Select column to parse as Datetime",
                            st.session_state["data"].columns,
                            key="datetime_column"
                        )
                    with st.container():
                        date_format_choice = st.radio(
                            "Select parsing method",
                            ["Auto-detect", "Manual Format"],
                            key="date_format_choice"
                        )

                manual_format = None
                if date_format_choice == "Manual Format":
                    manual_format = st.selectbox(
                        "Select or type format (e.g., %d/%m/%Y)",
                        ["%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%b-%Y"],
                        key="manual_format_input"
                    )
                    st.caption("Hint: %d=Day, %m=Month, %Y=Year (4-digit)")

                with st.container(horizontal_alignment="right"):
                    if st.button("Parse to Datetime", key="parse_datetime_btn", type="primary"):

                        new_df = st.session_state["data"].copy()

                        try:
                            if date_format_choice == "Auto-detect":
                                parsed = pd.to_datetime(new_df[datetime_column], errors="coerce")
                            else:
                                parsed = pd.to_datetime(new_df[datetime_column], format=manual_format, errors="coerce")

                            # Only commit if parsing actually produced dates.
                            if parsed.isna().all():
                                st.warning("All values became NaT (Not a Time). Check if your chosen format matches the data.")
                            else:
                                save_state()
                                new_df[datetime_column] = parsed
                                st.session_state["data"] = new_df
                                log_step(
                                    "Parse to datetime",
                                    {"Parsing method": date_format_choice},
                                    [datetime_column]
                                )
                                st.success(f"Successfully parsed '{datetime_column}' as Datetime.")

                        except Exception as e:
                            st.error(f"Error during parsing: {e}")

        # Show datatypes (from current data, so always up to date)
        datatypes = pd.DataFrame({
            "Column": st.session_state["data"].columns.astype(str),
            "Data type": st.session_state["data"].dtypes.astype(str).values
        })
        st.subheader("Data types:")
        st.dataframe(datatypes, hide_index=True, width="content")

    st.space("medium")

    with st.container(horizontal=True):

        # -------------------------------------- 5 CATEGORICAL DATA TOOLS ----------------------------

        with st.container(border=True):
            st.subheader("5 Categorical Data Tools")
            st.space("medium")

            cat_cols = st.session_state["data"].select_dtypes(include=["object", "category", "string"]).columns

            if len(cat_cols) > 0:

                cat_col = st.selectbox(
                    "Select categorical column",
                    cat_cols,
                    key="categorical_column"
                )

                clean_option = st.selectbox(
                    "Standardization option",
                    ["Trim whitespace", "Lowercase", "Title Case"],
                    key="categorical_clean_option"
                )

                if st.button("Apply Standardization", key="apply_standardization", type="primary"):

                    save_state()

                    new_df = st.session_state["data"].copy()

                    # Work on non-null values only, so missing stays missing
                    # (astype(str) on the whole column would turn NaN into "nan").
                    source = new_df[cat_col]
                    if isinstance(source.dtype, pd.CategoricalDtype):
                        source = source.astype(object)
                        new_df[cat_col] = source
                    mask = source.notna()
                    cleaned = source[mask].astype(str)

                    if clean_option == "Trim whitespace":
                        cleaned = cleaned.str.strip()
                    elif clean_option == "Lowercase":
                        cleaned = cleaned.str.lower()
                    elif clean_option == "Title Case":
                        cleaned = cleaned.str.title()

                    new_df.loc[mask, cat_col] = cleaned
                    st.session_state["data"] = new_df

                    log_step("Standardization", {"Option": clean_option}, [cat_col])
                    st.success("Categorical values standardized")

                st.divider()

                # ---------------- Mapping / Replacement ----------------
                st.markdown("**Mapping / Replacement**")

                fill_other_option = st.checkbox("Set unmatched values to 'Other'", value=False, key="set_unmatched_other")

                if 'mapping_data' not in st.session_state:
                    # Initialize the mapping table as text columns so the editor
                    # accepts text entries.
                    st.session_state['mapping_data'] = pd.DataFrame(
                        {"Original Value": pd.Series([], dtype="string"),
                         "Mapped Value": pd.Series([], dtype="string")}
                    )

                mapping_df = st.data_editor(
                    st.session_state['mapping_data'],
                    num_rows="dynamic",
                    key="mapping_editor"
                )

                if st.button("Apply Mapping", key="apply_mapping_btn", type="primary"):
                    save_state()
                    new_df = st.session_state["data"].copy()

                    mapping_dict = (
                        mapping_df.dropna(subset=["Original Value"])
                        .astype({"Original Value": "string"})
                        .set_index("Original Value")["Mapped Value"]
                        .to_dict()
                    )

                    original = new_df[cat_col]
                    # Compare keys as text, but keep NaN as NaN throughout.
                    keys = original.where(original.notna()).astype("string")
                    mapped = keys.map(mapping_dict)

                    if fill_other_option:
                        # Unmatched but non-null values become "Other"; nulls stay null.
                        mapped = mapped.mask(original.notna() & mapped.isna(), "Other")
                    else:
                        # Unmatched values keep their original value.
                        mapped = mapped.where(mapped.notna(), original)

                    changed_count = int(
                        (original.fillna("\x00").astype(str) != mapped.fillna("\x00").astype(str)).sum()
                    )
                    st.info(f"{changed_count} values were changed by the mapping.")

                    new_df[cat_col] = mapped
                    st.session_state["data"] = new_df
                    st.session_state['mapping_data'] = mapping_df

                    log_step(
                        "Mapping / Replacement",
                        {"Set unmatched to 'Other'": fill_other_option},
                        [cat_col]
                    )
                    st.success("Mapping applied successfully")

                st.divider()

                st.markdown("Rare Category Grouping")

                rare_threshold = st.slider(
                    "Minimum frequency (%)",
                    0, 20, 5
                )

                if st.button("Group Rare Categories", key="group_rare_btn"):

                    save_state()

                    new_df = st.session_state["data"].copy()

                    # Work on an object copy so adding the new "Other" label never
                    # fails on a fixed categorical dtype.
                    series = new_df[cat_col].astype(object)
                    freq = series.value_counts(normalize=True)
                    rare = freq[freq < rare_threshold / 100].index

                    series = series.where(~series.isin(rare), "Other")
                    new_df[cat_col] = series

                    st.session_state["data"] = new_df

                    log_step("Grouping rare categories", {"Frequency": rare_threshold}, [cat_col])
                    st.success("Rare categories grouped into 'Other'")

            else:
                st.info("No categorical columns found in the dataset.")

        st.space("medium")

        # --------------------------------------- 6 NUMERIC CLEANING ------------------------------

        with st.container(border=True):
            st.subheader("6 Numeric Cleaning")

            st.space("medium")

            num_cols = st.session_state["data"].select_dtypes(include=["number"]).columns.tolist()

            if num_cols:
                outlier_col = st.selectbox(
                    "Select numeric column",
                    num_cols
                )
                action = st.radio(
                    "Action for outliers",
                    ["Show Outliers", "Remove Outliers", "Cap Outliers"]
                )

                if st.button("Run Outlier Detection", type="primary"):

                    col_data = st.session_state["data"][outlier_col]

                    # IQR calculation
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR

                    # Detect outliers (missing values are never outliers)
                    outlier_mask = (col_data < lower) | (col_data > upper)
                    outliers = st.session_state["data"][outlier_mask.fillna(False)]

                    # Column stats
                    col_mean = col_data.mean()
                    col_median = col_data.median()
                    col_min = col_data.min()
                    col_max = col_data.max()

                    st.divider()
                    st.markdown("### Column Statistics")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Mean", round(col_mean, 2))
                    with c2:
                        st.metric("Median", round(col_median, 2))
                    with c3:
                        st.metric("Min", round(col_min, 2))
                    with c4:
                        st.metric("Max", round(col_max, 2))

                    # Outlier boundaries
                    st.markdown("### Outlier Boundaries (IQR Method)")
                    b1, b2 = st.columns(2)
                    with b1:
                        st.metric("Lower Limit", round(lower, 2))
                    with b2:
                        st.metric("Upper Limit", round(upper, 2))

                    st.divider()

                    # Outlier summary
                    st.markdown("### Outlier Detection Summary")
                    total_rows = len(st.session_state["data"])
                    outlier_count = int(outlier_mask.fillna(False).sum())
                    percent = (outlier_count / total_rows) * 100 if total_rows else 0.0

                    s1, s2, s3 = st.columns(3)
                    with s1:
                        st.metric("Total Rows", total_rows)
                    with s2:
                        st.metric("Outliers Found", outlier_count)
                    with s3:
                        st.metric("Percentage", f"{round(percent, 2)}%")

                    # Action
                    if action == "Show Outliers":
                        # Read-only: no history snapshot, no log entry.
                        st.markdown("### Detected Outliers")
                        st.dataframe(outliers)

                    elif action == "Remove Outliers":
                        save_state()
                        before = len(st.session_state["data"])

                        # Keep rows inside the bounds and keep missing values.
                        keep_mask = ~((col_data < lower) | (col_data > upper))
                        st.session_state["data"] = st.session_state["data"][keep_mask].reset_index(drop=True)

                        removed = before - len(st.session_state["data"])
                        st.success(f"{removed} rows removed")

                        log_step(
                            "Remove Outliers",
                            {"method": "IQR", "rows_removed": removed},
                            [outlier_col]
                        )

                    elif action == "Cap Outliers":
                        save_state()
                        cap_count = outlier_count

                        new_df = st.session_state["data"].copy()
                        new_df[outlier_col] = new_df[outlier_col].clip(lower, upper)
                        st.session_state["data"] = new_df

                        st.success(f"{cap_count} values capped")

                        log_step(
                            "Cap Outliers",
                            {"method": "IQR", "values_capped": cap_count},
                            [outlier_col]
                        )
            else:
                st.info("No numeric columns found in the dataset.")

        st.space("medium")

        # ------------------------------ 7 NORMALIZATION AND SCALING --------------------

        with st.container(border=True):
            st.subheader("7 Normalization / Scaling")

            st.space("medium")

            numeric_cols = st.session_state["data"].select_dtypes(include=["number"]).columns.tolist()

            if not numeric_cols:
                st.warning("No numeric columns available for scaling.")
            else:
                scale_columns = st.multiselect(
                    "Select numeric columns to scale",
                    numeric_cols
                )

                scaling_method = st.radio(
                    "Choose scaling method",
                    ["Min-Max Scaling", "Z-Score Standardization"]
                )

                if st.button("Apply Scaling", type="primary"):

                    if not scale_columns:
                        st.warning("Please select at least one column.")
                    else:
                        save_state()

                        # Before stats
                        before_stats = st.session_state["data"][scale_columns].describe()

                        new_df = st.session_state["data"].copy()

                        if scaling_method == "Min-Max Scaling":
                            for col in scale_columns:
                                min_val = new_df[col].min()
                                max_val = new_df[col].max()
                                if max_val - min_val != 0:
                                    new_df[col] = (new_df[col] - min_val) / (max_val - min_val)

                        elif scaling_method == "Z-Score Standardization":
                            for col in scale_columns:
                                mean_val = new_df[col].mean()
                                std_val = new_df[col].std()
                                if std_val != 0:
                                    new_df[col] = (new_df[col] - mean_val) / std_val

                        st.session_state["data"] = new_df

                        # After stats
                        after_stats = st.session_state["data"][scale_columns].describe()

                        st.success("Scaling applied successfully")
                        st.markdown("### Before Scaling Stats")
                        st.dataframe(before_stats)
                        st.markdown("### After Scaling Stats")
                        st.dataframe(after_stats)

                        log_step("Normalization", {"Method": scaling_method}, list(scale_columns))

    st.space("medium")

    # ----------------------------------- 8 DATA VALIDATION RULES --------------------------

    with st.container(border=True):

        st.subheader("8 Data Validation Rules", text_alignment="center")

        st.space("medium")

        col1, col2, col3 = st.columns(3)

        # --- Select column ---
        with col1:
            selected_column = st.selectbox(
                "Select Column",
                st.session_state["data"].columns,
                key="validation_column"
            )

        # --- Rule type ---
        with col2:
            rule_type = st.selectbox(
                "Rule Type",
                ["Numeric Range", "Allowed Categories", "Non-Null Constraint"],
                key="validation_rule_type"
            )

        # --- Rule inputs ---
        rule_config = {}

        with col3:
            if rule_type == "Numeric Range":
                min_val = st.number_input("Min Value", value=0.0, key="min_val")
                max_val = st.number_input("Max Value", value=100.0, key="max_val")
                rule_config = {"min": min_val, "max": max_val}

            elif rule_type == "Allowed Categories":
                categories = st.text_input(
                    "Allowed Categories (comma-separated)",
                    key="allowed_categories"
                )
                rule_config = {"categories": [c.strip() for c in categories.split(",") if c.strip()]}

            elif rule_type == "Non-Null Constraint":
                rule_config = {"non_null": True}

        # --- Apply rule (read-only check: finds violations, does not change data) ---
        if st.button("Validate Data", type="primary"):

            column_data = st.session_state["data"][selected_column]

            if rule_type == "Numeric Range":
                numeric = pd.to_numeric(column_data, errors="coerce")
                # Flag values outside the range (non-numeric values are ignored here).
                mask = ((numeric < rule_config["min"]) | (numeric > rule_config["max"])).fillna(False)

            elif rule_type == "Allowed Categories":
                # A violation is a value that is NOT in the allowed list (and is not missing).
                mask = (~column_data.astype("string").isin(rule_config["categories"])) & column_data.notna()

            elif rule_type == "Non-Null Constraint":
                mask = column_data.isna()

            violations = st.session_state["data"][mask].copy()

            if not violations.empty:
                violations["Violation_Column"] = selected_column
                violations["Violation_Type"] = rule_type

                st.session_state["violations"] = violations
                st.error(f"Found {len(violations)} violations.")
            else:
                st.session_state["violations"] = pd.DataFrame()
                st.success("No violations found")

        # Show violations table
        if "violations" in st.session_state and not st.session_state["violations"].empty:
            st.subheader("Violations Table")
            st.dataframe(st.session_state["violations"], width="stretch")

    st.space("xxlarge")

    # --------------------------------------- Dataset preview -------------------

    st.subheader("Current Dataset (Updated)")

    view_option = st.radio(
        "View Options",
        ["First 5 Rows", "Last 5 Rows", "Full Dataset"],
        horizontal=True
    )

    if view_option == "First 5 Rows":
        st.dataframe(st.session_state["data"].head())
    elif view_option == "Last 5 Rows":
        st.dataframe(st.session_state["data"].tail())
    else:
        st.dataframe(st.session_state["data"])

    # ------------------------------ Sidebar Transformation log ----------

    with st.sidebar:
        st.header("Transformation Log", help="Your activities will be reflected here")
        with st.container(border=True, height=300):

            if st.session_state["transform_log"]:
                for i, step in enumerate(st.session_state["transform_log"], 1):
                    with st.container(border=True):
                        st.markdown(f"**{i}. {step['operation']}**")
                        st.caption(f"Columns: {step['columns']}")
                        st.caption(f"Parameters: {step['parameters']}")
            else:
                st.info("No transformations yet.")

        with st.container():
            col1, col2 = st.columns(2)

            if col1.button("↩️ Undo", width="stretch", help="Undo your last action"):
                undo_last()
                st.rerun()

            if col2.button("🔄 Reset", width="stretch"):
                reset_all()
                st.rerun()
