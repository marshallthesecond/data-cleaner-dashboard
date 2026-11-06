# Data Wrangler and Visualizer

An interactive [Streamlit](https://streamlit.io/) web app for uploading, cleaning, exploring, visualizing, and exporting tabular datasets — all from the browser, with a full undo history and an exportable record of every transformation applied.

**Live demo:** https://data-wrangling-coursework.streamlit.app/

> _Tip: add a screenshot or short GIF of the app here — a `docs/screenshot.png` goes a long way on a portfolio project._

---

## Features

**Upload & overview**
- Load CSV, Excel (`.xlsx`), or JSON files
- Instant overview: shape, column names, data types, missing values, duplicate rows, and summary statistics

**Cleaning & preparation**
- Handle missing values: drop rows/columns, fill with mean / median / mode / most-frequent / a constant, or forward/backward fill
- Detect and remove duplicate rows (full-row or by a subset of columns)
- Column operations: drop, rename, create new columns from a formula, and bin numeric columns (equal-width or quantile)
- Convert and parse data types, including a robust numeric cleaner (currency symbols, thousands separators, accounting-style negatives) and datetime parsing
- Categorical tools: standardize case/whitespace, map or replace values, and group rare categories
- Numeric cleaning: IQR-based outlier detection, removal, and capping
- Normalization and scaling: Min-Max and Z-score
- Validation rules: numeric range, allowed categories, and non-null constraints
- **Undo** any step and **Reset** to the original upload at any time

**Visualization**
- Histogram, box plot, scatter, line, bar chart, and correlation heatmap
- Interactive filtering by categorical and numeric columns
- Optional aggregation and top-N selection for bar charts

**Export & report**
- Download the cleaned dataset as CSV or Excel
- Download a JSON **transformation report** and a **pipeline recipe** that record every step applied, so the cleaning process is reproducible

---

## Tech stack

Python · Streamlit · pandas · NumPy · Matplotlib · openpyxl

---

## Getting started

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd data-cleaner-dashboard

# 2. (Recommended) create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## How it works

1. Upload a dataset from the sidebar.
2. The app stores it in Streamlit's session state and keeps an untouched copy for Reset.
3. Use the **Cleaning and Prep** tab to transform the data. Every action is recorded and can be undone.
4. Use the **Visualization** tab to filter the data and build charts.
5. Use the **Export and Report** tab to download the cleaned data and a JSON log of everything you did.

---

## Project structure

```
.
├── app.py                      # Entry point: sidebar upload, session state, tab layout
├── requirements.txt
├── tabs/
│   ├── upload_and_overview.py  # Dataset overview
│   ├── cleaning_and_prep.py    # All cleaning and preparation tools
│   ├── visualization.py        # Filtering and chart builder
│   └── export_and_report.py    # CSV/Excel export and JSON report + recipe
├── datasets/                   # Sample datasets to try the app with
└── transformation report/      # Example exported report and recipe
```

---

## Notes

- Large datasets take longer to process; the app caches reads and expensive steps to stay responsive. The Excel export is built on demand (via the **Prepare Excel file** button) so large files don't slow down every interaction.
- Visualization tools adapt to column types automatically.
- Uploading a new file resets the session; avoid reloading the page mid-session, as that clears your progress.
