import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# PATHS AND PARAMETERS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed" / "csv"

OUTPUTS_DIR = PROJECT_DIR / "outputs"

CSV_DIR = OUTPUTS_DIR / "csv"
CSV_PROCESSED_DIR = CSV_DIR / "processed"
CSV_SUMMARY_DIR = CSV_DIR / "summary"
CSV_METADATA_DIR = CSV_DIR / "metadata"

TXT_DIR = OUTPUTS_DIR / "txt"

CHARTS_DIR = OUTPUTS_DIR / "charts"
FIRM_TS_CHARTS_DIR = CHARTS_DIR / "firm_side" / "time_series"
FIRM_CS_CHARTS_DIR = CHARTS_DIR / "firm_side" / "cross_section"
WORKER_TS_CHARTS_DIR = CHARTS_DIR / "worker_side" / "time_series"
WORKER_CS_CHARTS_DIR = CHARTS_DIR / "worker_side" / "cross_section"
COMPARATIVE_CHARTS_DIR = CHARTS_DIR / "comparative"

for path in [
    OUTPUTS_DIR,
    CSV_DIR,
    CSV_PROCESSED_DIR,
    CSV_SUMMARY_DIR,
    CSV_METADATA_DIR,
    TXT_DIR,
    CHARTS_DIR,
    FIRM_TS_CHARTS_DIR,
    FIRM_CS_CHARTS_DIR,
    WORKER_TS_CHARTS_DIR,
    WORKER_CS_CHARTS_DIR,
    COMPARATIVE_CHARTS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

TXT_FILE = TXT_DIR / "results_summary.txt"

FIRM_TS_FILE = "Table_Firm-side_Time-series.csv"
FIRM_CS_FILE = "Table_Firm-side_Cross-section.csv"
WORKER_TS_FILE = "Table_Worker-side_Time-series.csv"
WORKER_CS_FILE = "Table_Worker-side_Cross-section.csv"
METADATA_ACTIVE_FILE = "Table_Metadata_Active.csv"
METADATA_MATRIX_FILE = "Table_Metadata_Matrix.csv"

BENCHMARK_COUNTRIES = [
    "Canada",
    "France",
    "Germany",
    "Italy",
    "Japan",
    "United Kingdom",
    "United States",
]

MIN_OBS_REGRESSION = 8

# =========================================================
# HELPERS
# =========================================================

def write_to_txt(text: str) -> None:
    with open(TXT_FILE, "a", encoding="utf-8") as f:
        f.write(text)
        f.write("\n\n")

def reset_outputs() -> None:
    if TXT_FILE.exists():
        TXT_FILE.unlink()

    for folder in [
        CSV_PROCESSED_DIR,
        CSV_SUMMARY_DIR,
        CSV_METADATA_DIR,
        FIRM_TS_CHARTS_DIR,
        FIRM_CS_CHARTS_DIR,
        WORKER_TS_CHARTS_DIR,
        WORKER_CS_CHARTS_DIR,
        COMPARATIVE_CHARTS_DIR,
    ]:
        for file in folder.glob("*"):
            if file.is_file():
                file.unlink()

def load_csv(file_name: str) -> pd.DataFrame:
    file_path = DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    df.columns = [str(c).strip() for c in df.columns]

    cols_to_drop = [c for c in df.columns if c.startswith("Unnamed") or c.strip() == ""]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors="ignore")

    return df

def print_df_info(df: pd.DataFrame, name: str) -> None:
    text = (
        f"{name} - DATAFRAME INFO\n"
        f"Shape: {df.shape}\n"
        f"Columns: {', '.join(df.columns.tolist())}"
    )
    print(text)
    write_to_txt(text)

def report_missing_counts(df: pd.DataFrame, cols: list[str], title: str) -> None:
    missing_counts = df[cols].isna().sum().to_string()
    text = f"{title} - MISSING VALUES REPORT\n{missing_counts}"
    print(text)
    write_to_txt(text)

def report_sample_loss(df_before: pd.DataFrame, df_after: pd.DataFrame, title: str) -> None:
    rows_before = len(df_before)
    rows_after = len(df_after)
    dropped = rows_before - rows_after

    text = (
        f"{title} - SAMPLE SIZE REPORT\n"
        f"Initial rows: {rows_before}\n"
        f"Rows after cleaning/construction: {rows_after}\n"
        f"Rows dropped: {dropped}"
    )
    print(text)
    write_to_txt(text)

def report_country_coverage(df: pd.DataFrame, country_col: str, title: str) -> None:
    if df.empty:
        text = f"{title} - COUNTRY COVERAGE\nNo usable observations."
    else:
        coverage = df.groupby(country_col).size().sort_values(ascending=False).to_string()
        text = f"{title} - COUNTRY COVERAGE\n{coverage}"

    print(text)
    write_to_txt(text)

def print_and_save_corr(df: pd.DataFrame, cols: list[str], title: str) -> None:
    valid = df[cols].dropna()
    if valid.empty:
        text = f"{title}\nNo valid data available for correlation."
    else:
        corr = valid.corr()
        text = (
            "\n" + "=" * 100 + "\n"
            + f"{title}\n"
            + "=" * 100 + "\n"
            + corr.to_string()
        )

    print(text)
    write_to_txt(text)

def run_formula_ols(
    data: pd.DataFrame,
    formula: str,
    model_name: str,
    cluster_col: str | None = None,
    cov_type_default: str = "HC1",
):
    data_clean = data.dropna().copy()

    if data_clean.empty:
        raise ValueError(f"No valid observations available for model: {model_name}")

    if cluster_col is not None:
        model = smf.ols(formula, data=data_clean).fit(
            cov_type="cluster",
            cov_kwds={"groups": data_clean[cluster_col]},
        )
        se_text = f"clustered by {cluster_col}"
    else:
        model = smf.ols(formula, data=data_clean).fit(cov_type=cov_type_default)
        se_text = cov_type_default

    output = (
        "\n" + "=" * 100 + "\n"
        + f"{model_name}\n"
        + "=" * 100 + "\n"
        + f"Formula: {formula}\n"
        + f"Observations used: {len(data_clean)}\n"
        + f"Standard errors: {se_text}\n\n"
        + model.summary().as_text()
    )

    print(output)
    write_to_txt(output)

    return model, data_clean

def export_csv(df: pd.DataFrame, file_name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / file_name, index=False)

def safe_log(series: pd.Series) -> pd.Series:
    return np.where(series > 0, np.log(series), np.nan)

def safe_file_stub(text: str) -> str:
    return (
        str(text).lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("-", "_")
        .replace("[", "")
        .replace("]", "")
        .replace(".", "")
        .replace("'", "")
    )

def get_year_range(df: pd.DataFrame, year_col: str = "year") -> str:
    valid = df[[year_col]].dropna()
    if valid.empty:
        return "No data"

    start = int(valid[year_col].min())
    end = int(valid[year_col].max())

    if start == end:
        return f"{start}"
    return f"{start}-{end}"

def _auto_limits(series: pd.Series, pad_ratio: float = 0.05, min_pad: float = 0.25) -> tuple[float, float]:
    valid = series.dropna()
    if valid.empty:
        return (0.0, 1.0)
    vmin = float(valid.min())
    vmax = float(valid.max())

    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * pad_ratio, min_pad)
        return vmin - pad, vmax + pad

    pad = max((vmax - vmin) * pad_ratio, min_pad)
    return vmin - pad, vmax + pad

def build_country_index_from_level(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "country_name",
    year_col: str = "year",
) -> tuple[pd.Series, dict]:
    """
    Index for a single series within each country.
    Base year = first valid year available for that country.
    """
    result = pd.Series(index=df.index, dtype=float)
    base_years = {}

    for country, group in df.groupby(group_col):
        g = group.sort_values(year_col).copy()
        valid = g[[year_col, value_col]].dropna()

        if valid.empty:
            continue

        base_year = int(valid[year_col].iloc[0])
        base_value = valid[value_col].iloc[0]

        if pd.isna(base_value) or base_value == 0:
            continue

        result.loc[g.index] = 100.0 * g[value_col] / base_value
        base_years[country] = base_year

    return result, base_years

def build_pair_indices_country(
    df: pd.DataFrame,
    level_col: str,
    growth_col: str,
    group_col: str = "country_name",
    year_col: str = "year",
) -> tuple[pd.Series, pd.Series, dict]:
    """
    Build two indices for a country chart where:
    - first series is observed in levels
    - second series is reconstructed recursively from growth
    Base year = first year where BOTH comparisons can start consistently.
    """
    level_index = pd.Series(index=df.index, dtype=float)
    growth_index = pd.Series(index=df.index, dtype=float)
    base_years = {}

    for country, group in df.groupby(group_col):
        g = group.sort_values(year_col).copy()

        valid_level = g[[year_col, level_col]].dropna()
        valid_growth = g[[year_col, growth_col]].dropna()

        if valid_level.empty or valid_growth.empty:
            continue

        common_years = sorted(set(valid_level[year_col]).intersection(set(valid_growth[year_col])))
        if not common_years:
            continue

        base_year = int(common_years[0])
        base_row = g[g[year_col] == base_year]

        if base_row.empty:
            continue

        base_level = base_row[level_col].iloc[0]
        if pd.isna(base_level) or base_level == 0:
            continue

        base_years[country] = base_year

        # level index
        level_index.loc[g.index] = 100.0 * g[level_col] / base_level

        # recursive index
        base_idx = base_row.index[0]
        growth_index.loc[base_idx] = 100.0

        idx_list = list(g.index)
        base_pos = idx_list.index(base_idx)

        for pos in range(base_pos + 1, len(idx_list)):
            idx = idx_list[pos]
            prev_idx = idx_list[pos - 1]

            prev_val = growth_index.loc[prev_idx]
            growth = g.loc[idx, growth_col]

            if pd.notna(prev_val) and pd.notna(growth):
                growth_index.loc[idx] = prev_val * np.exp(growth)

    return level_index, growth_index, base_years

def rebase_single_series_common_year_across_countries(
    df: pd.DataFrame,
    value_col: str,
    country_col: str = "country_name",
    year_col: str = "year",
    countries: list[str] | None = None,
) -> tuple[pd.DataFrame, int | None]:
    """
    Rebase one series to a COMMON year across countries for comparative charts.
    Returns a filtered dataframe with a new column '<value_col>_common_index'.
    """
    work = df.copy()
    if countries is not None:
        work = work[work[country_col].isin(countries)].copy()

    year_sets = []
    for _, g in work.groupby(country_col):
        valid_years = set(g.loc[g[value_col].notna(), year_col].tolist())
        if valid_years:
            year_sets.append(valid_years)

    if not year_sets:
        work[f"{value_col}_common_index"] = np.nan
        return work, None

    common_years = sorted(set.intersection(*year_sets))
    if not common_years:
        work[f"{value_col}_common_index"] = np.nan
        return work, None

    base_year = int(common_years[0])
    result = pd.Series(index=work.index, dtype=float)

    for country, g in work.groupby(country_col):
        g = g.sort_values(year_col).copy()
        base_row = g[g[year_col] == base_year]
        if base_row.empty:
            continue

        base_value = base_row[value_col].iloc[0]
        if pd.isna(base_value) or base_value == 0:
            continue

        result.loc[g.index] = 100.0 * g[value_col] / base_value

    work[f"{value_col}_common_index"] = result
    return work, base_year

def save_line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_cols,
    labels,
    title: str,
    y_label: str,
    file_name: str,
    output_dir: Path,
    x_label: str = "Year",
    show_legend: bool = True,
    hlines=None,
) -> None:
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    if isinstance(labels, str):
        labels = [labels]

    valid_cols = [c for c in y_cols if c in df.columns]
    if not valid_cols:
        return

    plt.figure(figsize=(10, 5))

    plotted_cols = []
    for col, label in zip(y_cols, labels):
        if col not in df.columns:
            continue
        valid = df[[x_col, col]].dropna()
        if not valid.empty:
            plt.plot(valid[x_col], valid[col], label=label, linewidth=1.8)
            plotted_cols.append(col)

    if not plotted_cols:
        plt.close()
        return

    if hlines is not None:
        for h in hlines:
            plt.axhline(
                y=h["y"],
                linestyle=h.get("linestyle", "--"),
                linewidth=h.get("linewidth", 1.6),
                label=h.get("label"),
            )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    y_values = pd.concat([df[c].dropna() for c in plotted_cols], axis=0)
    if hlines is not None:
        y_values = pd.concat([y_values, pd.Series([h["y"] for h in hlines], dtype=float)], axis=0)

    ymin, ymax = _auto_limits(y_values)
    plt.ylim(ymin, ymax)

    if show_legend:
        plt.legend(frameon=False)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / file_name, dpi=300, bbox_inches="tight")
    plt.close()

def save_dual_axis_plot(
    df: pd.DataFrame,
    x_col: str,
    y1_col: str,
    y2_col: str,
    title: str,
    file_name: str,
    output_dir: Path,
    x_label: str = "Year",
    y1_label: str = "Labour share",
    y2_label: str = "Index",
) -> None:
    valid = df[[x_col, y1_col, y2_col]].dropna()
    if valid.empty:
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(valid[x_col], valid[y1_col], linewidth=1.8, label=y1_label)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y1_label)
    y1min, y1max = _auto_limits(valid[y1_col], pad_ratio=0.05, min_pad=0.01)
    ax1.set_ylim(y1min, y1max)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(valid[x_col], valid[y2_col], linewidth=1.8, linestyle="--", label=y2_label)
    ax2.set_ylabel(y2_label)
    y2min, y2max = _auto_limits(valid[y2_col], pad_ratio=0.05, min_pad=0.5)
    ax2.set_ylim(y2min, y2max)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False)

    plt.title(title)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / file_name, dpi=300, bbox_inches="tight")
    plt.close()

def save_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    file_name: str,
    output_dir: Path,
    fit_type: str | None = None,
    alpha: float = 0.75,
    label_col: str | None = None,
) -> None:
    cols_needed = [x_col, y_col]
    if label_col is not None:
        cols_needed.append(label_col)

    valid = df[cols_needed].dropna().copy()
    if valid.empty:
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(valid[x_col], valid[y_col], alpha=alpha)

    if label_col is not None:
        for _, row in valid.iterrows():
            plt.annotate(
                str(row[label_col]),
                (row[x_col], row[y_col]),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=8,
                alpha=0.85,
            )

    if fit_type is not None and len(valid) > 2:
        x = valid[x_col].astype(float).values
        y = valid[y_col].astype(float).values
        x_line = np.linspace(x.min(), x.max(), 400)

        if fit_type == "linear":
            coeffs = np.polyfit(x, y, 1)
            y_line = coeffs[0] * x_line + coeffs[1]
            fit_label = "Linear fit"

            debug_text = (
                f"{title} - linear fit coefficients:\n"
                f"slope = {coeffs[0]:.6f}, intercept = {coeffs[1]:.6f}"
            )
            print(debug_text)
            write_to_txt(debug_text)

            plt.plot(x_line, y_line, linewidth=2.0, label=fit_label)

        elif fit_type == "quadratic":
            coeffs = np.polyfit(x, y, 2)
            y_line = coeffs[0] * x_line**2 + coeffs[1] * x_line + coeffs[2]
            fit_label = "Quadratic fit"

            debug_text = (
                f"{title} - quadratic fit coefficients:\n"
                f"a = {coeffs[0]:.6f}, b = {coeffs[1]:.6f}, c = {coeffs[2]:.6f}"
            )
            print(debug_text)
            write_to_txt(debug_text)

            plt.plot(x_line, y_line, linewidth=2.0, label=fit_label)
        else:
            raise ValueError(f"Unsupported fit_type: {fit_type}")

        plt.legend(frameon=False)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    xmin, xmax = _auto_limits(valid[x_col], pad_ratio=0.05, min_pad=0.25)
    ymin, ymax = _auto_limits(valid[y_col], pad_ratio=0.05, min_pad=0.25)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / file_name, dpi=300, bbox_inches="tight")
    plt.close()

# =========================================================
# METADATA
# =========================================================

def run_metadata_check() -> tuple[pd.DataFrame, pd.DataFrame]:
    active_df = load_csv(METADATA_ACTIVE_FILE)
    matrix_df = load_csv(METADATA_MATRIX_FILE)

    print_df_info(active_df, "METADATA ACTIVE FILE")
    print_df_info(matrix_df, "METADATA MATRIX FILE")

    preview_cols = [c for c in [
        "MODULE",
        "PROJECT_ROLE",
        "DERIVED_VARIABLE",
        "CORE_MEASURE_NAMING_PYTHON",
    ] if c in active_df.columns]

    if preview_cols:
        preview = active_df[preview_cols].drop_duplicates().to_string(index=False)
        text = "METADATA ACTIVE PREVIEW\n" + preview
        print(text)
        write_to_txt(text)

    export_csv(active_df, "metadata_active_copy.csv", CSV_METADATA_DIR)
    export_csv(matrix_df, "metadata_matrix_copy.csv", CSV_METADATA_DIR)

    return active_df, matrix_df

# =========================================================
# MODULE 1 - FIRM-SIDE TIME SERIES
# =========================================================

def run_firm_side_time_series() -> pd.DataFrame:

    # ===================== LOAD =====================

    raw_df = load_csv(FIRM_TS_FILE)
    print_df_info(raw_df, "FIRM-SIDE TIME-SERIES FILE")

    required_cols = [
        "country_id",
        "country_name",
        "year",
        "gva_nom_nc",
        "lctot_nom_nc",
        "gva_ph_real_nc",
        "gva_ph_nom_nc",
        "lc_ph_nom_nc",
        "ulc_index",
        "ulc_nom_nc",
    ]

    missing = [c for c in required_cols if c not in raw_df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df = raw_df[required_cols].copy()
    report_missing_counts(df, required_cols, "MODULE 1")

    df_before = df.copy()
    df = df.sort_values(["country_name", "year"]).reset_index(drop=True)

    # ===================== LOGS =====================

    df["ln_gva_ph_real"] = safe_log(df["gva_ph_real_nc"])
    df["ln_gva_ph_nom"]  = safe_log(df["gva_ph_nom_nc"])
    df["ln_lc_ph_nom"]   = safe_log(df["lc_ph_nom_nc"])
    df["ln_ulc_nom"]     = safe_log(df["ulc_nom_nc"])

    # ===================== GROWTH =====================

    df["g_prod_real"] = df.groupby("country_name")["ln_gva_ph_real"].diff()
    df["g_prod_nom"]  = df.groupby("country_name")["ln_gva_ph_nom"].diff()

    df["g_p_output"]  = df["g_prod_nom"] - df["g_prod_real"]

    df["g_lc_nom"]    = df.groupby("country_name")["ln_lc_ph_nom"].diff()

    df["g_w_real_output"] = df["g_lc_nom"] - df["g_p_output"]

    # ===================== ULC =====================

    df["g_ulc_nom"] = df.groupby("country_name")["ln_ulc_nom"].diff()
    df["g_ulc_identity"] = df["g_lc_nom"] - df["g_prod_real"]
    df["g_ulc_gap_check"] = df["g_ulc_nom"] - df["g_ulc_identity"]

    # ===================== LABOUR SHARE =====================

    df["labour_share"] = np.where(
        (df["lctot_nom_nc"] > 0) & (df["gva_nom_nc"] > 0),
        df["lctot_nom_nc"] / df["gva_nom_nc"],
        np.nan,
    )

    df["ln_labour_share"] = safe_log(df["labour_share"])
    df["dln_labour_share"] = df.groupby("country_name")["ln_labour_share"].diff()

    # ===================== INDICES =====================

    df["ulc_index_rebased"], _ = build_country_index_from_level(df, "ulc_index")
    df["labour_share_index"], _ = build_country_index_from_level(df, "labour_share")

    df["prod_index"], df["w_real_output_index"], pair_base_years = build_pair_indices_country(
        df,
        level_col="gva_ph_real_nc",
        growth_col="g_w_real_output",
    )

    # ===================== GAP =====================

    df["gap_firm"] = np.where(
        (df["prod_index"] > 0) & (df["w_real_output_index"] > 0),
        np.log(df["prod_index"] / df["w_real_output_index"]),
        np.nan,
    )
    df.loc[~np.isfinite(df["gap_firm"]), "gap_firm"] = np.nan

    # ===================== REPORT =====================

    report_sample_loss(df_before, df, "MODULE 1")
    report_country_coverage(df, "country_name", "MODULE 1")

    write_to_txt("BASE YEARS:\n" + str(pair_base_years))

    consistency = df["g_ulc_gap_check"].dropna().describe().to_string()
    write_to_txt("ULC CONSISTENCY CHECK\n" + consistency)

    # ===================== CORRELATIONS =====================

    print_and_save_corr(
        df,
        ["g_prod_real", "g_w_real_output", "g_ulc_nom", "dln_labour_share"],
        "MODULE 1 CORRELATIONS",
    )

    # ===================== GAP vs LABOUR SHARE =====================

    df["gap_from_ls"] = -df["ln_labour_share"]
    corr_gap_ls = df[["gap_firm", "gap_from_ls"]].corr()
    write_to_txt("GAP vs LABOUR SHARE\n" + corr_gap_ls.to_string())

    # ===================== REGRESSIONS =====================

    # MODEL 1A
    model_df_1 = df[["country_name", "year", "g_w_real_output", "g_prod_real"]].dropna()
    if len(model_df_1) >= MIN_OBS_REGRESSION:
        run_formula_ols(
            data=model_df_1,
            formula="g_w_real_output ~ g_prod_real + C(country_name) + C(year)",
            model_name="MODEL 1A PASS-THROUGH",
            cluster_col="country_name",
        )

    # MODEL 1B (IDENTITY)
    model_df_2 = df[[
        "country_name", "year",
        "g_w_real_output", "g_prod_real", "dln_labour_share"
    ]].dropna()

    if len(model_df_2) >= MIN_OBS_REGRESSION:
        run_formula_ols(
            data=model_df_2,
            formula="g_w_real_output ~ g_prod_real + dln_labour_share + C(country_name) + C(year)",
            model_name="MODEL 1B IDENTITY",
            cluster_col="country_name",
        )

    write_to_txt("NOTE: MODEL 1B is accounting identity, not causal")

    # MODEL 1C
    model_df_3 = df[["country_name", "year", "g_ulc_nom", "g_prod_real"]].dropna()
    if len(model_df_3) >= MIN_OBS_REGRESSION:
        run_formula_ols(
            data=model_df_3,
            formula="g_ulc_nom ~ g_prod_real + C(country_name) + C(year)",
            model_name="MODEL 1C ULC",
            cluster_col="country_name",
        )

    # ===================== COUNTRY CHARTS =====================

    for country in BENCHMARK_COUNTRIES:

        tmp = df[df["country_name"] == country].copy()
        if tmp.empty:
            continue

        stub = safe_file_stub(country)
        time_range = get_year_range(tmp)

        scatter_tmp = tmp[["year", "g_prod_real", "g_w_real_output"]].dropna().copy()

        if len(scatter_tmp) >= 3:
            scatter_tmp["year_label"] = scatter_tmp["year"].astype(int).astype(str)

        save_line_plot(
            df=tmp,
            x_col="year",
            y_cols=["prod_index", "w_real_output_index"],
            labels=["Productivity", "Wage"],
            title=f"{country} indices ({time_range})",
            y_label="Index",
            file_name=f"firm_ts_indices_{stub}.png",
            output_dir=FIRM_TS_CHARTS_DIR,
        )

        save_line_plot(
            df=tmp,
            x_col="year",
            y_cols=["ulc_index_rebased"],
            labels=["ULC"],
            title=f"{country} ULC ({time_range})",
            y_label="Index",
            file_name=f"firm_ts_ulc_{stub}.png",
            output_dir=FIRM_TS_CHARTS_DIR,
            show_legend=False,
        )

        save_dual_axis_plot(
            df=tmp,
            x_col="year",
            y1_col="labour_share",
            y2_col="labour_share_index",
            title=f"{country} labour share ({time_range})",
            file_name=f"firm_ts_ls_{stub}.png",
            output_dir=FIRM_TS_CHARTS_DIR,
        )

        save_line_plot(
            df=tmp,
            x_col="year",
            y_cols=["gap_firm"],
            labels=["Gap"],
            title=f"{country} gap ({time_range})",
            y_label="log gap",
            file_name=f"firm_ts_gap_{stub}.png",
            output_dir=FIRM_TS_CHARTS_DIR,
            show_legend=False,
            hlines=[{"y": 0.0}],
        )

        save_scatter_plot(
            df=scatter_tmp,
            x_col="g_prod_real",
            y_col="g_w_real_output",
            title=f"Firm-side pass-through scatter: {country} ({time_range})",
            x_label="Δln(Productivity per hour, real)",
            y_label="Δln(Real product wage)",
            file_name=f"firm_ts_scatter_{stub}.png",
            output_dir=FIRM_TS_CHARTS_DIR,
            fit_type="linear",
            label_col="year_label",
        )

    # =========================================================
    # G7 COMPARATIVE CHARTS - FIRM SIDE TIME SERIES
    # =========================================================

    # --- 1) PRODUCTIVITY INDEX G7

    g7_prod, g7_prod_base = rebase_single_series_common_year_across_countries(
        df,
        "gva_ph_real_nc",
        countries=BENCHMARK_COUNTRIES
    )

    plt.figure(figsize=(10, 6))
    for country in BENCHMARK_COUNTRIES:
        tmp = g7_prod[
            g7_prod["country_name"] == country
        ].dropna(subset=["year", "gva_ph_real_nc_common_index"])

        if not tmp.empty:
            plt.plot(
                tmp["year"],
                tmp["gva_ph_real_nc_common_index"],
                linewidth=1.8,
                label=country
            )

    plt.title(f"Firm-side productivity index: G7 comparison (common base year = {g7_prod_base})")
    plt.xlabel("Year")
    plt.ylabel("Index")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        COMPARATIVE_CHARTS_DIR / "firm_ts_prod_index_g7.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --- 2) REAL PRODUCT WAGE INDEX G7

    plt.figure(figsize=(10, 6))
    for country in BENCHMARK_COUNTRIES:
        tmp = df[
            df["country_name"] == country
        ].dropna(subset=["year", "w_real_output_index"])

        if not tmp.empty:
            plt.plot(
                tmp["year"],
                tmp["w_real_output_index"],
                linewidth=1.8,
                label=country
            )

    plt.title("Firm-side real product wage index: G7 comparison")
    plt.xlabel("Year")
    plt.ylabel("Index")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        COMPARATIVE_CHARTS_DIR / "firm_ts_product_wage_index_g7.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --- 3) ULC INDEX G7

    g7_ulc, g7_ulc_base = rebase_single_series_common_year_across_countries(
        df,
        "ulc_index",
        countries=BENCHMARK_COUNTRIES
    )

    plt.figure(figsize=(10, 6))
    for country in BENCHMARK_COUNTRIES:
        tmp = g7_ulc[
            g7_ulc["country_name"] == country
        ].dropna(subset=["year", "ulc_index_common_index"])

        if not tmp.empty:
            plt.plot(
                tmp["year"],
                tmp["ulc_index_common_index"],
                linewidth=1.8,
                label=country
            )

    plt.title(f"Firm-side ULC index: G7 comparison (common base year = {g7_ulc_base})")
    plt.xlabel("Year")
    plt.ylabel("Index")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        COMPARATIVE_CHARTS_DIR / "firm_ts_ulc_index_g7.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --- 4) LABOUR SHARE INDEX G7

    g7_ls, g7_ls_base = rebase_single_series_common_year_across_countries(
        df,
        "labour_share",
        countries=BENCHMARK_COUNTRIES
    )

    plt.figure(figsize=(10, 6))
    for country in BENCHMARK_COUNTRIES:
        tmp = g7_ls[
            g7_ls["country_name"] == country
        ].dropna(subset=["year", "labour_share_common_index"])

        if not tmp.empty:
            plt.plot(
                tmp["year"],
                tmp["labour_share_common_index"],
                linewidth=1.8,
                label=country
            )

    plt.title(f"Firm-side labour share index: G7 comparison (common base year = {g7_ls_base})")
    plt.xlabel("Year")
    plt.ylabel("Index")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        COMPARATIVE_CHARTS_DIR / "firm_ts_labour_share_index_g7.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --- 5) PRODUCTIVITY-WAGE GAP G7

    plt.figure(figsize=(10, 6))
    for country in BENCHMARK_COUNTRIES:
        tmp = df[
            df["country_name"] == country
        ].dropna(subset=["year", "gap_firm"])

        if not tmp.empty:
            plt.plot(
                tmp["year"],
                tmp["gap_firm"],
                linewidth=1.8,
                label=country
            )

    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.title("Firm-side productivity-wage gap: G7 comparison")
    plt.xlabel("Year")
    plt.ylabel("ln(Productivity index / Real product wage index)")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        COMPARATIVE_CHARTS_DIR / "firm_ts_gap_g7.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # ===================== SAVE =====================

    export_csv(df, "firm_side_time_series_processed.csv", CSV_PROCESSED_DIR)

    return df

# =========================================================
# MODULE 2 - FIRM-SIDE CROSS SECTION
# =========================================================

def run_firm_side_cross_section() -> pd.DataFrame:
    raw_df = load_csv(FIRM_CS_FILE)
    print_df_info(raw_df, "FIRM-SIDE CROSS-SECTION FILE")

    required_cols = [
        "country_id",
        "country_name",
        "reference_year",
        "coe_nom_ppp",
        "gdp_fc_nom_ppp",
        "employment",
        "employees",
        "hours_total",
    ]
    missing = [c for c in required_cols if c not in raw_df.columns]
    if missing:
        raise KeyError(f"Missing columns in firm-side cross-section file: {missing}")

    df = raw_df[required_cols].copy()
    report_missing_counts(df, required_cols, "MODULE 2 - FIRM-SIDE CROSS SECTION")

    df_before = df.copy()

    df["correction_factor"] = np.where(
        (df["employment"] > 0) & (df["employees"] > 0),
        df["employment"] / df["employees"],
        np.nan,
    )

    df["prod_nom_ppp_per_hour"] = np.where(
        (df["gdp_fc_nom_ppp"] > 0) & (df["hours_total"] > 0),
        df["gdp_fc_nom_ppp"] / df["hours_total"],
        np.nan,
    )

    df["coe_nom_ppp_per_hour_raw"] = np.where(
        (df["coe_nom_ppp"] > 0) & (df["hours_total"] > 0),
        df["coe_nom_ppp"] / df["hours_total"],
        np.nan,
    )

    df["coe_nom_ppp_corrected"] = df["coe_nom_ppp"] * df["correction_factor"]

    df["coe_nom_ppp_per_hour_corrected"] = np.where(
        (df["coe_nom_ppp_corrected"] > 0) & (df["hours_total"] > 0),
        df["coe_nom_ppp_corrected"] / df["hours_total"],
        np.nan,
    )

    df["labour_share_emp_only"] = np.where(
        (df["coe_nom_ppp"] > 0) & (df["gdp_fc_nom_ppp"] > 0),
        df["coe_nom_ppp"] / df["gdp_fc_nom_ppp"],
        np.nan,
    )

    df["labour_share_corrected"] = np.where(
        (df["coe_nom_ppp_corrected"] > 0) & (df["gdp_fc_nom_ppp"] > 0),
        df["coe_nom_ppp_corrected"] / df["gdp_fc_nom_ppp"],
        np.nan,
    )

    df["ln_prod_nom_ppp_per_hour"] = safe_log(df["prod_nom_ppp_per_hour"])
    df["ln_coe_nom_ppp_per_hour_corrected"] = safe_log(df["coe_nom_ppp_per_hour_corrected"])

    report_sample_loss(df_before, df, "MODULE 2 - FIRM-SIDE CROSS SECTION")
    report_country_coverage(df, "country_name", "MODULE 2 - FIRM-SIDE CROSS SECTION")

    preview_cols = [
        "country_name",
        "reference_year",
        "prod_nom_ppp_per_hour",
        "coe_nom_ppp_per_hour_corrected",
        "labour_share_emp_only",
        "labour_share_corrected",
        "correction_factor",
    ]
    preview = df[preview_cols].to_string(index=False)
    print("\nMODULE 2 PREVIEW")
    print(preview)
    write_to_txt("MODULE 2 PREVIEW\n" + preview)

    print_and_save_corr(
        df,
        [
            "ln_prod_nom_ppp_per_hour",
            "ln_coe_nom_ppp_per_hour_corrected",
            "labour_share_emp_only",
            "labour_share_corrected",
        ],
        "MODULE 2 - FIRM-SIDE CROSS-SECTION CORRELATION MATRIX",
    )

    model_df = df[[
        "country_name",
        "ln_coe_nom_ppp_per_hour_corrected",
        "ln_prod_nom_ppp_per_hour",
    ]].dropna()
    if len(model_df) >= 3:
        run_formula_ols(
            data=model_df,
            formula="ln_coe_nom_ppp_per_hour_corrected ~ ln_prod_nom_ppp_per_hour",
            model_name="MODULE 2A - Firm-side cross-section: corrected labour compensation per hour on productivity per hour",
            cluster_col=None,
        )

    ref_year = int(df["reference_year"].mode().iloc[0]) if not df.empty else 0
    df["country_label"] = df["country_id"].astype(str)

    save_scatter_plot(
        df=df,
        x_col="ln_prod_nom_ppp_per_hour",
        y_col="ln_coe_nom_ppp_per_hour_corrected",
        title=f"Firm-side cross-section: productivity and labour compensation per hour ({ref_year})",
        x_label="ln(Productivity per hour, nominal PPP)",
        y_label="ln(Corrected labour compensation per hour, nominal PPP)",
        file_name="firm_cs_scatter_loglog_main.png",
        output_dir=FIRM_CS_CHARTS_DIR,
        fit_type="linear",
        label_col="country_label",
    )

    save_scatter_plot(
        df=df,
        x_col="prod_nom_ppp_per_hour",
        y_col="labour_share_corrected",
        title=f"Firm-side cross-section: productivity and corrected labour share ({ref_year})",
        x_label="Productivity per hour, nominal PPP",
        y_label="Corrected labour share",
        file_name="firm_cs_scatter_labour_share.png",
        output_dir=FIRM_CS_CHARTS_DIR,
        fit_type="linear",
        label_col="country_label",
    )

    cross_note = (
        "MODULE 2 - NOTE\n"
        "The cross-section is currently exploratory because the G7-only sample is too small.\n"
        "A stronger cross-sectional implementation should extend the sample, for example to a broader OECD-European set."
    )
    print(cross_note)
    write_to_txt(cross_note)

    export_csv(df, "firm_side_cross_section_processed.csv", CSV_PROCESSED_DIR)
    return df

# =========================================================
# MODULE 3 - WORKER-SIDE TIME SERIES
# =========================================================

def run_worker_side_time_series() -> pd.DataFrame:
    raw_df = load_csv(WORKER_TS_FILE)
    print_df_info(raw_df, "WORKER-SIDE TIME-SERIES FILE")

    required_cols = [
        "country_id",
        "country_name",
        "year",
        "gva_real_ppp",
        "employment",
        "employees",
        "wage_pe_fte_real_ppp",
    ]
    missing = [c for c in required_cols if c not in raw_df.columns]
    if missing:
        raise KeyError(f"Missing columns in worker-side time-series file: {missing}")

    df = raw_df[required_cols].copy()
    report_missing_counts(df, required_cols, "MODULE 3 - WORKER-SIDE TIME SERIES")

    df_before = df.copy()
    df = df.sort_values(["country_name", "year"]).reset_index(drop=True)
    df["prod_real_ppp_per_employee"] = np.where(
        (df["gva_real_ppp"] > 0) & (df["employees"] > 0),
        df["gva_real_ppp"] / df["employees"],
        np.nan,
    )

    df["prod_real_ppp_per_employed"] = np.where(
        (df["gva_real_ppp"] > 0) & (df["employment"] > 0),
        df["gva_real_ppp"] / df["employment"],
        np.nan,
    )

    df["employee_share_of_employment"] = np.where(
        (df["employment"] > 0) & (df["employees"] > 0),
        df["employees"] / df["employment"],
        np.nan,
    )

    df["ln_prod_real_ppp_per_employee"] = safe_log(df["prod_real_ppp_per_employee"])
    df["ln_prod_real_ppp_per_employed"] = safe_log(df["prod_real_ppp_per_employed"])
    df["ln_wage_real_ppp"] = safe_log(df["wage_pe_fte_real_ppp"])

    df["g_prod_employee"] = df.groupby("country_name")["ln_prod_real_ppp_per_employee"].diff()
    df["g_prod_employed"] = df.groupby("country_name")["ln_prod_real_ppp_per_employed"].diff()
    df["g_wage_real_ppp"] = df.groupby("country_name")["ln_wage_real_ppp"].diff()

    # Pair indices for country charts
    df["prod_employee_index"], df["wage_real_ppp_index"], base_years_employee = build_pair_indices_country(
        df,
        level_col="prod_real_ppp_per_employee",
        growth_col="g_wage_real_ppp",
    )

    df["prod_employed_index"], _, base_years_employed = build_pair_indices_country(
        df,
        level_col="prod_real_ppp_per_employed",
        growth_col="g_wage_real_ppp",
    )
    df["gap_worker_employee"] = np.where(
        (df["prod_employee_index"] > 0) & (df["wage_real_ppp_index"] > 0),
        np.log(df["prod_employee_index"] / df["wage_real_ppp_index"]),
        np.nan,
    )

    df["gap_worker_employed"] = np.where(
        (df["prod_employed_index"] > 0) & (df["wage_real_ppp_index"] > 0),
        np.log(df["prod_employed_index"] / df["wage_real_ppp_index"]),
        np.nan,
    )

    report_sample_loss(df_before, df, "MODULE 3 - WORKER-SIDE TIME SERIES")
    report_country_coverage(df, "country_name", "MODULE 3 - WORKER-SIDE TIME SERIES")

    base_year_text = (
        "MODULE 3 - BASE YEARS FOR PRODUCTIVITY VS REAL WAGES\n"
        + "Employee-based:\n"
        + "\n".join([f"{k}: {v}" for k, v in base_years_employee.items()])
        + "\n\nEmployed-based:\n"
        + "\n".join([f"{k}: {v}" for k, v in base_years_employed.items()])
    )
    print(base_year_text)
    write_to_txt(base_year_text)

    preview_cols = [
        "country_name",
        "year",
        "prod_real_ppp_per_employee",
        "prod_real_ppp_per_employed",
        "wage_pe_fte_real_ppp",
        "g_prod_employee",
        "g_prod_employed",
        "g_wage_real_ppp",
        "gap_worker_employee",
    ]
    preview = df[preview_cols].tail(30).to_string(index=False)
    print("\nMODULE 3 PREVIEW")
    print(preview)
    write_to_txt("MODULE 3 PREVIEW\n" + preview)

    print_and_save_corr(
        df,
        ["g_prod_employee", "g_prod_employed", "g_wage_real_ppp"],
        "MODULE 3 - WORKER-SIDE CORRELATION MATRIX",
    )

    model_df_1 = df[["country_name", "year", "g_wage_real_ppp", "g_prod_employee"]].dropna()
    if len(model_df_1) >= MIN_OBS_REGRESSION:
        run_formula_ols(
            data=model_df_1,
            formula="g_wage_real_ppp ~ g_prod_employee + C(country_name) + C(year)",
            model_name="MODULE 3A - Worker-side pass-through: real wage growth on productivity growth per employee",
            cluster_col="country_name",
        )

    model_df_2 = df[["country_name", "year", "g_wage_real_ppp", "g_prod_employed"]].dropna()
    if len(model_df_2) >= MIN_OBS_REGRESSION:
        run_formula_ols(
            data=model_df_2,
            formula="g_wage_real_ppp ~ g_prod_employed + C(country_name) + C(year)",
            model_name="MODULE 3B - Worker-side robustness: real wage growth on productivity growth per employed person",
            cluster_col="country_name",
        )

    robustness_note = (
        "MODULE 3 - NOTE\n"
        "Both labour-input definitions are retained. In practice, employee-based and employed-based\n"
        "results are treated as complementary specifications / robustness checks."
    )
    print(robustness_note)
    write_to_txt(robustness_note)

    # Country charts
    for country in BENCHMARK_COUNTRIES:
        tmp = df[df["country_name"] == country].copy()
        if tmp.empty:
            continue

        stub = safe_file_stub(country)
        time_range = get_year_range(tmp)

        save_line_plot(
            df=tmp,
            x_col="year",
            y_cols=["prod_employee_index", "wage_real_ppp_index"],
            labels=["Productivity per employee index", "Real wage PPP index"],
            title=f"Worker-side indices: {country} ({time_range})",
            y_label="Index (common start year within country)",
            file_name=f"worker_ts_indices_{stub}.png",
            output_dir=WORKER_TS_CHARTS_DIR,
        )

        save_line_plot(
            df=tmp,
            x_col="year",
            y_cols=["gap_worker_employee"],
            labels=["Worker-side gap"],
            title=f"Worker-side productivity-wage gap: {country} ({time_range})",
            y_label="ln(Productivity index / Real wage index)",
            file_name=f"worker_ts_gap_{stub}.png",
            output_dir=WORKER_TS_CHARTS_DIR,
            show_legend=False,
            hlines=[{"y": 0.0, "label": "Zero gap"}],
        )

        scatter_tmp = tmp[["year", "g_prod_employee", "g_wage_real_ppp"]].dropna().copy()
        if len(scatter_tmp) >= 3:
            scatter_tmp["year_label"] = scatter_tmp["year"].astype(int).astype(str)
            save_scatter_plot(
                df=scatter_tmp,
                x_col="g_prod_employee",
                y_col="g_wage_real_ppp",
                title=f"Worker-side pass-through scatter: {country} ({time_range})",
                x_label="Δln(Productivity per employee, real PPP)",
                y_label="Δln(Real wage PPP)",
                file_name=f"worker_ts_scatter_{stub}.png",
                output_dir=WORKER_TS_CHARTS_DIR,
                fit_type="linear",
                label_col="year_label",
            )

    # G7 comparative charts
    g7_prod_employee, g7_prod_emp_base = rebase_single_series_common_year_across_countries(
        df, "prod_real_ppp_per_employee", countries=BENCHMARK_COUNTRIES
    )
    g7_prod_employed, g7_prod_employed_base = rebase_single_series_common_year_across_countries(
        df, "prod_real_ppp_per_employed", countries=BENCHMARK_COUNTRIES
    )
    g7_wage, g7_wage_base = rebase_single_series_common_year_across_countries(
        df, "wage_pe_fte_real_ppp", countries=BENCHMARK_COUNTRIES
    )

    plt.figure(figsize=(10, 6))
    for country in BENCHMARK_COUNTRIES:
        tmp = df[df["country_name"] == country].dropna(subset=["year", "gap_worker_employee"])
        if not tmp.empty:
            plt.plot(tmp["year"], tmp["gap_worker_employee"], linewidth=1.8, label=country)
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.title("Worker-side productivity-wage gap: G7 comparison (employee-based)")
    plt.xlabel("Year")
    plt.ylabel("ln(Productivity index / Real wage index)")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(COMPARATIVE_CHARTS_DIR / "worker_ts_gap_g7_employee.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    for country in BENCHMARK_COUNTRIES:
        tmp = df[df["country_name"] == country].dropna(subset=["year", "gap_worker_employed"])
        if not tmp.empty:
            plt.plot(tmp["year"], tmp["gap_worker_employed"], linewidth=1.8, label=country)
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.title("Worker-side productivity-wage gap: G7 comparison (employed-based)")
    plt.xlabel("Year")
    plt.ylabel("ln(Productivity index / Real wage index)")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(COMPARATIVE_CHARTS_DIR / "worker_ts_gap_g7_employed.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    for country in BENCHMARK_COUNTRIES:
        tmp = g7_prod_employee[g7_prod_employee["country_name"] == country].dropna(subset=["year", "prod_real_ppp_per_employee_common_index"])
        if not tmp.empty:
            plt.plot(tmp["year"], tmp["prod_real_ppp_per_employee_common_index"], linewidth=1.8, label=country)
    plt.title(f"Worker-side productivity index: G7 comparison (employee-based, common base year = {g7_prod_emp_base})")
    plt.xlabel("Year")
    plt.ylabel("Index")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(COMPARATIVE_CHARTS_DIR / "worker_ts_prod_index_g7_employee.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    for country in BENCHMARK_COUNTRIES:
        tmp = g7_wage[g7_wage["country_name"] == country].dropna(subset=["year", "wage_pe_fte_real_ppp_common_index"])
        if not tmp.empty:
            plt.plot(tmp["year"], tmp["wage_pe_fte_real_ppp_common_index"], linewidth=1.8, label=country)
    plt.title(f"Worker-side real wage index: G7 comparison (common base year = {g7_wage_base})")
    plt.xlabel("Year")
    plt.ylabel("Index")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(COMPARATIVE_CHARTS_DIR / "worker_ts_wage_index_g7.png", dpi=300, bbox_inches="tight")
    plt.close()

    export_csv(df, "worker_side_time_series_processed.csv", CSV_PROCESSED_DIR)
    return df





# =========================================================
# MODULE 4 - WORKER-SIDE CROSS SECTION
# =========================================================

def run_worker_side_cross_section() -> pd.DataFrame:
    raw_df = load_csv(WORKER_CS_FILE)
    print_df_info(raw_df, "WORKER-SIDE CROSS-SECTION FILE")

    required_cols = [
        "country_id",
        "country_name",
        "reference_year",
        "gva_real_ppp",
        "employment",
        "employees",
        "wage_pe_fte_real_ppp",
    ]
    missing = [c for c in required_cols if c not in raw_df.columns]
    if missing:
        raise KeyError(f"Missing columns in worker-side cross-section file: {missing}")

    df = raw_df[required_cols].copy()
    report_missing_counts(df, required_cols, "MODULE 4 - WORKER-SIDE CROSS SECTION")

    df_before = df.copy()

    df["prod_real_ppp_per_employee"] = np.where(
        (df["gva_real_ppp"] > 0) & (df["employees"] > 0),
        df["gva_real_ppp"] / df["employees"],
        np.nan,
    )

    df["prod_real_ppp_per_employed"] = np.where(
        (df["gva_real_ppp"] > 0) & (df["employment"] > 0),
        df["gva_real_ppp"] / df["employment"],
        np.nan,
    )

    df["ln_prod_real_ppp_per_employee"] = safe_log(df["prod_real_ppp_per_employee"])
    df["ln_prod_real_ppp_per_employed"] = safe_log(df["prod_real_ppp_per_employed"])
    df["ln_wage_real_ppp"] = safe_log(df["wage_pe_fte_real_ppp"])
    df["country_label"] = df["country_id"].astype(str)

    report_sample_loss(df_before, df, "MODULE 4 - WORKER-SIDE CROSS SECTION")
    report_country_coverage(df, "country_name", "MODULE 4 - WORKER-SIDE CROSS SECTION")

    preview_cols = [
        "country_name",
        "reference_year",
        "prod_real_ppp_per_employee",
        "prod_real_ppp_per_employed",
        "wage_pe_fte_real_ppp",
    ]
    preview = df[preview_cols].to_string(index=False)
    print("\nMODULE 4 PREVIEW")
    print(preview)
    write_to_txt("MODULE 4 PREVIEW\n" + preview)

    print_and_save_corr(
        df,
        ["ln_prod_real_ppp_per_employee", "ln_prod_real_ppp_per_employed", "ln_wage_real_ppp"],
        "MODULE 4 - WORKER-SIDE CROSS-SECTION CORRELATION MATRIX",
    )

    model_df_1 = df[[
        "country_name",
        "ln_wage_real_ppp",
        "ln_prod_real_ppp_per_employee",
    ]].dropna()
    if len(model_df_1) >= 3:
        run_formula_ols(
            data=model_df_1,
            formula="ln_wage_real_ppp ~ ln_prod_real_ppp_per_employee",
            model_name="MODULE 4A - Worker-side cross-section: real wage PPP on productivity per employee",
            cluster_col=None,
        )

    model_df_2 = df[[
        "country_name",
        "ln_wage_real_ppp",
        "ln_prod_real_ppp_per_employed",
    ]].dropna()
    if len(model_df_2) >= 3:
        run_formula_ols(
            data=model_df_2,
            formula="ln_wage_real_ppp ~ ln_prod_real_ppp_per_employed",
            model_name="MODULE 4B - Worker-side cross-section robustness: real wage PPP on productivity per employed person",
            cluster_col=None,
        )

    ref_year = int(df["reference_year"].mode().iloc[0]) if not df.empty else 0

    save_scatter_plot(
        df=df,
        x_col="ln_prod_real_ppp_per_employee",
        y_col="ln_wage_real_ppp",
        title=f"Worker-side cross-section: productivity and real wages ({ref_year})",
        x_label="ln(Productivity per employee, real PPP)",
        y_label="ln(Real wage PPP)",
        file_name="worker_cs_scatter_loglog_employee.png",
        output_dir=WORKER_CS_CHARTS_DIR,
        fit_type="linear",
        label_col="country_label",
    )

    save_scatter_plot(
        df=df,
        x_col="ln_prod_real_ppp_per_employed",
        y_col="ln_wage_real_ppp",
        title=f"Worker-side cross-section robustness: productivity per employed person and real wages ({ref_year})",
        x_label="ln(Productivity per employed person, real PPP)",
        y_label="ln(Real wage PPP)",
        file_name="worker_cs_scatter_loglog_employed.png",
        output_dir=WORKER_CS_CHARTS_DIR,
        fit_type="linear",
        label_col="country_label",
    )

    cross_note = (
        "MODULE 4 - NOTE\n"
        "The worker-side cross-section is exploratory given the very small G7-only sample.\n"
        "A more informative implementation should expand the cross-sectional sample."
    )
    print(cross_note)
    write_to_txt(cross_note)

    export_csv(df, "worker_side_cross_section_processed.csv", CSV_PROCESSED_DIR)
    return df

# =========================================================
# SUMMARY TABLES
# =========================================================

def _cum_growth_from_index(df: pd.DataFrame, index_col: str) -> float:
    valid = df[[index_col]].dropna()
    if len(valid) < 2:
        return np.nan
    return float(np.log(valid[index_col].iloc[-1] / valid[index_col].iloc[0]))

def build_summary_tables(
    firm_ts_df: pd.DataFrame,
    firm_cs_df: pd.DataFrame,
    worker_ts_df: pd.DataFrame,
    worker_cs_df: pd.DataFrame,
) -> None:
    firm_latest = (
        firm_ts_df.sort_values(["country_name", "year"])
        .groupby("country_name")
        .tail(1)
        .copy()
    )
    worker_latest = (
        worker_ts_df.sort_values(["country_name", "year"])
        .groupby("country_name")
        .tail(1)
        .copy()
    )


    latest_summary = (
        firm_latest[[
            "country_name",
            "year",
            "prod_index",
            "w_real_output_index",
            "ulc_index_rebased",
            "labour_share",
            "labour_share_index",
            "gap_firm",
        ]]
        .merge(
            worker_latest[[
                "country_name",
                "year",
                "prod_employee_index",
                "prod_employed_index",
                "wage_real_ppp_index",
                "gap_worker_employee",
                "gap_worker_employed",
            ]],
            on=["country_name", "year"],
            how="outer",
        )
        .sort_values("country_name")
    )

    latest_text = "SUMMARY - LATEST TIME-SERIES SNAPSHOT\n" + latest_summary.to_string(index=False)
    print(latest_text)
    write_to_txt(latest_text)
    export_csv(latest_summary, "summary_latest_time_series_snapshot.csv", CSV_SUMMARY_DIR)

    cumulative_rows = []

    for country in BENCHMARK_COUNTRIES:
        row = {"country_name": country}

        firm = firm_ts_df[firm_ts_df["country_name"] == country].sort_values("year")
        worker = worker_ts_df[worker_ts_df["country_name"] == country].sort_values("year")

        row["firm_prod_cum_growth_log"] = _cum_growth_from_index(firm, "prod_index")
        row["firm_wage_cum_growth_log"] = _cum_growth_from_index(firm, "w_real_output_index")
        row["firm_ulc_cum_growth_log"] = _cum_growth_from_index(firm, "ulc_index_rebased")
        row["firm_labour_share_cum_growth_log"] = _cum_growth_from_index(firm, "labour_share_index")

        row["worker_prod_employee_cum_growth_log"] = _cum_growth_from_index(worker, "prod_employee_index")
        row["worker_prod_employed_cum_growth_log"] = _cum_growth_from_index(worker, "prod_employed_index")
        row["worker_wage_cum_growth_log"] = _cum_growth_from_index(worker, "wage_real_ppp_index")

        if pd.notna(row["firm_prod_cum_growth_log"]) and pd.notna(row["firm_wage_cum_growth_log"]):
            row["firm_cum_gap"] = row["firm_prod_cum_growth_log"] - row["firm_wage_cum_growth_log"]
        else:
            row["firm_cum_gap"] = np.nan

        if pd.notna(row["worker_prod_employee_cum_growth_log"]) and pd.notna(row["worker_wage_cum_growth_log"]):
            row["worker_cum_gap_employee"] = row["worker_prod_employee_cum_growth_log"] - row["worker_wage_cum_growth_log"]
        else:
            row["worker_cum_gap_employee"] = np.nan

        if pd.notna(row["worker_prod_employed_cum_growth_log"]) and pd.notna(row["worker_wage_cum_growth_log"]):
            row["worker_cum_gap_employed"] = row["worker_prod_employed_cum_growth_log"] - row["worker_wage_cum_growth_log"]
        else:
            row["worker_cum_gap_employed"] = np.nan

        cumulative_rows.append(row)

    cumulative_summary = pd.DataFrame(cumulative_rows).sort_values("country_name")
    cumulative_text = "SUMMARY - CUMULATIVE GROWTH TABLE\n" + cumulative_summary.to_string(index=False)
    print(cumulative_text)
    write_to_txt(cumulative_text)
    export_csv(cumulative_summary, "summary_cumulative_growth.csv", CSV_SUMMARY_DIR)

    cross_summary = (
        firm_cs_df[[
            "country_name",
            "reference_year",
            "prod_nom_ppp_per_hour",
            "coe_nom_ppp_per_hour_corrected",
            "labour_share_corrected",
        ]]
        .merge(
            worker_cs_df[[
                "country_name",
                "reference_year",
                "prod_real_ppp_per_employee",
                "prod_real_ppp_per_employed",
                "wage_pe_fte_real_ppp",
            ]],
            on=["country_name", "reference_year"],
            how="outer",
        )
        .sort_values("country_name")
    )

    cross_text = "SUMMARY - CROSS-SECTION SNAPSHOT\n" + cross_summary.to_string(index=False)
    print(cross_text)
    write_to_txt(cross_text)
    export_csv(cross_summary, "summary_cross_section.csv", CSV_SUMMARY_DIR)

# =========================================================
# MAIN
# =========================================================

def main() -> None:
    reset_outputs()

    intro = (
        "Running Productivity - Labour Compensation - Labour Share - ULC project.\n\n"
        "Project structure:\n"
        "1. Firm-side time series\n"
        "2. Firm-side cross section\n"
        "3. Worker-side time series\n"
        "4. Worker-side cross section\n\n"
        "Key methodological choices:\n"
        "- No sectoral/economic-activity extension in the baseline version.\n"
        "- Firm-side module includes pass-through, labour share, and unit labour cost.\n"
        "- Worker-side module focuses on real PPP wages and productivity, without labour share or ULC.\n"
        "- Worker-side productivity is estimated both per employee and per employed person.\n"
        "- Cross-section charts use log-log specification where appropriate.\n"
        "- Scatter plots use productivity on the x-axis and labour compensation / wages on the y-axis.\n"
        "- Time-series scatter plots retain the fitted line and do not connect dots chronologically.\n"
        "- Labour share is analysed in the firm-side framework, including a descriptive cross-section scatter.\n"
        "- Country charts use a common start year across the series being compared.\n"
        "- Comparative charts across countries use a common base year across countries for the same series.\n"
    )

    print(intro)
    write_to_txt(intro)

    metadata_active_df, metadata_matrix_df = run_metadata_check()
    firm_ts_df = run_firm_side_time_series()
    firm_cs_df = run_firm_side_cross_section()
    worker_ts_df = run_worker_side_time_series()
    worker_cs_df = run_worker_side_cross_section()
    build_summary_tables(firm_ts_df, firm_cs_df, worker_ts_df, worker_cs_df)

    final_message = (
        "\nAll analyses completed successfully.\n"
        f"Outputs saved in: {OUTPUTS_DIR}"
    )
    print(final_message)
    write_to_txt(final_message)

if __name__ == "__main__":
    main()
