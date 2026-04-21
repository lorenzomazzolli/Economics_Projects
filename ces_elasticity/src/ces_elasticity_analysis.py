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

TXT_FILE = OUTPUTS_DIR / "results_summary" / "txt" / "results_summary.txt"

def create_output_structure():
    paths = [
        # charts
        OUTPUTS_DIR / "charts" / "model1" / "full_sample",
        OUTPUTS_DIR / "charts" / "model1" / "trimmed_sample",
        OUTPUTS_DIR / "charts" / "model2" / "scatter",
        OUTPUTS_DIR / "charts" / "model2" / "lines",
        OUTPUTS_DIR / "charts" / "model3" / "scatter",
        OUTPUTS_DIR / "charts" / "model3" / "lines",

        # results
        OUTPUTS_DIR / "results_summary" / "txt",
        OUTPUTS_DIR / "results_summary" / "tables" / "model1",
        OUTPUTS_DIR / "results_summary" / "tables" / "model2",
        OUTPUTS_DIR / "results_summary" / "tables" / "model3",
    ]

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

LEVEL_FILE = "Table_CES_Total_Economy_Level.csv"
GROWTH_FILE = "Table_CES_Total_Economy_Growth.csv"
SECTOR_FILE = "Table_CES_by_Economic_Activity_Growth.csv"

BENCHMARK_COUNTRIES = [
    "Canada",
    "France",
    "Germany",
    "Italy",
    "Japan",
    "United Kingdom",
    "United States"
]

SECTOR_FOCUS = ["_T", "C", "BTE", "BTNXL", "GTNXL"]
MIN_OBS_SECTOR_CHARTS = 6

SECTOR_LABELS = {
    "_T": "Total Economy",
    "C": "Manufacturing",
    "BTE": "Industry (excl. construction)",
    "BTNXL": "Business Economy",
    "GTNXL": "Business Services"
}

# =========================================================
# HELPERS
# =========================================================

def write_to_txt(text: str) -> None:
    with open(TXT_FILE, "a", encoding="utf-8") as f:
        f.write(text)
        f.write("\n\n")

def load_csv(file_name: str) -> pd.DataFrame:
    file_path = DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(file_path, sep=";", decimal=",", encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    cols_to_drop = [c for c in df.columns if c.startswith("Unnamed") or c.strip() == ""]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors="ignore")

    if "Back to Index" in df.columns:
        df = df.drop(columns=["Back to Index"], errors="ignore")

    df = df.dropna(axis=1, how="all")
    return df

def print_df_info(df: pd.DataFrame, name: str) -> None:
    text = (
        f"{name} - DATAFRAME INFO\n"
        f"Shape: {df.shape}\n"
        f"Columns: {', '.join(df.columns.tolist())}"
    )
    print(text)
    write_to_txt(text)

def report_sample_loss(df_before: pd.DataFrame, df_after: pd.DataFrame, model_name: str) -> None:
    rows_before = len(df_before)
    rows_after = len(df_after)
    dropped = rows_before - rows_after

    text = (
        f"{model_name} - SAMPLE SIZE REPORT\n"
        f"Initial rows: {rows_before}\n"
        f"Rows after cleaning/construction: {rows_after}\n"
        f"Rows dropped: {dropped}"
    )
    print(text)
    write_to_txt(text)

def report_country_coverage(df: pd.DataFrame, country_col: str, model_name: str) -> None:
    if df.empty:
        text = f"{model_name} - COUNTRY COVERAGE\nNo usable observations."
    else:
        coverage = df.groupby(country_col).size().sort_values(ascending=False).to_string()
        text = f"{model_name} - COUNTRY COVERAGE\n{coverage}"
    print(text)
    write_to_txt(text)

def report_group_coverage(df: pd.DataFrame, group_cols: list[str], model_name: str, top_n: int = 50) -> None:
    if df.empty:
        text = f"{model_name} - GROUP COVERAGE\nNo usable observations."
    else:
        coverage = (
            df.groupby(group_cols)
            .size()
            .sort_values(ascending=False)
            .head(top_n)
            .to_string()
        )
        text = f"{model_name} - TOP {top_n} GROUPS BY USABLE OBSERVATIONS\n{coverage}"
    print(text)
    write_to_txt(text)

def report_missing_counts(df: pd.DataFrame, cols: list[str], model_name: str) -> None:
    missing_counts = df[cols].isna().sum().to_string()
    text = f"{model_name} - MISSING VALUES REPORT\n{missing_counts}"
    print(text)
    write_to_txt(text)
def compute_sigma(beta: float) -> float:
    denom = beta + 1
    if pd.isna(beta) or abs(denom) < 1e-12:
        return np.nan
    return 1 / denom

def interpret_sigma(sigma: float) -> str:
    if pd.isna(sigma):
        return "sigma not defined"
    if np.isclose(sigma, 1.0, atol=0.10):
        return "approximately Cobb-Douglas (sigma ≈ 1)"
    if sigma < 1:
        return "capital and labour appear relatively complementary (sigma < 1)"
    return "capital and labour appear relatively substitutable (sigma > 1)"

def summarize_sigma(model, slope_name: str, model_name: str) -> None:
    if slope_name not in model.params.index:
        text = f"{model_name}: slope '{slope_name}' not found in regression output."
        print(text)
        write_to_txt(text)
        return

    beta = model.params[slope_name]
    sigma = compute_sigma(beta)

    text = (
        f"{model_name} - IMPLIED CES ELASTICITY\n"
        f"beta = {beta:.6f}\n"
        f"sigma = 1 / (beta + 1) = {sigma:.6f}\n"
        f"Interpretation: {interpret_sigma(sigma)}"
    )
    print(text)
    write_to_txt(text)

def run_formula_ols(
    data: pd.DataFrame,
    formula: str,
    model_name: str,
    cluster_col: str | None = None,
    cov_type_default: str = "HC1"
):
    data_clean = data.dropna().copy()

    if data_clean.empty:
        raise ValueError(f"No valid observations available for model: {model_name}")

    if cluster_col is not None:
        model = smf.ols(formula, data=data_clean).fit(
            cov_type="cluster",
            cov_kwds={"groups": data_clean[cluster_col]}
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
    )

def get_time_range(df: pd.DataFrame, year_col: str = "Year") -> str:
    valid = df[[year_col]].dropna()
    if valid.empty:
        return "No data"

    start = int(valid[year_col].min())
    end = int(valid[year_col].max())

    if start == end:
        return f"{start}"
    return f"{start}-{end}"

def save_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    file_name: str,
    fit_type: str | None = None,
    alpha: float = 0.7,
    label_col: str | None = None,
    max_labels: int | None = None,
    label_fontsize: int = 8
) -> None:
    cols_needed = [x_col, y_col]
    if label_col is not None:
        cols_needed.append(label_col)

    valid = df[cols_needed].dropna().copy()

    if valid.empty:
        return

    if label_col is not None:
        valid[label_col] = valid[label_col].astype(str)

    plt.figure(figsize=(8, 6))
    plt.scatter(valid[x_col], valid[y_col], alpha=alpha)

    if label_col is not None:
        if max_labels is not None and len(valid) > max_labels:
            label_df = valid.sample(n=max_labels, random_state=42)
        else:
            label_df = valid

        for _, row in label_df.iterrows():
            plt.annotate(
                row[label_col],
                (row[x_col], row[y_col]),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=label_fontsize,
                alpha=0.85
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
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.close()

def save_line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    file_name: str
) -> None:
    valid = df[[x_col, y_col]].dropna().copy()

    if valid.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(valid[x_col], valid[y_col], linewidth=1.8)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.close()

def get_influence_table(model, df_used: pd.DataFrame, label_col: str) -> pd.DataFrame:
    influence = model.get_influence()

    table = df_used.copy()
    table["fitted"] = model.fittedvalues
    table["residual"] = model.resid
    table["studentized_residual"] = influence.resid_studentized_external
    table["leverage"] = influence.hat_matrix_diag
    table["cooks_d"] = influence.cooks_distance[0]

    cols = [label_col, "fitted", "residual", "studentized_residual", "leverage", "cooks_d"]
    other_cols = [c for c in ["x", "y", "kh_ratio", "Labour_Share_alcd2"] if c in table.columns]
    return table[cols + other_cols].copy()

def compare_models(model_full, model_trimmed, slope_name: str, model_name: str) -> None:
    beta_full = model_full.params.get(slope_name, np.nan)
    beta_trim = model_trimmed.params.get(slope_name, np.nan)

    sigma_full = compute_sigma(beta_full)
    sigma_trim = compute_sigma(beta_trim)

    text = (
        f"{model_name} - FULL VS TRIMMED COMPARISON\n"
        f"Full sample beta = {beta_full:.6f}\n"
        f"Full sample sigma = {sigma_full:.6f}\n"
        f"Full sample R-squared = {model_full.rsquared:.6f}\n\n"
        f"Trimmed sample beta = {beta_trim:.6f}\n"
        f"Trimmed sample sigma = {sigma_trim:.6f}\n"
        f"Trimmed sample R-squared = {model_trimmed.rsquared:.6f}"
    )
    print(text)
    write_to_txt(text)

def export_regression_table(model, file_path: Path) -> None:
    """
    Export full regression output as structured table.
    """
    df = pd.DataFrame({
        "variable": model.params.index,
        "coefficient": model.params.values,
        "std_error": model.bse.values,
        "t_or_z_stat": model.tvalues.values,
        "p_value": model.pvalues.values
    })
    df.to_csv(file_path, index=False)


def export_model_summary(
    model,
    slope_name: str,
    model_name: str,
    sample_name: str,
    file_path: Path
) -> None:
    """
    Export compact model summary with beta, sigma, fit statistics and sample size.
    """
    beta = model.params.get(slope_name, np.nan)
    sigma = compute_sigma(beta)

    df = pd.DataFrame([{
        "model": model_name,
        "sample": sample_name,
        "slope_variable": slope_name,
        "beta": beta,
        "sigma": sigma,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "p_value": model.pvalues.get(slope_name, np.nan),
        "observations": int(model.nobs)
    }])
    df.to_csv(file_path, index=False)


def export_dataset(df: pd.DataFrame, file_path: Path) -> None:
    """
    Export cleaned dataset / regression sample.
    """
    df.to_csv(file_path, index=False)


def export_influence_table(df: pd.DataFrame, file_path: Path) -> None:
    """
    Export Model 1 influence diagnostics.
    """
    df.to_csv(file_path, index=False)


def export_elasticities_summary(records: list[dict], file_path: Path) -> None:
    pd.DataFrame(records).to_csv(file_path, index=False)

def clean_sector_label(sector_code: str) -> str:
    mapping = {
        "_T": "total_economy",
        "C": "manufacturing",
        "BTE": "industry",
        "BTNXL": "business_economy",
        "GTNXL": "business_services"
    }
    return mapping.get(sector_code, sector_code.lower())

# =========================================================
# MODEL 1: CROSS-COUNTRY LEVELS (AMECO)
# =========================================================

def run_ces_level_model() -> pd.DataFrame:
    raw_df = load_csv(LEVEL_FILE)
    print_df_info(raw_df, "LEVEL FILE")

    required_cols = [
        "Year",
        "Reference_Area_Code",
        "Reference_Area",
        "Labour_Share_alcd2",
        "Capital_Stock_oknd",
        "Hours_nlht"
    ]
    missing = [c for c in required_cols if c not in raw_df.columns]
    if missing:
        raise KeyError(f"Missing columns in level file: {missing}")

    df = raw_df[required_cols].copy()
    report_missing_counts(df, required_cols, "MODEL 1 - LEVEL")

    df_before = df.copy()

    df["sL"] = df["Labour_Share_alcd2"] / 100.0
    df["sK"] = 1.0 - df["sL"]

    df = df[
        df["sL"].notna()
        & df["sK"].notna()
        & (df["sL"] > 0)
        & (df["sK"] > 0)
        & (df["Capital_Stock_oknd"] > 0)
        & (df["Hours_nlht"] > 0)
    ].copy()

    df["kh_ratio"] = df["Capital_Stock_oknd"] / df["Hours_nlht"]
    df["x"] = np.log(df["kh_ratio"])
    df["y"] = np.log(df["sL"] / df["sK"])

    report_sample_loss(df_before, df, "MODEL 1 - LEVEL")
    report_country_coverage(df, "Reference_Area", "MODEL 1 - LEVEL")

    level_year = int(df["Year"].mode().iloc[0]) if not df.empty else ""

    preview = df[[
        "Reference_Area",
        "Year",
        "Labour_Share_alcd2",
        "Capital_Stock_oknd",
        "Hours_nlht",
        "kh_ratio",
        "x",
        "y"
    ]].sort_values("Reference_Area").to_string(index=False)

    print("\nLEVEL MODEL PREVIEW")
    print(preview)
    write_to_txt("LEVEL MODEL PREVIEW\n" + preview)

    label_check_y = df[["Reference_Area", "Reference_Area_Code", "x", "y", "kh_ratio", "Labour_Share_alcd2"]].sort_values("y")
    label_check_x = df[["Reference_Area", "Reference_Area_Code", "x", "y", "kh_ratio", "Labour_Share_alcd2"]].sort_values("x")

    print("\nMODEL 1 LABEL CHECK - SORTED BY Y")
    print(label_check_y.to_string(index=False))
    write_to_txt("MODEL 1 LABEL CHECK - SORTED BY Y\n" + label_check_y.to_string(index=False))

    print("\nMODEL 1 LABEL CHECK - SORTED BY X")
    print(label_check_x.to_string(index=False))
    write_to_txt("MODEL 1 LABEL CHECK - SORTED BY X\n" + label_check_x.to_string(index=False))

    print_and_save_corr(
        df,
        ["x", "y"],
        "MODEL 1 - CES LEVEL CORRELATION: x = ln(K/H), y = ln(sL/sK)"
    )

    model_full, df_used_full = run_formula_ols(
        data=df[["Reference_Area", "Reference_Area_Code", "y", "x", "kh_ratio", "Labour_Share_alcd2"]],
        formula="y ~ x",
        model_name="MODEL 1A - CES Level Model (Full Sample): y ~ x",
        cluster_col=None
    )

    summarize_sigma(model_full, "x", "MODEL 1A - CES Level Model (Full Sample)")

    export_regression_table(
        model_full,
        OUTPUTS_DIR / "results_summary" / "tables" / "model1" / "model1_full_regression.csv"
    )

    export_model_summary(
        model_full,
        slope_name="x",
        model_name="model1",
        sample_name="full",
        file_path=OUTPUTS_DIR / "results_summary" / "tables" / "model1" / "model1_full_summary.csv"
    )

    influence_table = get_influence_table(model_full, df_used_full, "Reference_Area")
    influence_table_sorted = influence_table.sort_values("cooks_d", ascending=False)
    export_influence_table(
        influence_table_sorted,
        OUTPUTS_DIR / "results_summary" / "tables" / "model1" / "model1_influence_diagnostics.csv"
    )

    influence_text = (
        "MODEL 1 - INFLUENCE DIAGNOSTICS (SORTED BY COOK'S DISTANCE)\n"
        + influence_table_sorted.to_string(index=False)
    )
    print(influence_text)
    write_to_txt(influence_text)

    n_full = len(df_used_full)
    cooks_threshold = 4 / n_full if n_full > 0 else np.nan

    cooks_text = (
        f"MODEL 1 - COOK'S DISTANCE THRESHOLD\n"
        f"Threshold = 4 / n = {cooks_threshold:.6f}"
    )
    print(cooks_text)
    write_to_txt(cooks_text)

    influential_df = influence_table_sorted[influence_table_sorted["cooks_d"] > cooks_threshold].copy()

    if influential_df.empty:
        influential_text = "MODEL 1 - INFLUENTIAL OBSERVATIONS\nNo observations exceed Cook's distance threshold."
    else:
        influential_text = (
            "MODEL 1 - INFLUENTIAL OBSERVATIONS\n"
            + influential_df.to_string(index=False)
        )
    print(influential_text)
    write_to_txt(influential_text)

    outlier_countries = influential_df["Reference_Area"].tolist()

    model_trimmed = None

    if len(outlier_countries) > 0:
        df_trimmed = df[~df["Reference_Area"].isin(outlier_countries)].copy()

        trim_text = (
            "MODEL 1B - TRIMMED SAMPLE\n"
            f"Excluded countries based on Cook's distance > 4/n:\n"
            + ", ".join(outlier_countries)
        )
        print(trim_text)
        write_to_txt(trim_text)

        model_trimmed, df_used_trimmed = run_formula_ols(
            data=df_trimmed[["Reference_Area", "Reference_Area_Code", "y", "x", "kh_ratio", "Labour_Share_alcd2"]],
            formula="y ~ x",
            model_name="MODEL 1B - CES Level Model (Without Influential Outliers): y ~ x",
            cluster_col=None
        )

        summarize_sigma(model_trimmed, "x", "MODEL 1B - CES Level Model (Without Influential Outliers)")

        export_regression_table(
            model_trimmed,
            OUTPUTS_DIR / "results_summary" / "tables" / "model1" / "model1_trimmed_regression.csv"
        )

        export_model_summary(
            model_trimmed,
            slope_name="x",
            model_name="model1",
            sample_name="trimmed",
            file_path=OUTPUTS_DIR / "results_summary" / "tables" / "model1" / "model1_trimmed_summary.csv"
        )

        compare_models(model_full, model_trimmed, "x", "MODEL 1")

    else:
        df_trimmed = df.copy()

    export_dataset(
        df,
        OUTPUTS_DIR / "results_summary" / "tables" / "model1" / "model1_dataset.csv"
    )

    if model_trimmed is not None:
        export_dataset(
            df_trimmed,
            OUTPUTS_DIR / "results_summary" / "tables" / "model1" / "model1_trimmed_dataset.csv"
        )

    # PLOTS
    df["country_label"] = df["Reference_Area_Code"].astype(str)

    save_scatter_plot(
        df=df,
        x_col="x",
        y_col="y",
        title=f"Model 1A: Cross-country CES Relationship ({level_year})",
        x_label="ln(K/H)",
        y_label="ln(sL/sK)",
        file_name=OUTPUTS_DIR / "charts" / "model1" / "full_sample" / "model1a_loglog.png",
        fit_type="linear",
        alpha=0.85,
        label_col="country_label",
        max_labels=None,
        label_fontsize=8
    )

    save_scatter_plot(
        df=df,
        x_col="kh_ratio",
        y_col="Labour_Share_alcd2",
        title=f"Model 1A: Capital Intensity and Labour Share ({level_year})",
        x_label="Capital Stock / Hours Worked",
        y_label="Labour Share ALCD2 (%)",
        file_name=OUTPUTS_DIR / "charts" / "model1" / "full_sample" / "model1a_levels.png",
        fit_type=None,
        alpha=0.85,
        label_col="country_label",
        max_labels=None,
        label_fontsize=8
    )

    if len(outlier_countries) > 0:
        df_trimmed["country_label"] = df_trimmed["Reference_Area_Code"].astype(str)

        save_scatter_plot(
            df=df_trimmed,
            x_col="x",
            y_col="y",
            title=f"Model 1B: CES Relationship Without Influential Outliers ({level_year})",
            x_label="ln(K/H)",
            y_label="ln(sL/sK)",
            file_name=OUTPUTS_DIR / "charts" / "model1" / "trimmed_sample" / "model1b_loglog.png",
            fit_type="linear",
            alpha=0.85,
            label_col="country_label",
            max_labels=None,
            label_fontsize=8
        )

        save_scatter_plot(
            df=df_trimmed,
            x_col="kh_ratio",
            y_col="Labour_Share_alcd2",
            title=f"Model 1B: Capital Intensity and Labour Share Without Influential Outliers ({level_year})",
            x_label="Capital Stock / Hours Worked",
            y_label="Labour Share ALCD2 (%)",
            file_name=OUTPUTS_DIR / "charts" / "model1" / "trimmed_sample" / "model1b_levels.png",
            fit_type=None,
            alpha=0.85,
            label_col="country_label",
            max_labels=None,
            label_fontsize=8
        )

    return df

# =========================================================
# MODEL 2: TOTAL ECONOMY GROWTH PANEL (OECD)
# =========================================================

def run_ces_total_growth_model() -> pd.DataFrame:
    raw_df = load_csv(GROWTH_FILE)
    print_df_info(raw_df, "TOTAL GROWTH FILE")

    required_cols = [
        "Year",
        "Reference_Area_Code",
        "Reference_Area",
        "Gross_Value_Added_gva",
        "Total_Labour_Compensation_lctot",
        "Capital_Deepening_Growth_kserhrs",
        "Capital_Services_Growth_kser",
        "Hours_hrsto"
    ]
    missing = [c for c in required_cols if c not in raw_df.columns]
    if missing:
        raise KeyError(f"Missing columns in total growth file: {missing}")

    df = raw_df[required_cols].copy()
    report_missing_counts(df, required_cols, "MODEL 2 - TOTAL GROWTH")

    df_before = df.copy()
    df = df.sort_values(["Reference_Area", "Year"]).reset_index(drop=True)

    df["sL"] = df["Total_Labour_Compensation_lctot"] / df["Gross_Value_Added_gva"]
    df["sK"] = 1.0 - df["sL"]

    df = df[
        df["sL"].notna()
        & df["sK"].notna()
        & (df["sL"] > 0)
        & (df["sK"] > 0)
        & (df["Gross_Value_Added_gva"] > 0)
        & (df["Total_Labour_Compensation_lctot"] > 0)
        & (df["Hours_hrsto"] > 0)
    ].copy()

    df["log_share_ratio"] = np.log(df["sL"] / df["sK"])
    df["dy"] = df.groupby("Reference_Area")["log_share_ratio"].diff()

    df["gK"] = df["Capital_Services_Growth_kser"] / 100.0
    df["gH"] = df.groupby("Reference_Area")["Hours_hrsto"].transform(lambda s: np.log(s).diff())
    df["dx"] = df["gK"] - df["gH"]

    df["dx_oecd_direct"] = df["Capital_Deepening_Growth_kserhrs"] / 100.0
    df["dx_gap_check"] = df["dx"] - df["dx_oecd_direct"]

    report_sample_loss(df_before, df, "MODEL 2 - TOTAL GROWTH")
    report_country_coverage(df, "Reference_Area", "MODEL 2 - TOTAL GROWTH")

    consistency = df["dx_gap_check"].dropna().describe().to_string()
    consistency_text = (
        "MODEL 2 - TOTAL GROWTH CONSISTENCY CHECK\n"
        "dx = gK - gH (constructed in Python)\n"
        "dx_oecd_direct = Capital_Deepening_Growth_kserhrs / 100\n\n"
        + consistency
    )
    print(consistency_text)
    write_to_txt(consistency_text)

    preview = df[[
        "Reference_Area",
        "Year",
        "sL",
        "log_share_ratio",
        "dy",
        "gK",
        "gH",
        "dx",
        "dx_oecd_direct",
        "dx_gap_check"
    ]].tail(30).to_string(index=False)

    print("\nTOTAL ECONOMY GROWTH PREVIEW")
    print(preview)
    write_to_txt("TOTAL ECONOMY GROWTH PREVIEW\n" + preview)

    print_and_save_corr(
        df,
        ["dx", "dy"],
        "MODEL 2 - CES TOTAL ECONOMY GROWTH CORRELATION: dx vs dy"
    )

    model_df = df[["dy", "dx", "Reference_Area", "Year"]].dropna().copy()

    model, used_df = run_formula_ols(
        data=model_df,
        formula="dy ~ dx + C(Reference_Area) + C(Year)",
        model_name="MODEL 2 - CES Total Economy Growth: dy ~ dx + country FE + year FE",
        cluster_col="Reference_Area"
    )

    export_dataset(
        df,
        OUTPUTS_DIR / "results_summary" / "tables" / "model2" / "panel_dataset_model2.csv"
    )

    export_dataset(
        used_df,
        OUTPUTS_DIR / "results_summary" / "tables" / "model2" / "model2_regression_sample.csv"
    )

    export_regression_table(
        model,
        OUTPUTS_DIR / "results_summary" / "tables" / "model2" / "model2_regression.csv"
    )

    export_model_summary(
        model,
        slope_name="dx",
        model_name="model2_total_growth",
        sample_name="panel_fe",
        file_path=OUTPUTS_DIR / "results_summary" / "tables" / "model2" / "model2_summary.csv"
    )

    summarize_sigma(model, "dx", "MODEL 2 - CES Total Economy Growth")
    report_country_coverage(used_df, "Reference_Area", "MODEL 2 - REGRESSION SAMPLE COVERAGE")

    # Scatter by country
    for country in BENCHMARK_COUNTRIES:
        tmp = df[df["Reference_Area"] == country].dropna(subset=["dx", "dy"]).copy()
        if len(tmp) >= 3:
            stub = safe_file_stub(country)
            time_range = get_time_range(tmp)

            tmp["Year_label"] = tmp["Year"].astype(int).astype(str)

            save_scatter_plot(
                df=tmp,
                x_col="dx",
                y_col="dy",
                title=f"Model 2: {country} ({time_range})",
                x_label="Δln(K/H)",
                y_label="Δln(sL/sK)",
                file_name=OUTPUTS_DIR / "charts" / "model2" / "scatter" / f"model2_scatter_{stub}.png",
                fit_type=None,
                alpha=0.85,
                label_col="Year_label",
                max_labels=None,
                label_fontsize=8
            )

    # Labour share line chart by country
    for country in BENCHMARK_COUNTRIES:
        tmp = df[df["Reference_Area"] == country].dropna(subset=["Year", "sL"]).copy()
        if len(tmp) >= 3:
            stub = safe_file_stub(country)
            time_range = get_time_range(tmp)

            save_line_plot(
                df=tmp,
                x_col="Year",
                y_col="sL",
                title=f"Model 2 Labour Share: {country} ({time_range})",
                x_label="Year",
                y_label="Labour Share (LC/GVA)",
                file_name=OUTPUTS_DIR / "charts" / "model2" / "lines" / f"model2_labour_share_{stub}.png"
            )

    return df

# =========================================================
# MODEL 3: SECTORAL GROWTH PANEL (OECD)
# =========================================================

def run_ces_sectoral_growth_model() -> pd.DataFrame:
    raw_df = load_csv(SECTOR_FILE)
    print_df_info(raw_df, "SECTORAL GROWTH FILE")

    required_cols = [
        "Year",
        "Ref_Area_Code",
        "Reference_Area",
        "Economic_Activity_Code",
        "Economic_Activity",
        "Capital_Deepening_Growth_kserhrs",
        "Capital_Services_Growth_kser",
        "Gross_Value_Added_gva",
        "Hours_hrsto",
        "Total_Labour_Compensation_lctot"
    ]
    missing = [c for c in required_cols if c not in raw_df.columns]
    if missing:
        raise KeyError(f"Missing columns in sectoral growth file: {missing}")

    df = raw_df[required_cols].copy()
    report_missing_counts(df, required_cols, "MODEL 3 - SECTORAL GROWTH")

    sector_list = (
        df[["Economic_Activity_Code", "Economic_Activity"]]
        .drop_duplicates()
        .sort_values(["Economic_Activity_Code", "Economic_Activity"])
        .to_string(index=False)
    )
    sector_text = "MODEL 3 - AVAILABLE SECTORS\n" + sector_list
    print(sector_text)
    write_to_txt(sector_text)

    df_before = df.copy()
    df = df.sort_values(["Reference_Area", "Economic_Activity", "Year"]).reset_index(drop=True)

    df["sL"] = df["Total_Labour_Compensation_lctot"] / df["Gross_Value_Added_gva"]
    df["sK"] = 1.0 - df["sL"]

    df = df[
        df["sL"].notna()
        & df["sK"].notna()
        & (df["sL"] > 0)
        & (df["sK"] > 0)
        & (df["Gross_Value_Added_gva"] > 0)
        & (df["Total_Labour_Compensation_lctot"] > 0)
        & (df["Hours_hrsto"] > 0)
    ].copy()

    df["log_share_ratio"] = np.log(df["sL"] / df["sK"])
    df["dy"] = df.groupby(["Reference_Area", "Economic_Activity"])["log_share_ratio"].diff()

    df["gK"] = df["Capital_Services_Growth_kser"] / 100.0
    df["gH"] = df.groupby(["Reference_Area", "Economic_Activity"])["Hours_hrsto"].transform(
        lambda s: np.log(s).diff()
    )

    df["dx"] = df["gK"] - df["gH"]
    df["dx_oecd_direct"] = df["Capital_Deepening_Growth_kserhrs"] / 100.0
    df["dx_gap_check"] = df["dx"] - df["dx_oecd_direct"]

    report_sample_loss(df_before, df, "MODEL 3 - SECTORAL GROWTH")
    report_country_coverage(df, "Reference_Area", "MODEL 3 - SECTORAL GROWTH COUNTRY COVERAGE")
    report_group_coverage(
        df,
        ["Reference_Area", "Economic_Activity_Code"],
        "MODEL 3 - SECTORAL GROWTH COUNTRY-SECTOR COVERAGE",
        top_n=100
    )

    consistency = df["dx_gap_check"].dropna().describe().to_string()
    consistency_text = (
        "MODEL 3 - SECTORAL GROWTH CONSISTENCY CHECK\n"
        "dx = gK - gH (constructed in Python)\n"
        "dx_oecd_direct = Capital_Deepening_Growth_kserhrs / 100\n\n"
        + consistency
    )
    print(consistency_text)
    write_to_txt(consistency_text)

    preview = df[[
        "Reference_Area",
        "Economic_Activity_Code",
        "Economic_Activity",
        "Year",
        "sL",
        "log_share_ratio",
        "dy",
        "gK",
        "gH",
        "dx",
        "dx_oecd_direct",
        "dx_gap_check"
    ]].tail(40).to_string(index=False)

    print("\nSECTORAL GROWTH PREVIEW")
    print(preview)
    write_to_txt("SECTORAL GROWTH PREVIEW\n" + preview)

    print_and_save_corr(
        df,
        ["dx", "dy"],
        "MODEL 3 - CES SECTORAL GROWTH CORRELATION: dx vs dy"
    )

    df["country_sector"] = df["Reference_Area"] + " | " + df["Economic_Activity_Code"]

    model_df = df[["dy", "dx", "country_sector", "Year"]].dropna().copy()

    model, used_df = run_formula_ols(
        data=model_df,
        formula="dy ~ dx + C(country_sector) + C(Year)",
        model_name="MODEL 3 - CES Sectoral Growth: dy ~ dx + country-sector FE + year FE",
        cluster_col="country_sector"
    )

    model3_results = []

    export_dataset(
        df,
        OUTPUTS_DIR / "results_summary" / "tables" / "model3" / "panel_dataset_model3.csv"
    )

    export_dataset(
        used_df,
        OUTPUTS_DIR / "results_summary" / "tables" / "model3" / "model3_regression_sample.csv"
    )

    export_regression_table(
        model,
        OUTPUTS_DIR / "results_summary" / "tables" / "model3" / "model3_regression.csv"
    )

    export_model_summary(
        model,
        slope_name="dx",
        model_name="model3_sectoral_growth",
        sample_name="panel_fe",
        file_path=OUTPUTS_DIR / "results_summary" / "tables" / "model3" / "model3_summary.csv"
    )

    summarize_sigma(model, "dx", "MODEL 3 - CES Sectoral Growth")
    report_group_coverage(
        used_df,
        ["country_sector"],
        "MODEL 3 - REGRESSION SAMPLE COVERAGE",
        top_n=100
    )

    chart_df = df[df["Economic_Activity_Code"].isin(SECTOR_FOCUS)].copy()

    sector_focus_text = (
        "MODEL 3 - SECTOR FOCUS USED FOR CHARTS\n"
        + ", ".join(SECTOR_FOCUS)
    )
    print(sector_focus_text)
    write_to_txt(sector_focus_text)

    # Scatter by country and selected sector
    for country in BENCHMARK_COUNTRIES:
        for sector_code in SECTOR_FOCUS:
            tmp = chart_df[
                (chart_df["Reference_Area"] == country) &
                (chart_df["Economic_Activity_Code"] == sector_code)
            ].dropna(subset=["dx", "dy"]).copy()

            if len(tmp) >= MIN_OBS_SECTOR_CHARTS:
                country_stub = safe_file_stub(country)
                sector_stub = clean_sector_label(sector_code)

                sector_name_raw = tmp["Economic_Activity"].iloc[0]
                sector_label = SECTOR_LABELS.get(sector_code, sector_name_raw)
                time_range = get_time_range(tmp)

                tmp["Year_label"] = tmp["Year"].astype(int).astype(str)

                save_scatter_plot(
                    df=tmp,
                    x_col="dx",
                    y_col="dy",
                    title=f"Model 3: {country} - {sector_label} [{sector_code}] ({time_range})",
                    x_label="Δln(K/H)",
                    y_label="Δln(sL/sK)",
                    file_name=OUTPUTS_DIR / "charts" / "model3" / "scatter" / f"model3_scatter_{country_stub}_{sector_stub}.png",
                    fit_type=None,
                    alpha=0.85,
                    label_col="Year_label",
                    max_labels=None,
                    label_fontsize=8
                )

    # Labour share line chart by country and selected sector
    for country in BENCHMARK_COUNTRIES:
        for sector_code in SECTOR_FOCUS:
            tmp = chart_df[
                (chart_df["Reference_Area"] == country) &
                (chart_df["Economic_Activity_Code"] == sector_code)
            ].dropna(subset=["Year", "sL"]).copy()

            if len(tmp) >= MIN_OBS_SECTOR_CHARTS:
                country_stub = safe_file_stub(country)
                sector_stub = clean_sector_label(sector_code)

                sector_name_raw = tmp["Economic_Activity"].iloc[0]
                sector_label = SECTOR_LABELS.get(sector_code, sector_name_raw)
                time_range = get_time_range(tmp)

                save_line_plot(
                    df=tmp,
                    x_col="Year",
                    y_col="sL",
                    title=f"Model 3 Labour Share: {country} - {sector_label} [{sector_code}] ({time_range})",
                    x_label="Year",
                    y_label="Labour Share (LC/GVA)",
                    file_name=OUTPUTS_DIR / "charts" / "model3" / "lines" / f"model3_labour_share_{country_stub}_{sector_stub}.png"
                )

    return df

# =========================================================
# MAIN
# =========================================================

def main() -> None:
    print("Running CES Elasticity of Substitution project...\n")

    for file in OUTPUTS_DIR.rglob("*.txt"):
        file.unlink()

    for file in OUTPUTS_DIR.rglob("*.png"):
        file.unlink()

    intro = (
        "Running CES Elasticity of Substitution project.\n"
        "Methodological structure:\n"
        "1. Cross-country levels (AMECO)\n"
        "2. Total economy growth panel (OECD)\n"
        "3. Sectoral growth panel (OECD)\n\n"
        "Missing-data treatment:\n"
        "- Missing observations are excluded automatically from transformations, regressions and charts.\n"
        "- Growth models are unbalanced panels by construction when data gaps exist.\n\n"
        "Chart design:\n"
        "- Model 1A: full-sample scatter plots on level specification variables, with country-code labels.\n"
        "- Model 1B: trimmed-sample scatter plots excluding influential outliers based on Cook's distance.\n"
        "- Model 2: scatter dx vs dy by country, with selected year labels and country-specific time range.\n"
        "- Model 2: labour-share line charts by country.\n"
        "- Model 3: scatter dx vs dy by country and selected macro-sector, with selected year labels and country-sector time range.\n"
        "- Model 3: labour-share line charts by country and selected macro-sector.\n"
    )

    create_output_structure()

    def clean_legacy_csv():
        files_to_remove = [
            OUTPUTS_DIR / "results_summary" / "tables" / "model1" / "full.csv",
            OUTPUTS_DIR / "results_summary" / "tables" / "model1" / "trimmed.csv",
            OUTPUTS_DIR / "results_summary" / "tables" / "model2" / "results.csv",
            OUTPUTS_DIR / "results_summary" / "tables" / "model3" / "results.csv",
        ]

        for file in files_to_remove:
            if file.exists():
                file.unlink()

    clean_legacy_csv()

    print(intro)
    write_to_txt(intro)

    run_ces_level_model()
    run_ces_total_growth_model()
    run_ces_sectoral_growth_model()

    elasticity_records = []

    summary_files = [
        OUTPUTS_DIR / "results_summary" / "tables" / "model1" / "model1_full_summary.csv",
        OUTPUTS_DIR / "results_summary" / "tables" / "model1" / "model1_trimmed_summary.csv",
        OUTPUTS_DIR / "results_summary" / "tables" / "model2" / "model2_summary.csv",
        OUTPUTS_DIR / "results_summary" / "tables" / "model3" / "model3_summary.csv",
    ]

    for file_path in summary_files:
        if file_path.exists():
            tmp = pd.read_csv(file_path)
            elasticity_records.extend(tmp.to_dict(orient="records"))

    if elasticity_records:
        export_elasticities_summary(
            elasticity_records,
            OUTPUTS_DIR / "results_summary" / "tables" / "elasticities_summary.csv"
        )

    final_message = (
        "\nAll CES analyses completed successfully.\n"
        f"Outputs saved in: {OUTPUTS_DIR}"
    )
    print(final_message)
    write_to_txt(final_message)

if __name__ == "__main__":
    main()
