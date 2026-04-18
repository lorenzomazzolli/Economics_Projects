import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =========================================================
# PATHS AND PARAMETERS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed" / "csv"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

TXT_FILE = OUTPUTS_DIR / "results_summary.txt"

# Taylor (1993) benchmark calibration
R_STAR = 2.0
PI_STAR = 2.0
PHI_PI = 0.5
PHI_Y = 0.5
PHI_U = 0.5

# =========================================================
# HELPERS
# =========================================================

def write_to_txt(text: str) -> None:
    with open(TXT_FILE, "a", encoding="utf-8") as f:
        f.write(text)
        f.write("\n\n")


def load_csv(file_name: str) -> pd.DataFrame:
    """
    Load CSV using project-specific formatting:
    - separator = ;
    - decimal = ,
    - parse Date
    """
    file_path = DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=";",
        decimal=",",
        parse_dates=["Date"]
    )
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def run_ols(y: pd.Series, X: pd.DataFrame, model_name: str):
    """
    Run OLS with constant, print summary, and save it to txt.
    """
    data = pd.concat([y, X], axis=1).dropna()
    y_clean = data.iloc[:, 0]
    X_clean = data.iloc[:, 1:]

    X_clean = sm.add_constant(X_clean)
    model = sm.OLS(y_clean, X_clean).fit()

    output = (
        "\n" + "=" * 90 + "\n"
        + model_name + "\n"
        + "=" * 90 + "\n"
        + model.summary().as_text()
    )

    print(output)
    write_to_txt(output)

    return model


def save_line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_cols,
    labels,
    title: str,
    y_label: str,
    file_name: str,
    x_label: str = "Date",
    y_limits=None,
    show_legend: bool = True,
    hlines=None
) -> None:
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    if isinstance(labels, str):
        labels = [labels]

    plt.figure(figsize=(10, 5))

    for col, label in zip(y_cols, labels):
        plt.plot(df[x_col], df[col], label=label, linewidth=1.8)

    if hlines is not None:
        for h in hlines:
            plt.axhline(
                y=h["y"],
                linestyle=h.get("linestyle", "--"),
                linewidth=h.get("linewidth", 1.8),
                label=h.get("label")
            )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if y_limits == "auto":
        values = pd.concat([df[col] for col in y_cols], axis=0).dropna()

        if hlines is not None:
            h_values = [h["y"] for h in hlines]
            values = pd.concat([values, pd.Series(h_values)], axis=0)

        if len(values) > 0:
            y_min = values.min()
            y_max = values.max()
            margin = max((y_max - y_min) * 0.05, 0.25)
            plt.ylim(y_min - margin, y_max + margin)

    elif y_limits is not None:
        plt.ylim(y_limits)

    if show_legend:
        plt.legend(frameon=False)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / file_name, dpi=300, bbox_inches="tight")
    plt.close()


def save_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    file_name: str,
    fit_type: str = "linear",
    alpha: float = 0.7,
    x_limits=None,
    y_limits=None,
    show_legend: bool = True
) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(df[x_col], df[y_col], alpha=alpha, label="Data")

    valid = df[[x_col, y_col]].dropna()

    if len(valid) > 2:
        x = valid[x_col].astype(float)
        y = valid[y_col].astype(float)
        x_line = np.linspace(x.min(), x.max(), 400)

        if fit_type == "linear":
            coeffs = np.polyfit(x, y, 1)
            y_line = coeffs[0] * x_line + coeffs[1]

            debug_text = (
                f"{title} - linear fit coefficients:\n"
                f"slope = {coeffs[0]:.6f}, intercept = {coeffs[1]:.6f}"
            )
            print(debug_text)
            write_to_txt(debug_text)

            plt.plot(
                x_line,
                y_line,
                linewidth=2.5,
                label="Linear fit"
            )

        elif fit_type == "quadratic":
            coeffs = np.polyfit(x, y, 2)
            y_line = coeffs[0] * x_line**2 + coeffs[1] * x_line + coeffs[2]

            debug_text = (
                f"{title} - quadratic fit coefficients:\n"
                f"a = {coeffs[0]:.6f}, b = {coeffs[1]:.6f}, c = {coeffs[2]:.6f}"
            )
            print(debug_text)
            write_to_txt(debug_text)

            plt.plot(
                x_line,
                y_line,
                linewidth=2.5,
                label="Quadratic fit"
            )

        else:
            raise ValueError(f"Unsupported fit_type: {fit_type}")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if x_limits is not None:
        plt.xlim(x_limits)

    if y_limits == "auto":
        values = valid[y_col]
        if len(values) > 0:
            y_min = values.min()
            y_max = values.max()
            margin = max((y_max - y_min) * 0.05, 0.25)
            plt.ylim(y_min - margin, y_max + margin)

    elif y_limits is not None:
        plt.ylim(y_limits)

    if show_legend:
        plt.legend(frameon=False)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / file_name, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# SAHM RULE
# Indicator framework, not regression
# =========================================================

def run_sahm_rule() -> pd.DataFrame:
    df = load_csv("Table_Sahm's_Rule.csv")

    # Remove rows with missing unemployment before computing rolling stats
    df = df[df["U_RATE"].notna()].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # Sahm Rule:
    # 3-month moving average of unemployment minus minimum over previous 12 months
    df["U_RATE_MA3"] = df["U_RATE"].rolling(window=3, min_periods=3).mean()
    df["U_RATE_MA3_MIN_12"] = df["U_RATE_MA3"].rolling(window=12, min_periods=12).min()
    df["SAHM_RULE"] = df["U_RATE_MA3"] - df["U_RATE_MA3_MIN_12"]
    df["RECESSION_SIGNAL"] = pd.Series(pd.NA, index=df.index, dtype="boolean")

    valid_mask = df["SAHM_RULE"].notna()
    df.loc[valid_mask, "RECESSION_SIGNAL"] = df.loc[valid_mask, "SAHM_RULE"] >= 0.5

    sahm_preview = (
        df[["Date", "U_RATE", "U_RATE_MA3", "U_RATE_MA3_MIN_12", "SAHM_RULE", "RECESSION_SIGNAL"]]
        .tail(15)
        .to_string(index=False)
    )

    latest_valid = df.loc[df["SAHM_RULE"].notna()].iloc[-1]
    sahm_comment = (
        f"Latest valid Sahm Rule observation: {latest_valid['Date'].date()} | "
        f"SAHM_RULE = {latest_valid['SAHM_RULE']:.3f} | "
        f"Signal = {bool(latest_valid['RECESSION_SIGNAL'])}"
    )

    sahm_max = df["SAHM_RULE"].max(skipna=True)
    sahm_max_date = df.loc[df["SAHM_RULE"].idxmax(), "Date"] if df["SAHM_RULE"].notna().any() else None
    sahm_peak_comment = (
        f"Historical Sahm Rule peak: {sahm_max:.3f}"
        + (f" on {sahm_max_date.date()}" if sahm_max_date is not None else "")
    )

    print("\nSAHM RULE PREVIEW")
    print(sahm_preview)
    print(sahm_comment)
    print(sahm_peak_comment)

    write_to_txt("SAHM RULE PREVIEW\n" + sahm_preview)
    write_to_txt(sahm_comment)
    write_to_txt(sahm_peak_comment)

    save_line_plot(
        df=df,
        x_col="Date",
        y_cols="SAHM_RULE",
        labels="Sahm Rule",
        title="Sahm Rule Indicator",
        y_label="Unemployment Increase (pp)",
        file_name="sahm_rule.png",
        y_limits="auto",
        hlines=[{"y": 0.5, "label": "Threshold = 0.5"}]
    )

    return df


# =========================================================
# BEVERIDGE CURVE + TIGHTNESS
# Beveridge = regression
# Tightness = derived indicator
# =========================================================

def run_beveridge_curve() -> pd.DataFrame:
    df = load_csv("Table_Beveridge's_Curve.csv")
    df = df.dropna(subset=["U_RATE", "JOB_OPENINGS_RATE"]).copy()

    corr_text = df[["U_RATE", "JOB_OPENINGS_RATE"]].corr().to_string()

    print("\nBEVERIDGE CORRELATION")
    print(corr_text)
    write_to_txt("BEVERIDGE CORRELATION\n" + corr_text)

    run_ols(
        y=df["U_RATE"],
        X=df[["JOB_OPENINGS_RATE"]],
        model_name="Beveridge's Curve: U_RATE ~ JOB_OPENINGS_RATE"
    )

    save_scatter_plot(
        df=df,
        x_col="U_RATE",
        y_col="JOB_OPENINGS_RATE",
        title="Beveridge Curve",
        x_label="Unemployment Rate (%)",
        y_label="Job Openings Rate (%)",
        file_name="beveridge_curve_scatter.png",
        fit_type="quadratic",
        y_limits="auto"
    )

    df["TIGHTNESS"] = df["JOB_OPENINGS_RATE"] / df["U_RATE"]

    tightness_preview = (
        df[["Date", "JOB_OPENINGS_RATE", "U_RATE", "TIGHTNESS"]]
        .tail(15)
        .to_string(index=False)
    )

    print("\nTIGHTNESS PREVIEW")
    print(tightness_preview)
    write_to_txt("TIGHTNESS PREVIEW\n" + tightness_preview)

    save_line_plot(
        df=df,
        x_col="Date",
        y_cols="TIGHTNESS",
        labels="Labour Market Tightness",
        title="Labour Market Tightness Over Time",
        y_label="Vacancy-to-Unemployment Ratio",
        file_name="labour_market_tightness.png",
        show_legend=False,
        y_limits="auto"
    )

    df_ex_covid = df[
        (df["Date"] < pd.Timestamp("2020-04-01")) | (df["Date"] > pd.Timestamp("2021-10-01"))
    ].copy()

    corr_ex_covid_text = df_ex_covid[["U_RATE", "JOB_OPENINGS_RATE"]].corr().to_string()

    print("\nBEVERIDGE CORRELATION - EX COVID")
    print(corr_ex_covid_text)
    write_to_txt("BEVERIDGE CORRELATION - EX COVID\n" + corr_ex_covid_text)

    plt.figure(figsize=(7, 5))
    plt.scatter(df["U_RATE"], df["JOB_OPENINGS_RATE"], alpha=0.6, label="Full sample")
    plt.scatter(df_ex_covid["U_RATE"], df_ex_covid["JOB_OPENINGS_RATE"], alpha=0.6, label="Ex-COVID")

    # Quadratic fit - full sample
    valid_full = df[["U_RATE", "JOB_OPENINGS_RATE"]].dropna()
    if len(valid_full) > 2:
        x_full = valid_full["U_RATE"].astype(float)
        y_full = valid_full["JOB_OPENINGS_RATE"].astype(float)
        x_line_full = np.linspace(x_full.min(), x_full.max(), 400)
        coef_full = np.polyfit(x_full, y_full, 2)
        y_line_full = coef_full[0] * x_line_full**2 + coef_full[1] * x_line_full + coef_full[2]

        full_fit_text = (
            "Beveridge Curve: Full Sample vs Ex-COVID - quadratic fit coefficients (full sample):\n"
            f"a = {coef_full[0]:.6f}, b = {coef_full[1]:.6f}, c = {coef_full[2]:.6f}"
        )
        print(full_fit_text)
        write_to_txt(full_fit_text)

        plt.plot(x_line_full, y_line_full, linewidth=2.5, label="Quadratic fit - full sample")

    # Quadratic fit - ex-COVID
    valid_ex = df_ex_covid[["U_RATE", "JOB_OPENINGS_RATE"]].dropna()
    if len(valid_ex) > 2:
        x_ex = valid_ex["U_RATE"].astype(float)
        y_ex = valid_ex["JOB_OPENINGS_RATE"].astype(float)
        x_line_ex = np.linspace(x_ex.min(), x_ex.max(), 400)
        coef_ex = np.polyfit(x_ex, y_ex, 2)
        y_line_ex = coef_ex[0] * x_line_ex**2 + coef_ex[1] * x_line_ex + coef_ex[2]

        ex_fit_text = (
            "Beveridge Curve: Full Sample vs Ex-COVID - quadratic fit coefficients (ex-COVID):\n"
            f"a = {coef_ex[0]:.6f}, b = {coef_ex[1]:.6f}, c = {coef_ex[2]:.6f}"
        )
        print(ex_fit_text)
        write_to_txt(ex_fit_text)

        plt.plot(x_line_ex, y_line_ex, linewidth=2.5, label="Quadratic fit - ex-COVID")

    plt.title("Beveridge Curve: Full Sample vs Ex-COVID")
    plt.xlabel("Unemployment Rate (%)")
    plt.ylabel("Job Openings Rate (%)")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "beveridge_full_vs_ex_covid.png", dpi=300, bbox_inches="tight")
    plt.close()

    return df


# =========================================================
# OKUN'S LAW - LEVEL
# Regression
# =========================================================

def run_okun_level() -> pd.DataFrame:
    df = load_csv("Table_Okun's_Law_Level.csv")
    df = df.dropna(subset=["U_GAP", "Y_GAP"]).copy()

    corr_text = df[["U_GAP", "Y_GAP"]].corr().to_string()

    print("\nOKUN LEVEL CORRELATION")
    print(corr_text)
    write_to_txt("OKUN LEVEL CORRELATION\n" + corr_text)

    run_ols(
        y=df["U_GAP"],
        X=df[["Y_GAP"]],
        model_name="Okun's Law (Level): U_GAP ~ Y_GAP"
    )

    save_scatter_plot(
        df=df,
        x_col="Y_GAP",
        y_col="U_GAP",
        title="Okun's Law (Level Specification)",
        x_label="Output Gap (%)",
        y_label="Unemployment Gap (pp)",
        file_name="okun_level_scatter.png",
        fit_type="linear"
    )

    return df


# =========================================================
# OKUN'S LAW - DIFFERENCE
# Regression
# =========================================================

def run_okun_difference() -> pd.DataFrame:
    df = load_csv("Table_Okun's_Law_Differences.csv")
    df = df.dropna(subset=["U_DELTA", "Y_GROWTH_GAP"]).copy()

    corr_text = df[["U_DELTA", "Y_GROWTH_GAP"]].corr().to_string()

    print("\nOKUN DIFFERENCE CORRELATION")
    print(corr_text)
    write_to_txt("OKUN DIFFERENCE CORRELATION\n" + corr_text)

    run_ols(
        y=df["U_DELTA"],
        X=df[["Y_GROWTH_GAP"]],
        model_name="Okun's Law (Difference): U_DELTA ~ Y_GROWTH_GAP"
    )

    save_scatter_plot(
        df=df,
        x_col="Y_GROWTH_GAP",
        y_col="U_DELTA",
        title="Okun's Law (Difference Specification)",
        x_label="Output Growth Gap (pp)",
        y_label="Delta Unemployment Rate (pp)",
        file_name="okun_difference_scatter.png",
        fit_type="linear",
        x_limits=(-10, 10),
        y_limits=(-3, 2)
    )

    return df


# =========================================================
# PHILLIPS CURVE
# Multiple regression
# =========================================================

def run_phillips_curve() -> pd.DataFrame:
    df = load_csv("Table_Phillips's_Curve.csv")
    df = df.dropna(subset=["EXP_INF", "PCE_HEAD", "U_GAP"]).copy()

    corr_text = df[["EXP_INF", "PCE_HEAD", "U_GAP"]].corr().to_string()
    print("\nPHILLIPS CORRELATIONS")
    print(corr_text)
    write_to_txt("PHILLIPS CORRELATIONS\n" + corr_text)

    run_ols(
        y=df["PCE_HEAD"],
        X=df[["EXP_INF", "U_GAP"]],
        model_name="Phillips's Curve: PCE_HEAD ~ EXP_INF + U_GAP"
    )

    # Illustrative bivariate scatter only
    save_scatter_plot(
        df=df,
        x_col="U_GAP",
        y_col="PCE_HEAD",
        title="Phillips Curve (Illustrative Scatter)",
        x_label="Unemployment Gap (pp)",
        y_label="Headline PCE Inflation (%)",
        file_name="phillips_scatter_illustrative.png",
        fit_type="quadratic",
        alpha=0.5
    )

    return df


# =========================================================
# TAYLOR RULE - Y GAP / CORE
# Benchmark implied rule, not regression
# =========================================================

def run_taylor_y_gap_core() -> pd.DataFrame:
    df = load_csv("Table_Taylor's_Rule_Y_Gap_PCE_Core.csv")
    df = df.dropna(subset=["FEDFUNDS", "PCE_CORE", "Y_GAP"]).copy()

    df["TAYLOR_IMPLIED_Y_CORE"] = (
        R_STAR
        + df["PCE_CORE"]
        + PHI_PI * (df["PCE_CORE"] - PI_STAR)
        + PHI_Y * df["Y_GAP"]
    )

    preview = (
        df[["Date", "FEDFUNDS", "PCE_CORE", "Y_GAP", "TAYLOR_IMPLIED_Y_CORE"]]
        .tail(15)
        .to_string(index=False)
    )

    print("\nTAYLOR Y-GAP CORE PREVIEW")
    print(preview)
    write_to_txt("TAYLOR Y-GAP CORE PREVIEW\n" + preview)

    save_line_plot(
        df=df,
        x_col="Date",
        y_cols=["FEDFUNDS", "TAYLOR_IMPLIED_Y_CORE"],
        labels=["Fed Funds Rate", "Taylor Implied Rate"],
        title="Taylor Rule: Y-Gap + Core PCE",
        y_label="Interest Rate (%)",
        file_name="taylor_y_gap_core.png",
        y_limits="auto"
    )

    return df


# =========================================================
# TAYLOR RULE - Y GAP / HEADLINE
# Benchmark implied rule, not regression
# =========================================================

def run_taylor_y_gap_headline() -> pd.DataFrame:
    df = load_csv("Table_Taylor's_Rule_Y_Gap_PCE_Headline.csv")
    df = df.dropna(subset=["FEDFUNDS", "PCE_HEAD", "Y_GAP"]).copy()

    df["TAYLOR_IMPLIED_Y_HEAD"] = (
        R_STAR
        + df["PCE_HEAD"]
        + PHI_PI * (df["PCE_HEAD"] - PI_STAR)
        + PHI_Y * df["Y_GAP"]
    )

    preview = (
        df[["Date", "FEDFUNDS", "PCE_HEAD", "Y_GAP", "TAYLOR_IMPLIED_Y_HEAD"]]
        .tail(15)
        .to_string(index=False)
    )

    print("\nTAYLOR Y-GAP HEADLINE PREVIEW")
    print(preview)
    write_to_txt("TAYLOR Y-GAP HEADLINE PREVIEW\n" + preview)

    save_line_plot(
        df=df,
        x_col="Date",
        y_cols=["FEDFUNDS", "TAYLOR_IMPLIED_Y_HEAD"],
        labels=["Fed Funds Rate", "Taylor Implied Rate"],
        title="Taylor Rule: Y-Gap + Headline PCE",
        y_label="Interest Rate (%)",
        file_name="taylor_y_gap_headline.png",
        y_limits="auto"
    )

    return df


# =========================================================
# TAYLOR RULE - U GAP / CORE
# Benchmark implied rule, not regression
# =========================================================

def run_taylor_u_gap_core() -> pd.DataFrame:
    df = load_csv("Table_Taylor's_Rule_U_Gap_PCE_Core.csv")
    df = df.dropna(subset=["FEDFUNDS", "PCE_CORE", "U_GAP"]).copy()

    df["TAYLOR_IMPLIED_U_CORE"] = (
        R_STAR
        + df["PCE_CORE"]
        + PHI_PI * (df["PCE_CORE"] - PI_STAR)
        - PHI_U * df["U_GAP"]
    )

    preview = (
        df[["Date", "FEDFUNDS", "PCE_CORE", "U_GAP", "TAYLOR_IMPLIED_U_CORE"]]
        .tail(15)
        .to_string(index=False)
    )

    print("\nTAYLOR U-GAP CORE PREVIEW")
    print(preview)
    write_to_txt("TAYLOR U-GAP CORE PREVIEW\n" + preview)

    save_line_plot(
        df=df,
        x_col="Date",
        y_cols=["FEDFUNDS", "TAYLOR_IMPLIED_U_CORE"],
        labels=["Fed Funds Rate", "Taylor Implied Rate"],
        title="Taylor Rule: U-Gap + Core PCE",
        y_label="Interest Rate (%)",
        file_name="taylor_u_gap_core.png",
        y_limits="auto"
    )

    return df


# =========================================================
# TAYLOR RULE - U GAP / HEADLINE
# Benchmark implied rule, not regression
# =========================================================

def run_taylor_u_gap_headline() -> pd.DataFrame:
    df = load_csv("Table_Taylor's_Rule_U_Gap_PCE_Headline.csv")
    df = df.dropna(subset=["FEDFUNDS", "PCE_HEAD", "U_GAP"]).copy()

    df["TAYLOR_IMPLIED_U_HEAD"] = (
        R_STAR
        + df["PCE_HEAD"]
        + PHI_PI * (df["PCE_HEAD"] - PI_STAR)
        - PHI_U * df["U_GAP"]
    )

    preview = (
        df[["Date", "FEDFUNDS", "PCE_HEAD", "U_GAP", "TAYLOR_IMPLIED_U_HEAD"]]
        .tail(15)
        .to_string(index=False)
    )

    print("\nTAYLOR U-GAP HEADLINE PREVIEW")
    print(preview)
    write_to_txt("TAYLOR U-GAP HEADLINE PREVIEW\n" + preview)

    save_line_plot(
        df=df,
        x_col="Date",
        y_cols=["FEDFUNDS", "TAYLOR_IMPLIED_U_HEAD"],
        labels=["Fed Funds Rate", "Taylor Implied Rate"],
        title="Taylor Rule: U-Gap + Headline PCE",
        y_label="Interest Rate (%)",
        file_name="taylor_u_gap_headline.png",
        y_limits="auto"
    )

    return df


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    print("Running US Monetary Policy and Business Cycle project...\n")

    if TXT_FILE.exists():
        TXT_FILE.unlink()

    for file in OUTPUTS_DIR.glob("*.png"):
        file.unlink()

    write_to_txt("Running US Monetary Policy and Business Cycle project...")

    run_sahm_rule()
    run_beveridge_curve()
    run_okun_level()
    run_okun_difference()
    run_phillips_curve()
    run_taylor_y_gap_core()
    run_taylor_y_gap_headline()
    run_taylor_u_gap_core()
    run_taylor_u_gap_headline()

    final_message = (
        "\nAll analyses completed successfully.\n"
        f"Outputs saved in: {OUTPUTS_DIR}"
    )

    print(final_message)
    write_to_txt(final_message)


if __name__ == "__main__":
    main()