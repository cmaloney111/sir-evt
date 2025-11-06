import numpy as np
import pandas as pd


def load_cdc_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def extract_seasonal_peaks(
    df: pd.DataFrame, region_col: str = "REGION", value_col: str = "%WEIGHTED ILI"
) -> pd.DataFrame:
    if value_col not in df.columns:
        value_col = (
            "ILI" if "ILI" in df.columns else df.select_dtypes(include=[np.number]).columns[0]
        )
    if region_col not in df.columns:
        region_col = "REGION TYPE" if "REGION TYPE" in df.columns else None

    if "YEAR" in df.columns:
        df["year"] = df["YEAR"]
    elif "DATE" in df.columns:
        df["year"] = pd.to_datetime(df["DATE"]).dt.year
    else:
        raise ValueError("No year column")

    if "WEEK" in df.columns:
        df["season_year"] = df.apply(
            lambda row: row["year"] if row["WEEK"] < 30 else row["year"] + 1, axis=1
        )
    else:
        df["season_year"] = df["year"]

    if region_col and region_col in df.columns:
        peaks = df.groupby(["season_year", region_col])[value_col].max().reset_index()
        peaks.columns = ["season", "region", "peak_value"]
    else:
        peaks = df.groupby("season_year")[value_col].max().reset_index()
        peaks.columns = ["season", "peak_value"]
        peaks["region"] = "National"

    return peaks


def extract_weekly_exceedances(
    df: pd.DataFrame, threshold_percentile: float = 90, value_col: str = "%WEIGHTED ILI"
) -> tuple[np.ndarray, float]:
    values = df[value_col].dropna().values
    threshold = np.percentile(values, threshold_percentile)
    return values[values > threshold] - threshold, threshold


def prepare_sir_data(df: pd.DataFrame, region: str) -> pd.DataFrame:
    df_region = df[df["REGION"] == region].copy()

    # Keep only season weeks
    df_region = df_region[df_region["WEEK"].between(1, 20) | df_region["WEEK"].between(40, 52)]

    # Assign season_year
    df_region["season_year"] = df_region.apply(
        lambda row: row["YEAR"] if row["WEEK"] >= 40 else row["YEAR"] - 1, axis=1
    )

    # Map to 0-based season_week
    df_region["season_week"] = df_region.apply(
        lambda row: row["WEEK"] - 40 if row["WEEK"] >= 40 else row["WEEK"] + 12, axis=1
    )

    # Sort by season + week
    df_region = df_region.sort_values(["season_year", "season_week"]).reset_index(drop=True)

    return pd.DataFrame({"week": range(len(df_region)), "cases": df_region["%WEIGHTED ILI"].values})
