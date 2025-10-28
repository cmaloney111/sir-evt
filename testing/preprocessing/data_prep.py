import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_cdc_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def extract_seasonal_peaks(df: pd.DataFrame, region_col: str = 'REGION', value_col: str = '%WEIGHTED ILI') -> pd.DataFrame:
    if value_col not in df.columns:
        value_col = 'ILI' if 'ILI' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    if region_col not in df.columns:
        region_col = 'REGION TYPE' if 'REGION TYPE' in df.columns else None

    if 'YEAR' in df.columns:
        df['year'] = df['YEAR']
    elif 'DATE' in df.columns:
        df['year'] = pd.to_datetime(df['DATE']).dt.year
    else:
        raise ValueError("No year column")

    if 'WEEK' in df.columns:
        df['season_year'] = df.apply(lambda row: row['year'] if row['WEEK'] < 30 else row['year'] + 1, axis=1)
    else:
        df['season_year'] = df['year']

    if region_col and region_col in df.columns:
        peaks = df.groupby(['season_year', region_col])[value_col].max().reset_index()
        peaks.columns = ['season', 'region', 'peak_value']
    else:
        peaks = df.groupby('season_year')[value_col].max().reset_index()
        peaks.columns = ['season', 'peak_value']
        peaks['region'] = 'National'

    return peaks


def extract_weekly_exceedances(df: pd.DataFrame, threshold_percentile: float = 90, value_col: str = '%WEIGHTED ILI') -> Tuple[np.ndarray, float]:
    if value_col not in df.columns:
        value_col = 'ILI' if 'ILI' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    values = df[value_col].dropna().values
    threshold = np.percentile(values, threshold_percentile)
    return values[values > threshold] - threshold, threshold


def prepare_sir_data(df: pd.DataFrame, region: Optional[str] = None) -> pd.DataFrame:
    result_df = df.copy()
    if region is not None and 'REGION' in df.columns:
        result_df = result_df[result_df['REGION'] == region]

    value_col = next((c for c in ['ILI', '%WEIGHTED ILI', 'ILITOTAL'] if c in result_df.columns), None)
    if not value_col:
        raise ValueError("No ILI column")

    result_df = result_df.sort_values('WEEK' if 'WEEK' in result_df.columns else result_df.columns[0]).reset_index(drop=True)
    return pd.DataFrame({'week': range(len(result_df)), 'cases': result_df[value_col].values})
