import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data.download_cdc_data import create_simulated_cdc_data
from preprocessing.data_prep import load_cdc_data, extract_seasonal_peaks, extract_weekly_exceedances, prepare_sir_data
from models.gev_model import GEVModel
from models.gpd_model import GPDModel
from models.sir_model import SIRModel
from visualization.plots import plot_gev_fit, plot_gpd_fit, plot_sir_dynamics, plot_comparison_metrics, plot_data_overview


def run_analysis():
    print("EVT ANALYSIS OF INFLUENZA DATA")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    data_dir = Path("data")
    
    df = pd.read_csv(data_dir / "cdc_flu_data.csv")

    print(f"  {len(df)} rows loaded\n")

    if 'YEAR' not in df.columns and 'DATE' in df.columns:
        df['YEAR'] = pd.to_datetime(df['DATE']).dt.year
    if 'WEEK' in df.columns and 'YEAR' in df.columns:
        df['season_year'] = df.apply(lambda row: row['YEAR'] if row['WEEK'] < 30 else row['YEAR'] + 1, axis=1)

    seasonal_peaks_df = extract_seasonal_peaks(df)

    if 'region' in seasonal_peaks_df.columns:
        seasonal_peaks_df = seasonal_peaks_df.sort_values(['season', 'region']).reset_index(drop=True)
    else:
        seasonal_peaks_df = seasonal_peaks_df.sort_values('season').reset_index(drop=True)

    seasonal_peaks = seasonal_peaks_df['peak_value'].values

    weekly_data = df['%WEIGHTED ILI'].dropna().values
    sir_data = prepare_sir_data(df)
    time, infected = sir_data['week'].values, sir_data['cases'].values

    plot_data_overview(weekly_data, seasonal_peaks, str(output_dir / "data_overview.png"))

    n_train = int(len(seasonal_peaks) * 0.8)
    train_peaks = seasonal_peaks[:n_train]
    test_peaks = seasonal_peaks[n_train:]
    test_peaks_df = seasonal_peaks_df.iloc[n_train:].copy()

    print(f"Train/test split: {n_train}/{len(test_peaks)} region-seasons\n")

    print("Fitting GEV model on training data...")
    gev = GEVModel()
    gev.fit(train_peaks, method="mle")
    gev.goodness_of_fit(train_peaks)
    plot_gev_fit(gev, train_peaks, str(output_dir / "gev_fit.png"))
    print(f"  10-yr return: {gev.return_level(10):.1f}")
    print(f"  100-yr return: {gev.return_level(100):.1f}\n")

    print("Fitting GPD model on all weekly data...")
    gpd = GPDModel()
    gpd.fit(weekly_data, threshold_percentile=90, method="mle")
    gpd.goodness_of_fit(weekly_data)
    plot_gpd_fit(gpd, weekly_data, str(output_dir / "gpd_fit.png"))
    print(f"  P(exceed 2×u): {gpd.predict_exceedance_probability(gpd.threshold * 2):.4f}\n")

    print("Generating predictions for test region-seasons...")
    gev_predictions = np.array([gev.return_level(2) for _ in test_peaks])
    sir_predictions = []

    for idx, row in test_peaks_df.iterrows():
        season = row['season']
        region = row.get('region', None)

        if 'season_year' in df.columns:
            season_df = df[df['season_year'] == season]
        else:
            season_df = df[df['YEAR'] == season]

        if region and 'REGION' in df.columns:
            season_df = season_df[season_df['REGION'] == region]

        if len(season_df) < 10:
            sir_predictions.append(np.nan)
            continue

        season_sir_data = prepare_sir_data(season_df)
        season_time = season_sir_data['week'].values
        season_infected = season_sir_data['cases'].values

        sir_model = SIRModel(population=1e6)
        try:
            season_infected_scaled = season_infected * (sir_model.population / 100)
            sir_model.fit(season_time[:min(40, len(season_time))], season_infected_scaled[:min(40, len(season_time))])
            peak_i, _ = sir_model.peak_infected()
            peak_i_percentage = (peak_i / sir_model.population) * 100
            sir_predictions.append(peak_i_percentage)
        except:
            sir_predictions.append(np.nan)

    sir_predictions = np.array(sir_predictions)

    sir = SIRModel(population=1e6)
    n_weeks = min(40, len(time))
    infected_scaled = infected * (sir.population / 100)
    sir.fit(time[:n_weeks], infected_scaled[:n_weeks])
    plot_sir_dynamics(sir, time[:n_weeks], infected_scaled[:n_weeks], str(output_dir / "sir_dynamics.png"))
    print(f"  Example SIR R₀={sir.R0:.2f}\n")

    print("Generating comparison metrics...")
    metrics = plot_comparison_metrics(gev_predictions, sir_predictions, test_peaks, str(output_dir / "comparison_metrics.png"))

    print("COMPLETE - Plots saved to outputs/")
    print(f"\nKey Results:")
    print(f"  GEV ξ={gev.shape:.3f}, 100-yr={gev.return_level(100):.1f}")
    print(f"  GPD ξ={gpd.shape:.3f}")
    print(f"  GEV Test MAE: {metrics['gev_mae']:.2f}, RMSE: {metrics['gev_rmse']:.2f}")
    print(f"  SIR Test MAE: {metrics['sir_mae']:.2f}, RMSE: {metrics['sir_rmse']:.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run_analysis()
