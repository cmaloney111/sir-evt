import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from models.gev_model import GEVModel
from models.sir_model import SIRModel
from preprocessing.data_prep import extract_seasonal_peaks, prepare_sir_data
from visualization.plots import (
    plot_comparison_metrics,
    plot_data_overview,
    plot_gev_fit,
    plot_region_seasons,
    plot_sir_dynamics,
)

TRAINING_SPLIT = 0.8
N_SAMPLES = 200


def crps_empirical(samples: np.ndarray, observation: float) -> float:
    """CRPS using empirical distribution of samples."""
    samples = np.asarray(samples)
    samples = samples[~np.isnan(samples)]
    if len(samples) == 0:
        return np.nan
    term1 = np.mean(np.abs(samples - observation))
    term2 = np.mean(np.abs(samples[:, None] - samples[None, :])) / 2
    return term1 - term2


def log_score(samples: np.ndarray, observation: float) -> float:
    """Log score using KDE."""
    from scipy.stats import gaussian_kde

    samples = np.asarray(samples)
    samples = samples[~np.isnan(samples)]
    if len(samples) < 2:
        return np.nan
    try:
        kde = gaussian_kde(samples)
        density = kde.evaluate(observation)[0]
        return -np.log(density + 1e-10)
    except:
        return np.nan


def run_analysis(args):
    print("EVT ANALYSIS OF INFLUENZA DATA")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    data_dir = Path("data")

    df = pd.read_csv(data_dir / "cdc_flu_data.csv")

    print(f"  {len(df)} rows loaded\n")

    df["season_year"] = df.apply(
        lambda row: row["YEAR"] if row["WEEK"] < 30 else row["YEAR"] + 1, axis=1
    )

    seasonal_peaks_df = extract_seasonal_peaks(df)
    seasonal_peaks_df = seasonal_peaks_df.sort_values(["season", "region"]).reset_index(drop=True)
    seasonal_peaks = seasonal_peaks_df["peak_value"].values

    df["region_number"] = df.apply(lambda row: int(row["REGION"].split()[1]), axis=1)
    weekly_data_by_region = df[["%WEIGHTED ILI", "region_number"]].dropna().values

    plot_data_overview(weekly_data_by_region, seasonal_peaks, str(output_dir / "data_overview.png"))
    weekly_data = weekly_data_by_region[:, 0]

    n_train = int(len(seasonal_peaks) * TRAINING_SPLIT)
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

    print("Generating probabilistic predictions for test region-seasons...")
    gev_prediction_samples = []
    sir_prediction_samples = []

    for idx, row in test_peaks_df.iterrows():
        season = row["season"]
        region = row.get("region")
        print(f"Season: {season}, region: {region}")

        gev_samples = gev.sample(N_SAMPLES)
        gev_prediction_samples.append(gev_samples)

        season_df = df[df["season_year"] == season]
        season_sir_data = prepare_sir_data(season_df, region=region)
        season_time = season_sir_data["week"].values
        season_infected = season_sir_data["cases"].values

        sir_model = SIRModel(population=1e6)
        season_infected_scaled = season_infected * (sir_model.population / 100)

        n_fit = min(args.n_weeks if args.n_weeks else 40, len(season_time))
        sir_model.fit(season_time[:n_fit], season_infected_scaled[:n_fit], n_weeks=args.n_weeks)

        sir_samples = sir_model.bootstrap_predict_peak(
            season_time[:n_fit], season_infected_scaled[:n_fit], n_samples=N_SAMPLES
        )
        sir_prediction_samples.append(sir_samples)

    sir_data = prepare_sir_data(df, region="Region 6")
    plot_region_seasons(sir_data, str(output_dir / "one_season_overview.png"))
    time, infected = sir_data["week"].values, sir_data["cases"].values

    sir = SIRModel(population=1e6)

    year = args.year  # 13
    start_idx = year * 33
    end_idx = (year + 1) * 33
    infected_scaled = infected * (sir.population / 100)
    sir.fit(time[start_idx:end_idx], infected_scaled[start_idx:end_idx])
    plot_sir_dynamics(
        sir,
        time[start_idx:end_idx],
        infected_scaled[start_idx:end_idx],
        str(output_dir / "sir_dynamics.png"),
    )
    print(f"  Example SIR R₀={sir.R0:.2f}\n")

    print("Computing proper scoring rules...")
    gev_crps = np.array(
        [crps_empirical(gev_prediction_samples[i], test_peaks[i]) for i in range(len(test_peaks))]
    )
    sir_crps = np.array(
        [crps_empirical(sir_prediction_samples[i], test_peaks[i]) for i in range(len(test_peaks))]
    )

    gev_logscore = np.array(
        [log_score(gev_prediction_samples[i], test_peaks[i]) for i in range(len(test_peaks))]
    )
    sir_logscore = np.array(
        [log_score(sir_prediction_samples[i], test_peaks[i]) for i in range(len(test_peaks))]
    )

    gev_medians = np.array([np.median(s) for s in gev_prediction_samples])
    sir_medians = np.array([np.median(s) for s in sir_prediction_samples])

    metrics = plot_comparison_metrics(
        gev_medians,
        sir_medians,
        test_peaks,
        str(output_dir / "comparison_metrics.png"),
        gev_samples=gev_prediction_samples,
        sir_samples=sir_prediction_samples,
        gev_crps=gev_crps,
        sir_crps=sir_crps,
        gev_logscore=gev_logscore,
        sir_logscore=sir_logscore,
    )

    print("COMPLETE - Plots saved to outputs/")
    print("\nKey Results:")
    print(f"  GEV ξ={gev.shape:.3f}, 100-yr={gev.return_level(100):.1f}")
    print("\nProper Scoring Rules (lower is better):")
    print(f"  GEV CRPS: {np.nanmean(gev_crps):.3f}, LogScore: {np.nanmean(gev_logscore):.3f}")
    print(f"  SIR CRPS: {np.nanmean(sir_crps):.3f}, LogScore: {np.nanmean(sir_logscore):.3f}")
    print("\nMedian Predictions:")
    print(f"  GEV MAE: {metrics['gev_mae']:.2f}, RMSE: {metrics['gev_rmse']:.2f}")
    print(f"  SIR MAE: {metrics['sir_mae']:.2f}, RMSE: {metrics['sir_rmse']:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int)
    parser.add_argument("--n_weeks", type=int)
    args = parser.parse_args()
    run_analysis(args)
