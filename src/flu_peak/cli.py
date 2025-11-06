"""Command-line interface for running experiments."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from flu_peak.data import generate_synthetic_data, load_flu_data
from flu_peak.models.gev_testing import GEVModel
from flu_peak.models.sir_testing import SIRModel
from flu_peak.plot import (
    plot_comparison_metrics,
    plot_data_overview,
    plot_gev_fit,
    plot_region_seasons,
    plot_sir_dynamics,
)


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
    samples = np.asarray(samples)
    samples = samples[~np.isnan(samples)]
    if len(samples) < 2:
        return np.nan
    try:
        kde = gaussian_kde(samples)
        density = kde.evaluate(observation)[0]
        return -np.log(density + 1e-10)
    except Exception:
        return np.nan


def extract_seasonal_peaks_simple(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract seasonal peaks from CDC data."""
    df = df.copy()
    df["season_year"] = df.apply(
        lambda row: row["YEAR"] if row["WEEK"] < 30 else row["YEAR"] + 1, axis=1
    )

    peaks_list = []
    for region in df["REGION"].unique():
        region_df = df[df["REGION"] == region]
        for season in region_df["season_year"].unique():
            season_df = region_df[region_df["season_year"] == season]
            season_df = season_df[
                ((season_df["WEEK"] >= 40) & (season_df["YEAR"] == season - 1))
                | ((season_df["WEEK"] <= 20) & (season_df["YEAR"] == season))
            ]
            if len(season_df) == 0:
                continue
            peak_val = season_df["%WEIGHTED ILI"].max()
            peaks_list.append({"season": season, "region": region, "peak_value": peak_val})

    peaks_df = pd.DataFrame(peaks_list)
    peaks_df = peaks_df.sort_values(["season", "region"]).reset_index(drop=True)
    return peaks_df, peaks_df["peak_value"].values


def prepare_sir_data_simple(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Prepare SIR time series data for a specific region."""
    df_region = df[df["REGION"] == region].copy()
    df_region = df_region[df_region["WEEK"].between(1, 20) | df_region["WEEK"].between(40, 52)]
    df_region["season_year"] = df_region.apply(
        lambda row: row["YEAR"] if row["WEEK"] >= 40 else row["YEAR"] - 1, axis=1
    )
    df_region["season_week"] = df_region.apply(
        lambda row: row["WEEK"] - 40 if row["WEEK"] >= 40 else row["WEEK"] + 12, axis=1
    )
    df_region = df_region.sort_values(["season_year", "season_week"]).reset_index(drop=True)
    return pd.DataFrame({"week": range(len(df_region)), "cases": df_region["%WEIGHTED ILI"].values})


def main() -> None:
    """Run influenza peak prediction experiment."""
    parser = argparse.ArgumentParser(description="Influenza peak prediction with GEV and SIR")
    parser.add_argument("--data", type=Path, help="Path to CDC CSV data file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--seasons", type=int, default=5, help="Seasons for synthetic data")
    parser.add_argument("--regions", type=int, default=3, help="Regions for synthetic data")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output dir")
    parser.add_argument("--test-seasons", type=int, default=1, help="Seasons to hold out")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--year", type=int, default=13, help="Year index for SIR example plot")
    parser.add_argument("--n-weeks", type=int, default=40, help="Weeks to fit SIR on")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("INFLUENZA PEAK PREDICTION ANALYSIS\n")

    # Load data
    if args.synthetic:
        print(f"Generating synthetic data: {args.seasons} seasons, {args.regions} regions")
        df = generate_synthetic_data(n_seasons=args.seasons, n_regions=args.regions, seed=args.seed)
        df.to_csv(args.output / "synthetic_data.csv", index=False)
        print(f"Saved to {args.output / 'synthetic_data.csv'}")
    elif args.data:
        print(f"Loading data from {args.data}")
        df = load_flu_data(args.data)
    else:
        print("Error: Must specify --data or --synthetic")
        return

    print(f"  {len(df)} rows loaded\n")

    # Extract peaks
    df["season_year"] = df.apply(
        lambda row: row["YEAR"] if row["WEEK"] < 30 else row["YEAR"] + 1, axis=1
    )
    seasonal_peaks_df, seasonal_peaks = extract_seasonal_peaks_simple(df)

    # Data overview plot
    df["region_number"] = df.apply(lambda row: int(row["REGION"].split()[1]), axis=1)
    weekly_data_by_region = df[["%WEIGHTED ILI", "region_number"]].dropna().values
    plot_data_overview(
        weekly_data_by_region, seasonal_peaks, str(args.output / "data_overview.png")
    )

    # Train/test split
    n_train = len(seasonal_peaks) - (args.test_seasons * len(df["REGION"].unique()))
    train_peaks = seasonal_peaks[:n_train]
    test_peaks = seasonal_peaks[n_train:]
    test_peaks_df = seasonal_peaks_df.iloc[n_train:].copy()

    print(f"Train/test split: {n_train}/{len(test_peaks)} region-seasons\n")

    # Fit GEV
    print("Fitting GEV model on training data...")
    gev = GEVModel()
    gev.fit(train_peaks, method="mle")
    gev.goodness_of_fit(train_peaks)
    plot_gev_fit(gev, train_peaks, str(args.output / "gev_fit.png"))
    print(f"  10-yr return: {gev.return_level(10):.1f}")
    print(f"  100-yr return: {gev.return_level(100):.1f}\n")

    # Generate predictions
    print("Generating probabilistic predictions for test region-seasons...")
    gev_prediction_samples = []
    sir_prediction_samples = []
    n_samples = 200

    for idx, row in test_peaks_df.iterrows():
        season = row["season"]
        region = row.get("region")
        print(f"Season: {season}, region: {region}")

        # GEV samples
        gev_samples = gev.sample(n_samples)
        gev_prediction_samples.append(gev_samples)

        # SIR samples
        season_df = df[df["season_year"] == season]
        season_sir_data = prepare_sir_data_simple(season_df, region=region)
        season_time = season_sir_data["week"].values
        season_infected = season_sir_data["cases"].values

        sir_model = SIRModel(population=1e6)
        season_infected_scaled = season_infected * (sir_model.population / 100)

        n_fit = min(args.n_weeks, len(season_time))
        sir_model.fit(season_time[:n_fit], season_infected_scaled[:n_fit], n_weeks=args.n_weeks)

        sir_samples = sir_model.bootstrap_predict_peak(
            season_time[:n_fit], season_infected_scaled[:n_fit], n_samples=n_samples
        )
        sir_prediction_samples.append(sir_samples)

    # Example SIR plot
    example_region = df["REGION"].unique()[0]
    sir_data = prepare_sir_data_simple(df, region=example_region)
    plot_region_seasons(sir_data, str(args.output / "one_season_overview.png"))

    time, infected = sir_data["week"].values, sir_data["cases"].values
    sir = SIRModel(population=1e6)
    start_idx = min(args.year * 33, len(time) - 33)
    end_idx = min((args.year + 1) * 33, len(time))
    if end_idx - start_idx > 0:
        infected_scaled = infected * (sir.population / 100)
        sir.fit(time[start_idx:end_idx], infected_scaled[start_idx:end_idx])
        plot_sir_dynamics(
            sir,
            time[start_idx:end_idx],
            infected_scaled[start_idx:end_idx],
            str(args.output / "sir_dynamics.png"),
        )
        print(f"  Example SIR R₀={sir.R0:.2f}\n")
    else:
        print("  Skipping SIR dynamics plot (not enough data)\n")

    # Compute scoring rules
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

    # Comparison plot
    metrics = plot_comparison_metrics(
        gev_medians,
        sir_medians,
        test_peaks,
        str(args.output / "comparison_metrics.png"),
        gev_samples=gev_prediction_samples,
        sir_samples=sir_prediction_samples,
        gev_crps=gev_crps,
        sir_crps=sir_crps,
        gev_logscore=gev_logscore,
        sir_logscore=sir_logscore,
    )

    # Print results
    print(f"\nCOMPLETE - Plots saved to {args.output}/")
    print("\nKey Results:")
    print(f"  GEV ξ={gev.shape:.3f}, 100-yr={gev.return_level(100):.1f}")
    print("\nProper Scoring Rules (lower is better):")
    print(f"  GEV CRPS: {np.nanmean(gev_crps):.3f}, LogScore: {np.nanmean(gev_logscore):.3f}")
    print(f"  SIR CRPS: {np.nanmean(sir_crps):.3f}, LogScore: {np.nanmean(sir_logscore):.3f}")
    print("\nMedian Predictions:")
    print(f"  GEV MAE: {metrics['gev_mae']:.2f}, RMSE: {metrics['gev_rmse']:.2f}")
    print(f"  SIR MAE: {metrics['sir_mae']:.2f}, RMSE: {metrics['sir_rmse']:.2f}")


if __name__ == "__main__":
    main()
