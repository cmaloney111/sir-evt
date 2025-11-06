import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_gev_fit(gev_model, data: np.ndarray, output_path: str = "gev_fit.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(data, bins=15, density=True, alpha=0.7, edgecolor="black", label="Data")
    x = np.linspace(data.min() * 0.8, data.max() * 1.2, 200)
    axes[0, 0].plot(x, gev_model.pdf(x), "r-", linewidth=2, label="GEV fit")
    axes[0, 0].set_xlabel("Peak Value")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("GEV Fit to Seasonal Peaks")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    data_sorted = np.sort(data)
    n = len(data_sorted)
    p = (np.arange(1, n + 1) - 0.5) / n
    theo = gev_model.cdf(data_sorted)
    axes[0, 1].scatter(p, theo, alpha=0.6, s=30)
    axes[0, 1].plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect fit")
    axes[0, 1].set_xlabel("Empirical Probability")
    axes[0, 1].set_ylabel("Theoretical Probability")
    axes[0, 1].set_title("Probability-Probability Plot")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    rp = np.array([2, 5, 10, 20, 50, 100])
    rl = [gev_model.return_level(r) for r in rp]
    axes[1, 0].semilogx(rp, rl, "bo-", linewidth=2, markersize=8, label="GEV return levels")
    axes[1, 0].axhline(
        data.max(), color="r", linestyle="--", linewidth=2, label=f"Observed max: {data.max():.1f}"
    )
    axes[1, 0].set_xlabel("Return Period (years)")
    axes[1, 0].set_ylabel("Return Level")
    axes[1, 0].set_title("Return Level Plot")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(data_sorted, p, "o", alpha=0.6, label="Empirical CDF", markersize=5)
    axes[1, 1].plot(data_sorted, gev_model.cdf(data_sorted), "r-", linewidth=2, label="GEV CDF")
    axes[1, 1].set_xlabel("Peak Value")
    axes[1, 1].set_ylabel("Cumulative Probability")
    axes[1, 1].set_title("CDF Comparison")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sir_dynamics(
    sir_model, time: np.ndarray, observed: np.ndarray, output_path: str = "sir_dynamics.png"
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    time = time - time[0]
    I0 = observed[0] / sir_model.population
    initial = [1 - I0, I0, 0.0]

    extended_time = np.linspace(0, max(time) * 2, 52)
    pred = sir_model.predict(extended_time, initial)

    axes[0, 0].plot(extended_time, pred["S"] / sir_model.population, "b-", linewidth=2, label="S")
    axes[0, 0].plot(extended_time, pred["I"] / sir_model.population, "r-", linewidth=2, label="I")
    axes[0, 0].plot(extended_time, pred["R"] / sir_model.population, "g-", linewidth=2, label="R")
    axes[0, 0].set_xlabel("Time (weeks)")
    axes[0, 0].set_ylabel("Proportion")
    axes[0, 0].set_title(f"SIR Dynamics (R₀={sir_model.R0:.2f})")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    predicted = sir_model.predict(time, initial)["I"]
    axes[0, 1].plot(time, observed, "ko", markersize=8, alpha=0.6, label="Observed")
    axes[0, 1].plot(time, predicted, "r-", linewidth=2, label="SIR fit")
    axes[0, 1].set_xlabel("Time (weeks)")
    axes[0, 1].set_ylabel("Infected")
    axes[0, 1].set_title("Model Fit to Data")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    residuals = observed - predicted
    axes[1, 0].plot(time, residuals, "bo-", linewidth=2, markersize=6)
    axes[1, 0].axhline(0, color="r", linestyle="--", linewidth=2)
    axes[1, 0].set_xlabel("Time (weeks)")
    axes[1, 0].set_ylabel("Residual (Obs - Pred)")
    axes[1, 0].set_title(f"Residuals (RMSE={np.sqrt(np.mean(residuals**2)):.2f})")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(
        pred["S"] / sir_model.population, pred["I"] / sir_model.population, "b-", linewidth=2
    )
    axes[1, 1].plot(
        (pred["S"] / sir_model.population)[0],
        (pred["I"] / sir_model.population)[0],
        "go",
        markersize=10,
        label="Start",
    )
    peak_idx = np.argmax(pred["I"])
    axes[1, 1].plot(
        (pred["S"] / sir_model.population)[peak_idx],
        (pred["I"] / sir_model.population)[peak_idx],
        "r*",
        markersize=15,
        label="Peak",
    )
    axes[1, 1].set_xlabel("S (proportion)")
    axes[1, 1].set_ylabel("I (proportion)")
    axes[1, 1].set_title("Phase Portrait (S-I plane)")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison_metrics(
    gev_predictions,
    sir_predictions,
    actual_peaks,
    output_path: str = "comparison_metrics.png",
    gev_samples=None,
    sir_samples=None,
    gev_crps=None,
    sir_crps=None,
    gev_logscore=None,
    sir_logscore=None,
):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    test_indices = np.arange(len(actual_peaks))

    # Plot median predictions with IQR
    if gev_samples is not None:
        gev_q25 = np.array([np.percentile(s, 25) for s in gev_samples])
        gev_q75 = np.array([np.percentile(s, 75) for s in gev_samples])
        axes[0, 0].fill_between(
            test_indices, gev_q25, gev_q75, alpha=0.2, color="blue", label="GEV IQR"
        )

    if sir_samples is not None:
        sir_q25 = np.array(
            [
                np.percentile(s[~np.isnan(s)], 25) if np.sum(~np.isnan(s)) > 0 else np.nan
                for s in sir_samples
            ]
        )
        sir_q75 = np.array(
            [
                np.percentile(s[~np.isnan(s)], 75) if np.sum(~np.isnan(s)) > 0 else np.nan
                for s in sir_samples
            ]
        )
        axes[0, 0].fill_between(
            test_indices, sir_q25, sir_q75, alpha=0.2, color="red", label="SIR IQR"
        )

    axes[0, 0].plot(
        test_indices,
        actual_peaks,
        "ko-",
        linewidth=2,
        markersize=8,
        label="Actual",
        alpha=0.7,
        zorder=10,
    )
    axes[0, 0].plot(
        test_indices,
        gev_predictions,
        "bs-",
        linewidth=2,
        markersize=6,
        label="GEV Median",
        alpha=0.7,
        zorder=5,
    )
    axes[0, 0].plot(
        test_indices,
        sir_predictions,
        "r^-",
        linewidth=2,
        markersize=6,
        label="SIR Median",
        alpha=0.7,
        zorder=5,
    )
    axes[0, 0].set_xlabel("Test Season")
    axes[0, 0].set_ylabel("Peak Value (%ILI)")
    axes[0, 0].set_title("Probabilistic Predictions vs Actual")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    # Absolute error bar plot
    gev_errors = np.abs(gev_predictions - actual_peaks)
    sir_errors = np.abs(sir_predictions - actual_peaks)
    axes[0, 1].bar(test_indices, gev_errors, width=0.4, alpha=0.7, label="GEV", align="edge")
    axes[0, 1].bar(
        np.array(test_indices) + 0.4,
        sir_errors,
        width=0.4,
        alpha=0.7,
        label="SIR",
        color="red",
        align="edge",
    )
    axes[0, 1].set_xlabel("Test Season")
    axes[0, 1].set_ylabel("Absolute Error")
    axes[0, 1].set_title("Prediction Errors by Season")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Summary metrics (MAE, RMSE, MAPE)
    metrics = ["MAE", "RMSE", "MAPE (%)"]
    gev_mae = np.mean(gev_errors)
    sir_mae = np.mean(sir_errors)
    gev_rmse = np.sqrt(np.mean((gev_predictions - actual_peaks) ** 2))
    sir_rmse = np.sqrt(np.mean((sir_predictions - actual_peaks) ** 2))
    gev_mape = np.mean(np.abs((actual_peaks - gev_predictions) / actual_peaks)) * 100
    sir_mape = np.mean(np.abs((actual_peaks - sir_predictions) / actual_peaks)) * 100

    # Include CRPS and log score if provided
    if gev_crps is not None and sir_crps is not None:
        gev_crps_mean = np.mean(gev_crps)
        sir_crps_mean = np.mean(sir_crps)
        metrics.append("CRPS")
    else:
        gev_crps_mean = sir_crps_mean = np.nan

    if gev_logscore is not None and sir_logscore is not None:
        gev_logscore_mean = np.mean(gev_logscore)
        sir_logscore_mean = np.mean(sir_logscore)
        metrics.append("Log Score")
    else:
        gev_logscore_mean = sir_logscore_mean = np.nan

    gev_vals = [gev_mae, gev_rmse, gev_mape]
    sir_vals = [sir_mae, sir_rmse, sir_mape]

    if not np.isnan(gev_crps_mean):
        gev_vals.append(gev_crps_mean)
        sir_vals.append(sir_crps_mean)
    if not np.isnan(gev_logscore_mean):
        gev_vals.append(gev_logscore_mean)
        sir_vals.append(sir_logscore_mean)

    x = np.arange(len(metrics))
    width = 0.35
    axes[1, 0].bar(x - width / 2, gev_vals, width, alpha=0.7, label="GEV")
    axes[1, 0].bar(x + width / 2, sir_vals, width, alpha=0.7, label="SIR", color="red")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metrics)
    axes[1, 0].set_ylabel("Error / Score")
    axes[1, 0].set_title("Comparison Metrics")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3, axis="y")

    # Summary text
    axes[1, 1].axis("off")
    summary = f"""MODEL COMPARISON SUMMARY

Test Seasons: {len(actual_peaks)}

GEV Model:
  MAE:  {gev_mae:.2f}
  RMSE: {gev_rmse:.2f}
  MAPE: {gev_mape:.1f}%
  CRPS: {gev_crps_mean:.3f}
  Log Score: {gev_logscore_mean:.3f}

SIR Model:
  MAE:  {sir_mae:.2f}
  RMSE: {sir_rmse:.2f}
  MAPE: {sir_mape:.1f}%
  CRPS: {sir_crps_mean:.3f}
  Log Score: {sir_logscore_mean:.3f}

Winner: {"GEV" if gev_mae < sir_mae else "SIR"}
(lower MAE is better)
"""
    axes[1, 1].text(0.1, 0.5, summary, fontsize=12, family="monospace", verticalalignment="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "gev_mae": gev_mae,
        "gev_rmse": gev_rmse,
        "gev_mape": gev_mape,
        "sir_mae": sir_mae,
        "sir_rmse": sir_rmse,
        "sir_mape": sir_mape,
        "gev_crps": gev_crps_mean,
        "sir_crps": sir_crps_mean,
        "gev_logscore": gev_logscore_mean,
        "sir_logscore": sir_logscore_mean,
    }


def plot_data_overview(
    weekly: np.ndarray, peaks: np.ndarray, output_path: str = "data_overview.png"
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    values = weekly[:, 0]
    regions = weekly[:, 1]
    unique_regions = np.unique(regions)

    num_weeks = len(values) // len(unique_regions)

    for i, region in enumerate(unique_regions):
        region_mask = regions == region
        week_idx = np.arange(num_weeks)
        axes[0, 0].plot(
            week_idx, values[region_mask], linewidth=1, alpha=0.7, label=f"Region {int(region)}"
        )

    axes[0, 0].set_xlabel("Week")
    axes[0, 0].set_ylabel("ILI Value")
    axes[0, 0].set_title("Weekly Flu Activity by Region")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(title="Region", ncol=2)

    axes[0, 1].hist(values, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0, 1].set_xlabel("ILI Value")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of Weekly Values")
    axes[0, 1].grid(alpha=0.3, axis="y")

    axes[1, 0].plot(range(len(peaks)), peaks, "ro-", linewidth=2, markersize=8, alpha=0.7)
    axes[1, 0].axhline(
        peaks.mean(), color="g", linestyle="--", linewidth=2, label=f"Mean: {peaks.mean():.2f}"
    )
    axes[1, 0].set_xlabel("Season")
    axes[1, 0].set_ylabel("Peak ILI Value")
    axes[1, 0].set_title("Seasonal Peaks Over Time")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis("off")
    stats_text = f"""DATA SUMMARY

Weekly Data:
  n:      {len(values)}
  mean:   {values.mean():.2f}
  std:    {values.std():.2f}
  max:    {values.max():.2f}

Seasonal Peaks:
  n:      {len(peaks)}
  mean:   {peaks.mean():.2f}
  std:    {peaks.std():.2f}
  max:    {peaks.max():.2f}"""
    axes[1, 1].text(
        0.1, 0.5, stats_text, fontsize=12, family="monospace", verticalalignment="center"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_region_seasons(df_region: pd.DataFrame, output_path: str = "one_season_overview.png"):
    """
    Plots influenza incidence for a single region across multiple seasons.

    Args:
        df_region: pd.DataFrame with columns 'week' and 'cases',
                   33 weeks per season, multiple seasons concatenated.
    """
    weeks_per_season = 33
    n_seasons = df_region.shape[0] // weeks_per_season

    # Middle seasons if more than 15
    max_seasons = 15
    start_idx = max(0, (n_seasons - max_seasons) // 2)
    end_idx = start_idx + min(max_seasons, n_seasons)
    df_region = df_region.iloc[start_idx * weeks_per_season : end_idx * weeks_per_season].copy()

    n_plot_seasons = end_idx - start_idx
    seasons = range(1, n_plot_seasons + 1)

    # Dynamic grid sizing
    if n_plot_seasons <= 4:
        n_rows, n_cols = 2, 2
    elif n_plot_seasons <= 6:
        n_rows, n_cols = 2, 3
    elif n_plot_seasons <= 9:
        n_rows, n_cols = 3, 3
    elif n_plot_seasons <= 12:
        n_rows, n_cols = 3, 4
    else:
        n_rows, n_cols = 3, 5

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 3.6, n_rows * 4), sharex=True, sharey=True
    )
    if n_plot_seasons == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for i, season in enumerate(seasons):
        # Slice for this season
        season_start = i * weeks_per_season
        season_end = season_start + weeks_per_season
        df_season = df_region.iloc[season_start:season_end].copy()

        # Reset week index per season to 0..32
        df_season["season_week"] = np.arange(weeks_per_season)

        axes[i].plot(
            df_season["season_week"],
            df_season["cases"],
            color="steelblue",
            alpha=0.8,
            linewidth=1.5,
        )
        axes[i].set_title(f"Season {season}", fontsize=12)
        axes[i].grid(True, linestyle="--", alpha=0.3)

    # Hide unused subplots
    for i in range(n_plot_seasons, len(axes)):
        axes[i].set_visible(False)

    # Set shared labels
    for ax in axes[-n_cols:]:
        if ax.get_visible():
            ax.set_xlabel("Week (season index 0=week40)", fontsize=12)
    for ax in axes[::n_cols]:
        if ax.get_visible():
            ax.set_ylabel("% Weighted ILI / Cases", fontsize=12)

    plt.suptitle("Influenza Incidence by Season", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
