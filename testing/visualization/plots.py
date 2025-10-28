import numpy as np
import matplotlib.pyplot as plt


def plot_gev_fit(gev_model, data: np.ndarray, output_path: str = "gev_fit.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(data, bins=15, density=True, alpha=0.7, edgecolor='black', label='Data')
    x = np.linspace(data.min() * 0.8, data.max() * 1.2, 200)
    axes[0, 0].plot(x, gev_model.pdf(x), 'r-', linewidth=2, label='GEV fit')
    axes[0, 0].set_xlabel('Peak Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('GEV Fit to Seasonal Peaks')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    data_sorted = np.sort(data)
    n = len(data_sorted)
    p = (np.arange(1, n + 1) - 0.5) / n
    theo = gev_model.cdf(data_sorted)
    axes[0, 1].scatter(p, theo, alpha=0.6, s=30)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect fit')
    axes[0, 1].set_xlabel('Empirical Probability')
    axes[0, 1].set_ylabel('Theoretical Probability')
    axes[0, 1].set_title('Probability-Probability Plot')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    rp = np.array([2, 5, 10, 20, 50, 100])
    rl = [gev_model.return_level(r) for r in rp]
    axes[1, 0].semilogx(rp, rl, 'bo-', linewidth=2, markersize=8, label='GEV return levels')
    axes[1, 0].axhline(data.max(), color='r', linestyle='--', linewidth=2, label=f'Observed max: {data.max():.1f}')
    axes[1, 0].set_xlabel('Return Period (years)')
    axes[1, 0].set_ylabel('Return Level')
    axes[1, 0].set_title('Return Level Plot')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(data_sorted, p, 'o', alpha=0.6, label='Empirical CDF', markersize=5)
    axes[1, 1].plot(data_sorted, gev_model.cdf(data_sorted), 'r-', linewidth=2, label='GEV CDF')
    axes[1, 1].set_xlabel('Peak Value')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('CDF Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_gpd_fit(gpd_model, data: np.ndarray, output_path: str = "gpd_fit.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    exceedances = data[data > gpd_model.threshold] - gpd_model.threshold

    axes[0, 0].hist(exceedances, bins=20, density=True, alpha=0.7, edgecolor='black', label='Exceedances')
    x = np.linspace(0, exceedances.max() * 1.1, 200)
    axes[0, 0].plot(x, gpd_model.pdf(x), 'r-', linewidth=2, label='GPD fit')
    axes[0, 0].set_xlabel('Exceedance')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title(f'GPD Fit (threshold={gpd_model.threshold:.2f})')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    thresholds, mean_exc = gpd_model._mean_excess_function(data)
    axes[0, 1].plot(thresholds, mean_exc, 'bo-', linewidth=2, markersize=6)
    axes[0, 1].axvline(gpd_model.threshold, color='r', linestyle='--', linewidth=2, label=f'u={gpd_model.threshold:.2f}')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Mean Excess')
    axes[0, 1].set_title('Mean Excess Plot (should be linear)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    values = np.linspace(gpd_model.threshold, data.max() * 1.2, 50)
    probs = [gpd_model.predict_exceedance_probability(v) for v in values]
    axes[1, 0].semilogy(values, probs, 'b-', linewidth=2)
    axes[1, 0].axvline(gpd_model.threshold, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('P(exceed value)')
    axes[1, 0].set_title('Exceedance Probability (log scale)')
    axes[1, 0].grid(alpha=0.3, which='both')

    exc_sorted = np.sort(exceedances)
    n = len(exc_sorted)
    p = (np.arange(1, n + 1) - 0.5) / n
    theo = gpd_model.cdf(exc_sorted)
    axes[1, 1].scatter(p, theo, alpha=0.6, s=30)
    axes[1, 1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect fit')
    axes[1, 1].set_xlabel('Empirical Probability')
    axes[1, 1].set_ylabel('Theoretical Probability')
    axes[1, 1].set_title('Probability-Probability Plot')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sir_dynamics(sir_model, time: np.ndarray, observed: np.ndarray, output_path: str = "sir_dynamics.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    I0 = observed[0] / sir_model.population
    initial = [1 - I0, I0, 0.0]

    extended_time = np.linspace(0, max(time) * 2, 200)
    pred = sir_model.predict(extended_time, initial)

    axes[0, 0].plot(extended_time, pred['S'] / sir_model.population, 'b-', linewidth=2, label='S')
    axes[0, 0].plot(extended_time, pred['I'] / sir_model.population, 'r-', linewidth=2, label='I')
    axes[0, 0].plot(extended_time, pred['R'] / sir_model.population, 'g-', linewidth=2, label='R')
    axes[0, 0].set_xlabel('Time (weeks)')
    axes[0, 0].set_ylabel('Proportion')
    axes[0, 0].set_title(f'SIR Dynamics (R₀={sir_model.R0:.2f})')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    predicted = sir_model.predict(time, initial)['I']
    axes[0, 1].plot(time, observed, 'ko', markersize=8, alpha=0.6, label='Observed')
    axes[0, 1].plot(time, predicted, 'r-', linewidth=2, label='SIR fit')
    axes[0, 1].set_xlabel('Time (weeks)')
    axes[0, 1].set_ylabel('Infected')
    axes[0, 1].set_title('Model Fit to Data')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    residuals = observed - predicted
    axes[1, 0].plot(time, residuals, 'bo-', linewidth=2, markersize=6)
    axes[1, 0].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Time (weeks)')
    axes[1, 0].set_ylabel('Residual (Obs - Pred)')
    axes[1, 0].set_title(f'Residuals (RMSE={np.sqrt(np.mean(residuals**2)):.2f})')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(pred['S'] / sir_model.population, pred['I'] / sir_model.population, 'b-', linewidth=2)
    axes[1, 1].plot((pred['S'] / sir_model.population)[0], (pred['I'] / sir_model.population)[0], 'go', markersize=10, label='Start')
    peak_idx = np.argmax(pred['I'])
    axes[1, 1].plot((pred['S'] / sir_model.population)[peak_idx], (pred['I'] / sir_model.population)[peak_idx], 'r*', markersize=15, label='Peak')
    axes[1, 1].set_xlabel('S (proportion)')
    axes[1, 1].set_ylabel('I (proportion)')
    axes[1, 1].set_title('Phase Portrait (S-I plane)')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_metrics(gev_predictions, sir_predictions, actual_peaks, output_path: str = "comparison_metrics.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    test_indices = range(len(actual_peaks))

    axes[0, 0].plot(test_indices, actual_peaks, 'ko-', linewidth=2, markersize=8, label='Actual', alpha=0.7)
    axes[0, 0].plot(test_indices, gev_predictions, 'bs-', linewidth=2, markersize=6, label='GEV', alpha=0.7)
    axes[0, 0].plot(test_indices, sir_predictions, 'r^-', linewidth=2, markersize=6, label='SIR', alpha=0.7)
    axes[0, 0].set_xlabel('Test Season')
    axes[0, 0].set_ylabel('Peak Value')
    axes[0, 0].set_title('Predictions vs Actual')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    gev_errors = np.abs(gev_predictions - actual_peaks)
    sir_errors = np.abs(sir_predictions - actual_peaks)
    axes[0, 1].bar(test_indices, gev_errors, width=0.4, alpha=0.7, label='GEV', align='edge')
    axes[0, 1].bar(np.array(test_indices) + 0.4, sir_errors, width=0.4, alpha=0.7, label='SIR', color='red', align='edge')
    axes[0, 1].set_xlabel('Test Season')
    axes[0, 1].set_ylabel('Absolute Error')
    axes[0, 1].set_title('Prediction Errors by Season')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    metrics = ['MAE', 'RMSE', 'MAPE (%)']
    gev_mae = np.mean(gev_errors)
    sir_mae = np.mean(sir_errors)
    gev_rmse = np.sqrt(np.mean((gev_predictions - actual_peaks) ** 2))
    sir_rmse = np.sqrt(np.mean((sir_predictions - actual_peaks) ** 2))
    gev_mape = np.mean(np.abs((actual_peaks - gev_predictions) / actual_peaks)) * 100
    sir_mape = np.mean(np.abs((actual_peaks - sir_predictions) / actual_peaks)) * 100

    gev_vals = [gev_mae, gev_rmse, gev_mape]
    sir_vals = [sir_mae, sir_rmse, sir_mape]

    x = np.arange(len(metrics))
    width = 0.35
    axes[1, 0].bar(x - width/2, gev_vals, width, alpha=0.7, label='GEV')
    axes[1, 0].bar(x + width/2, sir_vals, width, alpha=0.7, label='SIR', color='red')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metrics)
    axes[1, 0].set_ylabel('Error')
    axes[1, 0].set_title('Comparison Metrics')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3, axis='y')

    axes[1, 1].axis('off')
    summary = f"""MODEL COMPARISON SUMMARY

Test Seasons: {len(actual_peaks)}

GEV Model:
  MAE:  {gev_mae:.2f}
  RMSE: {gev_rmse:.2f}
  MAPE: {gev_mape:.1f}%

SIR Model:
  MAE:  {sir_mae:.2f}
  RMSE: {sir_rmse:.2f}
  MAPE: {sir_mape:.1f}%

Winner: {'GEV' if gev_mae < sir_mae else 'SIR'}
(lower MAE is better)
"""
    axes[1, 1].text(0.1, 0.5, summary, fontsize=12, family='monospace', verticalalignment='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'gev_mae': gev_mae, 'gev_rmse': gev_rmse, 'gev_mape': gev_mape,
        'sir_mae': sir_mae, 'sir_rmse': sir_rmse, 'sir_mape': sir_mape
    }


def plot_data_overview(weekly: np.ndarray, peaks: np.ndarray, output_path: str = "data_overview.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(weekly, 'b-', linewidth=1, alpha=0.7)
    axes[0, 0].set_xlabel('Week')
    axes[0, 0].set_ylabel('ILI Value')
    axes[0, 0].set_title('Weekly Flu Activity')
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].hist(weekly, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 1].set_xlabel('ILI Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Weekly Values')
    axes[0, 1].grid(alpha=0.3, axis='y')

    axes[1, 0].plot(range(len(peaks)), peaks, 'ro-', linewidth=2, markersize=8, alpha=0.7)
    axes[1, 0].axhline(peaks.mean(), color='g', linestyle='--', linewidth=2, label=f'Mean: {peaks.mean():.2f}')
    axes[1, 0].set_xlabel('Season')
    axes[1, 0].set_ylabel('Peak ILI Value')
    axes[1, 0].set_title('Seasonal Peaks Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')
    stats_text = f"""DATA SUMMARY

Weekly Data:
  n:      {len(weekly)}
  mean:   {weekly.mean():.2f}
  std:    {weekly.std():.2f}
  max:    {weekly.max():.2f}

Seasonal Peaks:
  n:      {len(peaks)}
  mean:   {peaks.mean():.2f}
  std:    {peaks.std():.2f}
  max:    {peaks.max():.2f}"""
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace', verticalalignment='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
