"""Evaluation metrics for predictive distributions."""

import numpy as np
from scipy import stats


def crps(samples: np.ndarray, observation: float) -> float:
    """Continuous Ranked Probability Score.

    Args:
        samples: Predictive samples
        observation: Observed value

    Returns:
        CRPS (lower is better)
    """
    samples_sorted = np.sort(samples)
    n = len(samples_sorted)

    term1 = np.mean(np.abs(samples_sorted - observation))
    term2 = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            term2 += np.abs(samples_sorted[i] - samples_sorted[j])
    term2 = term2 / (n * (n - 1) / 2)

    return float(term1 - 0.5 * term2)


def log_score(
    samples: np.ndarray,
    observation: float,
    bandwidth: float | None = None,
) -> float:
    """Negative log-likelihood score using kernel density estimation.

    Args:
        samples: Predictive samples
        observation: Observed value
        bandwidth: KDE bandwidth (auto if None)

    Returns:
        Log score (higher is better)
    """
    if len(np.unique(samples)) < 2 or np.std(samples) < 1e-10:
        return -np.inf if observation not in samples else 0.0

    if bandwidth is None:
        bandwidth = max(1e-6, 1.06 * np.std(samples) * len(samples) ** (-1 / 5))

    try:
        kde = stats.gaussian_kde(samples, bw_method=bandwidth / max(1e-10, np.std(samples)))
        density = kde.evaluate([observation])[0]
    except Exception:
        return -np.inf

    if density <= 0:
        return -np.inf

    return float(np.log(density))


def coverage(lower: float, upper: float, observation: float) -> float:
    """Check if observation falls within prediction interval.

    Args:
        lower: Lower bound of interval
        upper: Upper bound of interval
        observation: Observed value

    Returns:
        1.0 if covered, 0.0 otherwise
    """
    return float(lower <= observation <= upper)


def brier_score(forecast_prob: float, observed: bool) -> float:
    """Brier score for binary outcomes.

    Args:
        forecast_prob: Forecast probability [0, 1]
        observed: Observed outcome

    Returns:
        Brier score (lower is better)
    """
    return float((forecast_prob - float(observed)) ** 2)


def pit_histogram(
    samples_list: list[np.ndarray],
    observations: np.ndarray,
    bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Probability Integral Transform histogram for calibration check.

    Args:
        samples_list: List of predictive sample arrays (one per observation)
        observations: Array of observed values
        bins: Number of histogram bins

    Returns:
        Tuple of (histogram counts, bin edges)
    """
    pit_values = []

    for samples, obs in zip(samples_list, observations):
        pit = np.mean(samples <= obs)
        pit_values.append(pit)

    hist, bin_edges = np.histogram(pit_values, bins=bins, range=(0, 1))

    return hist, bin_edges
