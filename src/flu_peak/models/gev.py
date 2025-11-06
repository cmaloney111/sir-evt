"""Generalized Extreme Value (GEV) model for block maxima."""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class GEVModel:
    """Generalized Extreme Value distribution for seasonal peak maxima.

    Parameters:
        mu: Location parameter
        sigma: Scale parameter
        xi: Shape parameter (tail index)
    """

    mu: float
    sigma: float
    xi: float

    def return_level(self, return_period: float) -> float:
        """Compute return level for given return period.

        Args:
            return_period: Return period (e.g., 10 for 10-year return level)

        Returns:
            Return level value
        """
        p = 1 - 1 / return_period
        return float(stats.genextreme.ppf(p, c=-self.xi, loc=self.mu, scale=self.sigma))

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function.

        Args:
            x: Values to evaluate

        Returns:
            PDF values
        """
        return stats.genextreme.pdf(x, c=-self.xi, loc=self.mu, scale=self.sigma)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function.

        Args:
            x: Values to evaluate

        Returns:
            CDF values
        """
        return stats.genextreme.cdf(x, c=-self.xi, loc=self.mu, scale=self.sigma)

    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        """Generate samples from the distribution.

        Args:
            n: Number of samples
            seed: Random seed

        Returns:
            Array of samples
        """
        return stats.genextreme.rvs(
            c=-self.xi, loc=self.mu, scale=self.sigma, size=n, random_state=seed
        )


def _fit_gev_mom(data: np.ndarray) -> tuple[float, float, float]:
    """Fit GEV using Method of Moments.

    Args:
        data: Block maxima

    Returns:
        Tuple of (mu, sigma, xi)
    """
    mean, std = float(np.mean(data)), float(np.std(data, ddof=1))
    skew = float(stats.skew(data))

    xi = 0.0 if abs(skew) < 0.01 else np.sign(skew) * min(abs(skew) / 3, 0.5)
    xi = float(np.clip(xi, -0.5, 0.5))

    if abs(xi) < 0.01:
        sigma = std * np.sqrt(6) / np.pi
        mu = mean - 0.5772 * sigma
    else:
        from math import gamma as gamma_fn

        gamma_1 = float(gamma_fn(1 - xi))
        gamma_2 = float(gamma_fn(1 - 2 * xi))
        sigma = std / np.sqrt(gamma_2 - gamma_1**2)
        mu = mean - sigma * (gamma_1 - 1) / xi

    return float(mu), max(sigma, 1e-6), float(xi)


def fit_gev_to_peaks(
    peaks: np.ndarray,
    method: str = "mle",
    use_mom_fallback: bool = True,
) -> GEVModel:
    """Fit GEV to block maxima using maximum likelihood.

    Args:
        peaks: Block maxima (e.g., seasonal peaks)
        method: Fitting method ('mle' or 'mom')
        use_mom_fallback: Use Method of Moments if MLE fails

    Returns:
        Fitted GEVModel

    Raises:
        ValueError: If fit fails
    """
    if len(peaks) < 3:
        raise ValueError("Need at least 3 peaks for GEV fitting")

    if method == "mom":
        mu, sigma, xi = _fit_gev_mom(peaks)
        return GEVModel(mu=mu, sigma=sigma, xi=xi)

    try:
        shape_scipy, loc, scale = stats.genextreme.fit(peaks)
        xi = -shape_scipy

        if scale > 0 and -0.5 <= xi <= 0.5:
            return GEVModel(mu=loc, sigma=scale, xi=xi)

        if use_mom_fallback:
            mu, sigma, xi = _fit_gev_mom(peaks)
            return GEVModel(mu=mu, sigma=sigma, xi=xi)

        raise ValueError(f"Invalid GEV parameters: mu={loc:.3f}, sigma={scale:.3f}, xi={xi:.3f}")

    except (ValueError, RuntimeError) as e:
        if use_mom_fallback:
            try:
                mu, sigma, xi = _fit_gev_mom(peaks)
                return GEVModel(mu=mu, sigma=sigma, xi=xi)
            except Exception as mom_e:
                raise RuntimeError(f"Both MLE and MOM failed: MLE={e}, MOM={mom_e}") from mom_e
        raise RuntimeError(f"GEV fit failed: {e}") from e
