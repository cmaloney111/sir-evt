import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple


class GPDModel:
    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold
        self.scale = None
        self.shape = None
        self.fitted = False
        self.exceedance_rate = 0.0

    def fit(self, data: np.ndarray, threshold: Optional[float] = None,
            threshold_percentile: float = 90, method: str = "mle") -> 'GPDModel':
        data = np.asarray(data)
        self.threshold = threshold if threshold is not None else np.percentile(data, threshold_percentile)
        exceedances = data[data > self.threshold] - self.threshold
        self.exceedance_rate = len(exceedances) / len(data)

        if len(exceedances) < 3:
            raise ValueError(f"Only {len(exceedances)} exceedances")

        if method == "mle":
            try:
                shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
                self.shape = shape
                self.scale = scale
                if self.scale <= 0 or self.shape < -0.5:
                    raise ValueError("Invalid params")
            except:
                method = "mom"

        if method == "mom":
            mean_exc = np.mean(exceedances)
            var_exc = np.var(exceedances, ddof=1)
            cv2 = var_exc / (mean_exc ** 2)
            self.shape = 0.0 if cv2 <= 0.5 or np.isnan(cv2) else np.clip((cv2 - 1) / (2 * cv2), -0.4, 0.4)
            self.scale = max(mean_exc if self.shape == 0 else mean_exc * (1 - self.shape), 1e-6)

        self.fitted = True
        print(f"GPD: u={self.threshold:.2f}, σ={self.scale:.2f}, ξ={self.shape:.3f}")
        return self

    def predict_exceedance_probability(self, value: float) -> float:
        if value <= self.threshold:
            return self.exceedance_rate
        excess = value - self.threshold
        return self.exceedance_rate * (1 - self.cdf(excess))

    def return_level(self, return_period_weeks: float) -> float:
        p = 1 - 1 / return_period_weeks
        return self.threshold + stats.genpareto.ppf(p, c=self.shape, scale=self.scale)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return stats.genpareto.pdf(x, c=self.shape, scale=self.scale)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return stats.genpareto.cdf(x, c=self.shape, scale=self.scale)

    def goodness_of_fit(self, data: np.ndarray) -> Dict[str, float]:
        exceedances = data[data > self.threshold] - self.threshold
        ks_stat, ks_pvalue = stats.kstest(exceedances, lambda x: self.cdf(x))
        print(f"  KS: {ks_stat:.3f} (p={ks_pvalue:.3f})")
        return {'ks_statistic': ks_stat, 'ks_pvalue': ks_pvalue}

    def _mean_excess_function(self, data: np.ndarray, n_thresholds: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        data_sorted = np.sort(data)
        n = len(data_sorted)
        threshold_indices = np.linspace(int(n * 0.5), int(n * 0.95), n_thresholds).astype(int)
        thresholds = data_sorted[threshold_indices]
        mean_excesses = [np.mean(data[data > u] - u) if len(data[data > u]) > 0 else np.nan for u in thresholds]
        return thresholds, np.array(mean_excesses)
