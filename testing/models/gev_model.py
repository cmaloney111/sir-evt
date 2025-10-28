import numpy as np
from scipy import stats
from typing import Dict, Optional


class GEVModel:
    def __init__(self):
        self.location = None
        self.scale = None
        self.shape = None
        self.fitted = False

    def fit(self, data: np.ndarray, method: str = "mle") -> 'GEVModel':
        data = np.asarray(data)
        if len(data) < 3:
            raise ValueError(f"Need ≥3 samples")

        if method == "mle":
            try:
                shape_scipy, loc, scale = stats.genextreme.fit(data)
                self.shape = -shape_scipy
                self.location = loc
                self.scale = scale
                if self.scale <= 0:
                    raise ValueError("Invalid scale")
            except:
                method = "mom"

        if method == "mom":
            mean, std, skew = np.mean(data), np.std(data, ddof=1), stats.skew(data)
            self.shape = 0.0 if abs(skew) < 0.01 else np.clip(np.sign(skew) * min(abs(skew) / 3, 0.5), -0.5, 0.5)
            self.scale = abs(std * np.sqrt(6) / np.pi if self.shape == 0 else std / np.sqrt((1 - 2 * (1 + self.shape) ** (-2))))
            self.location = mean - 0.5772 * self.scale if self.shape == 0 else mean - self.scale * ((1 - (1 + self.shape) ** (-1)))

        self.fitted = True
        print(f"GEV: μ={self.location:.2f}, σ={self.scale:.2f}, ξ={self.shape:.3f}")
        return self

    def return_level(self, return_period: float) -> float:
        p = 1 - 1 / return_period
        return float(stats.genextreme.ppf(p, c=-self.shape, loc=self.location, scale=self.scale))

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return stats.genextreme.pdf(x, c=-self.shape, loc=self.location, scale=self.scale)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return stats.genextreme.cdf(x, c=-self.shape, loc=self.location, scale=self.scale)

    def goodness_of_fit(self, data: np.ndarray) -> Dict[str, float]:
        ks_stat, ks_pvalue = stats.kstest(data, lambda x: self.cdf(x))
        print(f"  KS: {ks_stat:.3f} (p={ks_pvalue:.3f})")
        return {'ks_statistic': ks_stat, 'ks_pvalue': ks_pvalue}
