"""SIR model for influenza forecasting."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize


@dataclass
class SIRModel:
    """Deterministic SIR epidemic model.

    Parameters:
        beta: Transmission rate
        gamma: Recovery rate
        N: Population size
    """

    beta: float
    gamma: float
    N: float

    def _deriv(
        self,
        y: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """SIR differential equations."""
        S, I, R = y
        dSdt = -self.beta * S * I / self.N
        dIdt = self.beta * S * I / self.N - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dIdt, dRdt])

    def simulate(
        self,
        t: np.ndarray,
        I0: float = 1,
        R0: float = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate SIR dynamics.

        Args:
            t: Time points
            I0: Initial infected
            R0: Initial recovered

        Returns:
            Tuple of (S, I, R) trajectories
        """
        S0 = self.N - I0 - R0
        y0 = [S0, I0, R0]
        sol = odeint(self._deriv, y0, t)
        return sol[:, 0], sol[:, 1], sol[:, 2]


def fit_sir_to_incidence(
    incidence: np.ndarray,
    t: np.ndarray,
    N: float,
) -> SIRModel:
    """Fit SIR model to incidence data via least squares.

    Args:
        incidence: Observed incidence (infected counts)
        t: Time points
        N: Population size

    Returns:
        Fitted SIRModel

    Raises:
        ValueError: If optimization fails
    """

    def objective(params: np.ndarray) -> float:
        beta, gamma = params
        if beta <= 0 or gamma <= 0:
            return 1e10
        model = SIRModel(beta=beta, gamma=gamma, N=N)
        I0 = max(1, incidence[0])
        try:
            _, I_pred, _ = model.simulate(t=t, I0=I0, R0=0)
            return float(np.sum((incidence - I_pred) ** 2))
        except Exception:
            return 1e10

    result = minimize(
        objective,
        x0=[0.5, 0.2],
        bounds=[(0.01, 5.0), (0.01, 1.0)],
        method="L-BFGS-B",
    )

    if not result.success or np.any(result.x <= 0):
        raise ValueError("SIR fit failed")

    return SIRModel(beta=result.x[0], gamma=result.x[1], N=N)


def bootstrap_predict(
    model: SIRModel,
    incidence: np.ndarray,
    t: np.ndarray,
    n_bootstrap: int = 100,
    predict_horizon: int = 10,
    seed: int | None = None,
) -> np.ndarray:
    """Bootstrap prediction for SIR model.

    Args:
        model: Fitted SIR model
        incidence: Observed incidence
        t: Time points
        n_bootstrap: Number of bootstrap samples
        predict_horizon: Steps ahead to predict
        seed: Random seed

    Returns:
        Array of shape (n_bootstrap, predict_horizon) with predictions
    """
    rng = np.random.default_rng(seed)
    n = len(incidence)
    predictions = np.zeros((n_bootstrap, predict_horizon))

    for i in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        boot_incidence = incidence[indices]
        boot_t = t[indices]
        sort_idx = np.argsort(boot_t)
        boot_t = boot_t[sort_idx]
        boot_incidence = boot_incidence[sort_idx]

        try:
            boot_model = fit_sir_to_incidence(
                incidence=boot_incidence,
                t=boot_t,
                N=model.N,
            )
            t_future = np.linspace(t[-1], t[-1] + predict_horizon, predict_horizon)
            I0 = max(1, incidence[-1])
            _, I_pred, _ = boot_model.simulate(t=t_future, I0=I0, R0=0)
            predictions[i, :] = I_pred
        except Exception:
            predictions[i, :] = incidence[-1]

    return predictions


def predict_peak_distribution(
    model: SIRModel,
    incidence: np.ndarray,
    t: np.ndarray,
    n_samples: int = 200,
    seed: int | None = None,
) -> np.ndarray:
    """Generate distribution of peak predictions via bootstrap.

    Args:
        model: Fitted SIR model
        incidence: Observed incidence
        t: Time points
        n_samples: Number of bootstrap samples
        seed: Random seed

    Returns:
        Array of peak predictions
    """
    rng = np.random.default_rng(seed)
    n = len(incidence)
    peaks = []

    for _ in range(n_samples):
        indices = rng.choice(n, size=n, replace=True)
        boot_incidence = incidence[indices]
        boot_t = t[indices]
        sort_idx = np.argsort(boot_t)
        boot_t = boot_t[sort_idx]
        boot_incidence = boot_incidence[sort_idx]

        try:
            boot_model = fit_sir_to_incidence(
                incidence=boot_incidence,
                t=boot_t,
                N=model.N,
            )
            t_ext = np.linspace(0, 500, 5000)
            I0 = max(1, boot_incidence[0])
            _, I_pred, _ = boot_model.simulate(t=t_ext, I0=I0, R0=0)
            peak_val = float(np.max(I_pred))
            peaks.append(peak_val)
        except Exception:
            continue

    return np.array(peaks) if peaks else np.array([np.nan])


SIRModel.bootstrap_predict = bootstrap_predict
SIRModel.predict_peak_distribution = predict_peak_distribution
