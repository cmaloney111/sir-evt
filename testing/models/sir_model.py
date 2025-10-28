import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from typing import Tuple, Dict, Optional


class SIRModel:
    def __init__(self, population: float = 1e6):
        self.population = population
        self.beta = None
        self.gamma = None
        self.R0 = None
        self.fitted = False

    def _sir_ode(self, y: np.ndarray, t: float, beta: float, gamma: float) -> np.ndarray:
        S, I, R = y
        dS = -beta * S * I
        dI = beta * S * I - gamma * I
        dR = gamma * I
        return [dS, dI, dR]

    def fit(self, time: np.ndarray, infected: np.ndarray,
            initial_infected: Optional[float] = None) -> 'SIRModel':
        time = np.asarray(time)
        infected = np.asarray(infected)

        I0 = (initial_infected if initial_infected else infected[0]) / self.population
        S0, R0 = 1 - I0, 0.0
        infected_normalized = infected / self.population

        def objective(params):
            beta, gamma = params
            if beta <= 0 or gamma <= 0:
                return 1e10
            try:
                solution = odeint(self._sir_ode, [S0, I0, R0], time, args=(beta, gamma))
                return np.mean((solution[:, 1] - infected_normalized) ** 2)
            except:
                return 1e10

        result = minimize(objective, x0=[0.3, 0.1], method='Nelder-Mead', bounds=[(0.01, 5.0), (0.01, 1.0)])
        self.beta, self.gamma = result.x
        self.R0 = self.beta / self.gamma
        self.fitted = True
        print(f"SIR: β={self.beta:.3f}, γ={self.gamma:.3f}, R₀={self.R0:.2f}")
        return self

    def predict(self, time: np.ndarray, initial_state: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        if initial_state is None:
            initial_state = [0.999, 0.001, 0.0]
        solution = odeint(self._sir_ode, initial_state, time, args=(self.beta, self.gamma))
        return {'S': solution[:, 0] * self.population, 'I': solution[:, 1] * self.population, 'R': solution[:, 2] * self.population}

    def peak_infected(self, initial_state: Optional[np.ndarray] = None) -> Tuple[float, float]:
        if initial_state is None:
            initial_state = [0.999, 0.001, 0.0]
        time = np.linspace(0, 500, 5000)
        solution = odeint(self._sir_ode, initial_state, time, args=(self.beta, self.gamma))
        infected = solution[:, 1] * self.population
        peak_idx = np.argmax(infected)
        return infected[peak_idx], time[peak_idx]

    def goodness_of_fit(self, time: np.ndarray, infected: np.ndarray) -> Dict[str, float]:
        I0 = infected[0] / self.population
        solution = odeint(self._sir_ode, [1 - I0, I0, 0.0], time, args=(self.beta, self.gamma))
        predicted = solution[:, 1] * self.population
        residuals = infected - predicted
        rmse = np.sqrt(np.mean(residuals ** 2))
        r2 = 1 - np.sum(residuals ** 2) / np.sum((infected - np.mean(infected)) ** 2)
        print(f"  RMSE: {rmse:.2f}, R²: {r2:.3f}")
        return {'rmse': rmse, 'r2': r2}
