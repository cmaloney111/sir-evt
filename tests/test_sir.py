"""Tests for SIR model fitting and prediction."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flu_peak.models.sir import SIRModel, fit_sir_to_incidence


def test_sir_model_basic() -> None:
    model = SIRModel(beta=0.5, gamma=0.1, N=10000)
    t = np.linspace(0, 100, 100)
    S, I, R = model.simulate(t=t, I0=10, R0=0)

    assert len(S) == len(t)
    assert len(I) == len(t)
    assert len(R) == len(t)
    assert np.all(S >= 0)
    assert np.all(I >= 0)
    assert np.all(R >= 0)


def test_sir_model_conservation() -> None:
    model = SIRModel(beta=0.5, gamma=0.1, N=10000)
    t = np.linspace(0, 100, 100)
    S, I, R = model.simulate(t=t, I0=10, R0=0)

    total = S + I + R
    np.testing.assert_allclose(total, model.N, rtol=1e-5)


@given(
    beta=st.floats(min_value=0.1, max_value=2.0),
    gamma=st.floats(min_value=0.05, max_value=0.5),
)
def test_sir_model_decreasing_susceptible(beta: float, gamma: float) -> None:
    model = SIRModel(beta=beta, gamma=gamma, N=10000)
    t = np.linspace(0, 100, 50)
    S, I, R = model.simulate(t=t, I0=10, R0=0)

    assert S[0] >= S[-1]


def test_fit_sir_to_incidence_parameter_recovery() -> None:
    true_beta = 0.5
    true_gamma = 0.2
    true_model = SIRModel(beta=true_beta, gamma=true_gamma, N=10000)

    t = np.linspace(0, 100, 100)
    _, I_true, _ = true_model.simulate(t=t, I0=10, R0=0)

    incidence = np.maximum(0, I_true + np.random.default_rng(42).normal(0, 5, len(I_true)))

    fitted_model = fit_sir_to_incidence(incidence=incidence, t=t, N=10000)

    assert 0.1 <= fitted_model.beta <= 1.0
    assert 0.05 <= fitted_model.gamma <= 0.5
    np.testing.assert_allclose(fitted_model.beta, true_beta, rtol=0.03)
    np.testing.assert_allclose(fitted_model.gamma, true_gamma, rtol=0.03)


def test_sir_model_bootstrap() -> None:
    model = SIRModel(beta=0.5, gamma=0.2, N=10000)
    t = np.linspace(0, 100, 100)
    _, I, _ = model.simulate(t=t, I0=10, R0=0)

    samples = model.bootstrap_predict(
        incidence=I,
        t=t,
        n_bootstrap=50,
        predict_horizon=20,
        seed=42,
    )

    assert samples.shape[0] == 50
    assert samples.shape[1] == 20
    assert np.all(samples >= 0)


@pytest.mark.slow
@given(seed=st.integers(min_value=0, max_value=100))
@settings(max_examples=100, deadline=None)
def test_sir_bootstrap_deterministic(seed: int) -> None:
    model = SIRModel(beta=0.5, gamma=0.2, N=10000)
    t = np.linspace(0, 50, 50)
    _, I, _ = model.simulate(t=t, I0=10, R0=0)

    samples1 = model.bootstrap_predict(
        incidence=I,
        t=t,
        n_bootstrap=20,
        predict_horizon=10,
        seed=seed,
    )
    samples2 = model.bootstrap_predict(
        incidence=I,
        t=t,
        n_bootstrap=20,
        predict_horizon=10,
        seed=seed,
    )

    np.testing.assert_array_equal(samples1, samples2)


def test_sir_model_peak_exists() -> None:
    model = SIRModel(beta=0.5, gamma=0.1, N=10000)
    t = np.linspace(0, 200, 200)
    _, I, _ = model.simulate(t=t, I0=10, R0=0)

    peak_idx = np.argmax(I)
    assert 0 < peak_idx < len(I) - 1


def test_predict_peak_distribution() -> None:
    """Test peak distribution prediction."""
    model = SIRModel(beta=0.5, gamma=0.2, N=10000)
    t = np.linspace(0, 100, 100)
    _, I, _ = model.simulate(t=t, I0=10, R0=0)

    peaks = model.predict_peak_distribution(
        incidence=I,
        t=t,
        n_samples=50,
        seed=42,
    )

    assert len(peaks) > 0
    assert np.all(peaks > 0)
    assert np.std(peaks) > 0


def test_predict_peak_distribution_reproducible() -> None:
    """Test peak distribution is reproducible with seed."""
    model = SIRModel(beta=0.5, gamma=0.2, N=10000)
    t = np.linspace(0, 50, 50)
    _, I, _ = model.simulate(t=t, I0=10, R0=0)

    peaks1 = model.predict_peak_distribution(incidence=I, t=t, n_samples=30, seed=42)
    peaks2 = model.predict_peak_distribution(incidence=I, t=t, n_samples=30, seed=42)

    np.testing.assert_array_equal(peaks1, peaks2)
