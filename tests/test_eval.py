"""Tests for evaluation metrics."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from flu_peak.eval import (
    brier_score,
    coverage,
    crps,
    log_score,
    pit_histogram,
)


def test_crps_deterministic() -> None:
    samples = np.array([10.0, 10.0, 10.0])
    observation = 10.0

    score = crps(samples=samples, observation=observation)

    assert score >= 0
    np.testing.assert_allclose(score, 0, atol=1e-10)


@given(
    mean=st.floats(min_value=0, max_value=100),
    std=st.floats(min_value=0.1, max_value=10),
)
def test_crps_positive(mean: float, std: float) -> None:
    samples = np.random.default_rng(42).normal(mean, std, 100)
    observation = mean

    score = crps(samples=samples, observation=observation)

    assert score >= 0


def test_log_score_normal_distribution() -> None:
    samples = np.random.default_rng(42).normal(50, 10, 1000)
    observation = 50.0

    score = log_score(samples=samples, observation=observation)

    assert np.isfinite(score)


def test_log_score_extreme_observation() -> None:
    samples = np.random.default_rng(42).normal(50, 10, 1000)
    observation = 150.0

    score = log_score(samples=samples, observation=observation)

    assert score < 0
    assert np.isfinite(score)


@given(
    lower=st.floats(min_value=0, max_value=50),
    upper=st.floats(min_value=51, max_value=100),
)
def test_coverage_within_interval(lower: float, upper: float) -> None:
    observation = (lower + upper) / 2
    cov = coverage(lower=lower, upper=upper, observation=observation)

    assert cov == 1.0


@given(
    lower=st.floats(min_value=10, max_value=50),
    upper=st.floats(min_value=51, max_value=100),
)
def test_coverage_outside_interval(lower: float, upper: float) -> None:
    observation = upper + 10
    cov = coverage(lower=lower, upper=upper, observation=observation)

    assert cov == 0.0


def test_brier_score_perfect_prediction() -> None:
    forecast_prob = 1.0
    observed = True

    score = brier_score(forecast_prob=forecast_prob, observed=observed)

    np.testing.assert_allclose(score, 0, atol=1e-10)


def test_brier_score_worst_prediction() -> None:
    forecast_prob = 0.0
    observed = True

    score = brier_score(forecast_prob=forecast_prob, observed=observed)

    np.testing.assert_allclose(score, 1.0, atol=1e-10)


@given(
    forecast_prob=st.floats(min_value=0, max_value=1),
    observed=st.booleans(),
)
def test_brier_score_bounds(forecast_prob: float, observed: bool) -> None:
    score = brier_score(forecast_prob=forecast_prob, observed=observed)

    assert 0 <= score <= 1


def test_pit_histogram_uniform() -> None:
    #
    samples = np.random.default_rng(42).uniform(0, 100, 1000)
    observations = np.random.default_rng(43).uniform(0, 100, 50)

    hist, bins = pit_histogram(samples_list=[samples] * 50, observations=observations, bins=10)

    assert len(hist) == 10
    assert len(bins) == 11
    assert np.all(hist >= 0)
    np.testing.assert_allclose(np.sum(hist), 50)


@given(n_obs=st.integers(min_value=10, max_value=100))
def test_pit_histogram_shape(n_obs: int) -> None:
    samples = np.random.default_rng(42).normal(50, 10, 100)
    observations = np.random.default_rng(43).normal(50, 10, n_obs)

    hist, bins = pit_histogram(
        samples_list=[samples] * n_obs,
        observations=observations,
        bins=5,
    )

    assert len(hist) == 5
    assert len(bins) == 6


def test_crps_ensemble() -> None:
    #
    samples = np.array([8, 9, 10, 11, 12])
    observation = 10.0

    score = crps(samples=samples, observation=observation)

    assert 0 < score < 5


def test_log_score_with_bandwidth() -> None:
    samples = np.array([9.0, 10.0, 11.0])
    observation = 10.0

    score = log_score(samples=samples, observation=observation, bandwidth=0.5)

    assert np.isfinite(score)
    assert score < 0


def test_log_score_constant_samples() -> None:
    """Test log_score with constant samples (no variance)."""
    samples = np.array([10.0, 10.0, 10.0, 10.0])
    observation = 10.0

    score = log_score(samples=samples, observation=observation)

    assert score == 0.0


def test_log_score_constant_samples_miss() -> None:
    """Test log_score with constant samples and observation miss."""
    samples = np.array([10.0, 10.0, 10.0, 10.0])
    observation = 15.0

    score = log_score(samples=samples, observation=observation)

    assert score == -np.inf


def test_log_score_kde_exception() -> None:
    """Test log_score with data that might cause KDE exception."""
    samples = np.array([1e-15, 1e-14, 1e-13])
    observation = 1e-12

    score = log_score(samples=samples, observation=observation)

    # Should handle exception gracefully
    assert score == -np.inf or np.isfinite(score)


def test_log_score_zero_density() -> None:
    """Test log_score when density evaluates to zero."""
    samples = np.random.default_rng(42).normal(100, 0.1, 100)
    observation = 0.0  # Very far from the distribution

    score = log_score(samples=samples, observation=observation)

    assert score == -np.inf or score < -100
