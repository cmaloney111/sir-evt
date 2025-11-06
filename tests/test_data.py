"""Tests for data loading and synthetic generation."""

from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flu_peak.data import generate_synthetic_data, load_flu_data


@given(
    n_seasons=st.integers(min_value=1, max_value=10),
    n_regions=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100, deadline=None)
def test_synthetic_data_shape(n_seasons: int, n_regions: int, seed: int) -> None:
    df = generate_synthetic_data(n_seasons=n_seasons, n_regions=n_regions, seed=seed)
    assert len(df) > 0
    assert set(df.columns) == {"YEAR", "WEEK", "REGION", "%WEIGHTED ILI"}
    assert df["REGION"].nunique() == n_regions
    assert df["%WEIGHTED ILI"].min() >= 0


@given(seed=st.integers(min_value=0, max_value=1000))
def test_synthetic_data_deterministic(seed: int) -> None:
    df1 = generate_synthetic_data(n_seasons=2, n_regions=2, seed=seed)
    df2 = generate_synthetic_data(n_seasons=2, n_regions=2, seed=seed)
    pd.testing.assert_frame_equal(df1, df2)


def test_synthetic_data_weeks_valid() -> None:
    df = generate_synthetic_data(n_seasons=2, n_regions=1, seed=42)
    assert df["WEEK"].min() >= 1
    assert df["WEEK"].max() <= 52


def test_load_flu_data_valid(tmp_path: Path) -> None:
    csv_path = tmp_path / "test.csv"
    df = generate_synthetic_data(n_seasons=1, n_regions=1, seed=0)
    df.to_csv(csv_path, index=False)

    loaded = load_flu_data(csv_path)
    assert set(loaded.columns) == {"YEAR", "WEEK", "REGION", "%WEIGHTED ILI"}
    assert len(loaded) == len(df)


def test_load_flu_data_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    bad_df = pd.DataFrame({"YEAR": [2020], "WEEK": [1]})
    bad_df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_flu_data(csv_path)


def test_synthetic_data_positive_ili() -> None:
    df = generate_synthetic_data(n_seasons=1, n_regions=1, seed=42)
    assert (df["%WEIGHTED ILI"] > 0).all()
