"""Tests for season extraction and peak detection."""

from hypothesis import given
from hypothesis import strategies as st

from flu_peak.data import generate_synthetic_data
from flu_peak.preprocess import SeasonConfig, SeasonPeak, extract_season_peaks, split_train_test


def test_extract_season_peaks_basic() -> None:
    df = generate_synthetic_data(n_seasons=2, n_regions=2, seed=42)
    peaks = extract_season_peaks(df)
    assert len(peaks) > 0
    assert all(isinstance(p, SeasonPeak) for p in peaks)


@given(seed=st.integers(min_value=0, max_value=100))
def test_extract_season_peaks_deterministic(seed: int) -> None:
    df = generate_synthetic_data(n_seasons=1, n_regions=1, seed=seed)
    peaks1 = extract_season_peaks(df)
    peaks2 = extract_season_peaks(df)
    assert len(peaks1) == len(peaks2)
    for p1, p2 in zip(peaks1, peaks2):
        assert p1.peak_cases == p2.peak_cases


def test_peak_is_maximum_in_season() -> None:
    df = generate_synthetic_data(n_seasons=2, n_regions=1, seed=42)
    peaks = extract_season_peaks(df)

    df["season_year"] = df.apply(
        lambda row: row["YEAR"] if row["WEEK"] < 30 else row["YEAR"] + 1, axis=1
    )

    for peak in peaks:
        season_df = df[(df["REGION"] == peak.region) & (df["season_year"] == peak.season)]
        season_df = season_df[
            ((season_df["WEEK"] >= 40) & (season_df["YEAR"] == peak.season - 1))
            | ((season_df["WEEK"] <= 20) & (season_df["YEAR"] == peak.season))
        ]
        assert peak.peak_cases == season_df["%WEIGHTED ILI"].max()


def test_season_config_custom() -> None:
    config = SeasonConfig(start_month=10, start_day=15, end_month=5, end_day=15)
    assert config.start_month == 10
    assert config.end_month == 5


def test_extract_season_peaks_multiple_regions() -> None:
    df = generate_synthetic_data(n_seasons=2, n_regions=3, seed=42)
    peaks = extract_season_peaks(df)
    regions = {p.region for p in peaks}
    assert len(regions) <= 3


def test_split_train_test_basic() -> None:
    df = generate_synthetic_data(n_seasons=5, n_regions=2, seed=42)
    peaks = extract_season_peaks(df)
    train, test = split_train_test(peaks, test_seasons=1)

    assert len(train) > 0
    assert len(test) > 0
    assert len(train) + len(test) == len(peaks)


def test_split_train_test_no_overlap() -> None:
    df = generate_synthetic_data(n_seasons=5, n_regions=2, seed=42)
    peaks = extract_season_peaks(df)
    train, test = split_train_test(peaks, test_seasons=1)

    train_seasons = {p.season for p in train}
    test_seasons = {p.season for p in test}
    assert len(train_seasons & test_seasons) == 0


def test_split_preserves_all_peaks() -> None:
    df = generate_synthetic_data(n_seasons=4, n_regions=2, seed=42)
    peaks = extract_season_peaks(df)
    train, test = split_train_test(peaks, test_seasons=1)

    all_peaks_sorted = sorted(peaks, key=lambda p: (p.season, p.region))
    train_test_sorted = sorted(train + test, key=lambda p: (p.season, p.region))

    assert len(all_peaks_sorted) == len(train_test_sorted)
