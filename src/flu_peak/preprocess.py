"""Season extraction and peak detection."""

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass
class SeasonConfig:
    """Configuration for flu season boundaries."""

    start_month: int = 11
    start_day: int = 1
    end_month: int = 4
    end_day: int = 30


@dataclass
class SeasonPeak:
    """Peak information for a flu season."""

    season: int
    region: str
    peak_cases: float
    peak_week: int
    peak_year: int


def extract_season_peaks(
    df: pd.DataFrame,
    config: SeasonConfig | None = None,
) -> list[SeasonPeak]:
    """Extract one peak per season per region from CDC format data.

    Args:
        df: DataFrame with columns: YEAR, WEEK, REGION, %WEIGHTED ILI
        config: Season configuration (unused, kept for compatibility)

    Returns:
        List of SeasonPeak objects
    """
    df = df.copy()
    df["season_year"] = df.apply(
        lambda row: row["YEAR"] if row["WEEK"] < 30 else row["YEAR"] + 1, axis=1
    )

    peaks = []
    for region in df["REGION"].unique():
        region_df = df[df["REGION"] == region]
        for season in region_df["season_year"].unique():
            season_df = region_df[region_df["season_year"] == season]
            season_df = season_df[
                ((season_df["WEEK"] >= 40) & (season_df["YEAR"] == season - 1))
                | ((season_df["WEEK"] <= 20) & (season_df["YEAR"] == season))
            ]

            if len(season_df) == 0:
                continue

            peak_row = season_df.loc[season_df["%WEIGHTED ILI"].idxmax()]
            peaks.append(
                SeasonPeak(
                    season=int(season),
                    region=str(region),
                    peak_cases=float(peak_row["%WEIGHTED ILI"]),
                    peak_week=int(peak_row["WEEK"]),
                    peak_year=int(peak_row["YEAR"]),
                )
            )

    return peaks


def _old_extract_season_peaks(
    df: pd.DataFrame,
    config: SeasonConfig | None = None,
) -> list[SeasonPeak]:
    """Old date-based extraction (kept for reference)."""
    if config is None:
        config = SeasonConfig()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    peaks = []

    for region in df["region"].unique():
        region_df = df[df["region"] == region].sort_values("date")

        year_min = region_df["date"].dt.year.min()
        year_max = region_df["date"].dt.year.max()

        for year in range(year_min, year_max + 1):
            season_start = date(year, config.start_month, config.start_day)
            if config.end_month < config.start_month:
                season_end = date(year + 1, config.end_month, config.end_day)
            else:
                season_end = date(year, config.end_month, config.end_day)

            season_df = region_df[
                (region_df["date"] >= pd.Timestamp(season_start))
                & (region_df["date"] <= pd.Timestamp(season_end))
            ]

            if len(season_df) == 0:
                continue

            peak_idx = season_df["cases"].idxmax()
            peak_row = season_df.loc[peak_idx]

            season_label = f"{season_start.year}-{season_end.year}"

            peaks.append(
                SeasonPeak(
                    season=season_label,
                    region=region,
                    peak_cases=int(peak_row["cases"]),
                    peak_date=peak_row["date"].date(),
                    season_start=season_start,
                    season_end=season_end,
                )
            )

    return peaks


def split_train_test(
    peaks: list[SeasonPeak],
    test_seasons: int = 1,
) -> tuple[list[SeasonPeak], list[SeasonPeak]]:
    """Split peaks into train and test sets by season.

    Args:
        peaks: List of season peaks
        test_seasons: Number of most recent seasons to hold out

    Returns:
        Tuple of (train_peaks, test_peaks)
    """
    seasons = sorted({p.season for p in peaks})
    test_season_set = set(seasons[-test_seasons:])

    train = [p for p in peaks if p.season not in test_season_set]
    test = [p for p in peaks if p.season in test_season_set]

    return train, test
