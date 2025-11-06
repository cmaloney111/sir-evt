"""Data loading and synthetic data generation.

Example usage:
    >>> import tempfile
    >>> from pathlib import Path
    >>> df = generate_synthetic_data(n_seasons=1, n_regions=1, seed=42)
    >>> len(df) > 0
    True
    >>> set(df.columns) == {"date", "region", "cases", "tests"}
    True
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_data(
    n_seasons: int = 3,
    n_regions: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic influenza surveillance data in CDC format.

    Args:
        n_seasons: Number of flu seasons to generate
        n_regions: Number of regions
        seed: Random seed for reproducibility

    Returns:
        DataFrame with CDC-style columns: YEAR, WEEK, REGION, %WEIGHTED ILI
    """
    rng = np.random.default_rng(seed)
    records = []

    for season_idx in range(n_seasons):
        start_year = 2015 + season_idx

        for region_idx in range(n_regions):
            region = f"Region {region_idx + 1}"
            peak_week_in_season = rng.integers(10, 23)

            # Generate peak from Gumbel distribution for heavier tails
            baseline = rng.uniform(1.5, 2.5)
            peak_height = rng.gumbel(5.0, 1.5)  # Gumbel with location=5, scale=1.5
            peak_height = np.clip(peak_height, 3.0, 10.0)  # Keep reasonable bounds

            for week_in_season in range(33):
                if week_in_season < 13:
                    year = start_year
                    week = 40 + week_in_season
                else:
                    year = start_year + 1
                    week = week_in_season - 13 + 1

                # Asymmetric peak shape (steeper decline after peak)
                distance_to_peak = week_in_season - peak_week_in_season
                if distance_to_peak < 0:
                    # Before peak: gradual rise
                    peak_effect = np.exp(-(distance_to_peak**2) / (2 * 5**2))
                else:
                    # After peak: faster decline
                    peak_effect = np.exp(-(distance_to_peak**2) / (2 * 3**2))

                # Add noise with occasional outliers (Gumbel-like)
                noise = rng.normal(0, 0.2) if rng.random() > 0.1 else rng.gumbel(0, 0.3)
                ili = baseline + peak_height * peak_effect + noise
                ili = max(0.5, ili)

                records.append(
                    {
                        "YEAR": year,
                        "WEEK": week,
                        "REGION": region,
                        "%WEIGHTED ILI": round(ili, 2),
                    }
                )

    return pd.DataFrame(records)


def load_flu_data(csv_path: Path | str) -> pd.DataFrame:
    """Load influenza data from CSV in CDC format.

    Args:
        csv_path: Path to CSV file with columns: YEAR, WEEK, REGION, %WEIGHTED ILI

    Returns:
        DataFrame with validated data

    Raises:
        ValueError: If required columns are missing
    """
    df = pd.read_csv(csv_path)

    required = {"YEAR", "WEEK", "REGION", "%WEIGHTED ILI"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    return df
