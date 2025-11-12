# Influenza Peak Prediction

Probabilistic forecasting of seasonal influenza peaks using GEV and SIR models with proper scoring rules.

## Installation

```bash
# Install package
pip install -e .

# For development (required for make all)
pip install -e ".[dev]"

# or using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

**Note:** The `[dev]` extras install pytest, ruff, mypy, and other development tools needed for `make all`.

## Quick Start

### CLI Usage

```bash
# Use real CDC flu data
run_experiment --data data/cdc_flu_data.csv --output results/

# With custom parameters
run_experiment --data data/cdc_flu_data.csv --output results_custom/ \
  --test-seasons 2 --n-weeks 40 --year 13

# Generate synthetic data
run_experiment --synthetic --seasons 10 --regions 5 --output results_synth/
```

**Generated plots:**
- `data_overview.png` - Time series and distributions
- `gev_fit.png` - GEV diagnostics (PDF, CDF, return levels)
- `sir_dynamics.png` - SIR trajectories and fit
- `one_season_overview.png` - Seasonal patterns
- `comparison_metrics.png` - Probabilistic comparison with IQR bands

### Python API

```python
import numpy as np
from flu_peak.models import GEVModel, fit_gev_to_peaks

# Fit GEV to seasonal peaks
peaks = np.array([4.5, 6.2, 5.1, 7.8, 9.3, 5.5, 6.1])
gev = fit_gev_to_peaks(peaks, method="mle")

# Generate probabilistic predictions
samples = gev.sample(n=1000, seed=42)
print(f"Median: {np.median(samples):.2f}")
print(f"95% CI: [{np.percentile(samples, 2.5):.2f}, {np.percentile(samples, 97.5):.2f}]")

# Return levels
print(f"10-year: {gev.return_level(10):.1f}")
print(f"100-year: {gev.return_level(100):.1f}")
```

## Models

**GEV (Generalized Extreme Value)**: Block maxima approach for seasonal peaks. Returns distributions of extreme values with shape parameter ξ controlling tail behavior.

**SIR (Susceptible-Infected-Recovered)**: Mechanistic compartmental model. Uses bootstrap resampling to generate peak distributions.

## Evaluation

Models compared using proper scoring rules:
- **CRPS** (Continuous Ranked Probability Score): Lower is better
- **Log Score**: Negative log-likelihood via KDE, lower is better

## Data Format

CDC FluView format required:
- Columns: `YEAR`, `WEEK`, `REGION`, `%WEIGHTED ILI`
- Season: Weeks 40-52 (fall) + 1-20 (winter/spring)

## Development

```bash
make test          # Run tests
make lint          # Lint code
make all           # All checks
```

## License

MIT
