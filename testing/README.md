# EVT Analysis of CDC Flu Data

Extreme Value Theory models for influenza surveillance data.

## Models

- **GEV**: Seasonal peak distribution (block maxima)
- **GPD**: Weekly extreme value probabilities (peaks-over-threshold)
- **SIR**: Epidemiological baseline

## Usage

```bash
pip install -r requirements.txt

python run_analysis.py
```

## Output

Generates 5 plots in `outputs/`:
- `data_overview.png` - Time series and distributions
- `gev_fit.png` - GEV diagnostics (Q-Q plot, return levels)
- `gpd_fit.png` - GPD diagnostics (exceedances, mean excess)
- `sir_dynamics.png` - SIR epidemic curves
- `model_comparison.png` - All models compared

## Results on Real CDC Data (2010-2024)

**GEV Model**:
- Shape ξ = -0.040 (near-Gumbel)
- 100-year return level: 12.7% ILI
- KS p-value: 0.958

**GPD Model**:
- Shape ξ = -0.118 (bounded tail)
- 10-year return level: 13.4% ILI
- KS p-value: 0.919

## Model Details

**GEV**: F(x) = exp(-(1 + ξ(x-μ)/σ)^(-1/ξ))
- μ = location, σ = scale, ξ = shape
- ξ > 0: heavy tail, ξ = 0: Gumbel, ξ < 0: bounded

**GPD**: F(x) = 1 - (1 + ξx/σ)^(-1/ξ)
- Fits exceedances above threshold u
- Models P(X > x | X > u)

**SIR**: dS/dt = -βSI/N, dI/dt = βSI/N - γI, dR/dt = γI
- β = transmission rate, γ = recovery rate
- R₀ = β/γ (basic reproduction number)

## Interpretation

**Return Levels**: N-year return level = value exceeded on average once per N years

**Goodness of Fit**: KS test p > 0.05 = good fit (cannot reject model)

**Tail Types**:
- Heavy (ξ > 0.1): No upper bound, severe outliers possible
- Medium (|ξ| < 0.1): Gumbel-like, moderate extremes
- Bounded (ξ < -0.1): Finite maximum exists

## Data Source

CDC FluView via Delphi Epidata API: https://api.delphi.cmu.edu/epidata/
