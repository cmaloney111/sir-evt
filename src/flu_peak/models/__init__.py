"""Predictive models for flu peaks."""

from flu_peak.models.gev import GEVModel, fit_gev_to_peaks
from flu_peak.models.sir import SIRModel, fit_sir_to_incidence

__all__ = [
    "GEVModel",
    "SIRModel",
    "fit_gev_to_peaks",
    "fit_sir_to_incidence",
]
