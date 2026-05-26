"""The partnership generator: core simulation engine."""

from partnersim_dynet.generator.kernels import (
    compute_breakage_events,
    compute_failure_rates,
    fast_digitise_age_group,
    fast_normal_pdf,
)
from partnersim_dynet.generator.records import PartnershipRecord

__all__ = [
    "PartnershipRecord",
    "compute_breakage_events",
    "compute_failure_rates",
    "fast_digitise_age_group",
    "fast_normal_pdf",
]
