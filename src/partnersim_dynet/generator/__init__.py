"""The partnership generator: core simulation engine."""

from partnersim_dynet.generator.concurrency import select_concurrent_indices
from partnersim_dynet.generator.core import PartnershipGenerator
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
    "select_concurrent_indices",
    "PartnershipGenerator",
]

__version__ = "0.1.0"
