"""Algorithms used by the paper experiments."""

from polar_ai.algorithms import deactivation


DEACTIVATION_ALGORITHMS = {
    "No-Intervention": deactivation.no_intervention,
    "USampling-Greedy": deactivation.usampling_greedy,
    "NeighborPD-Greedy": deactivation.neighbor_pd_greedy,
    "AIS-Greedy": deactivation.ais_greedy,
    "Greedy": deactivation.greedy,
}

__all__ = ["DEACTIVATION_ALGORITHMS", "deactivation"]
