"""
harness/cost_accounting.py
---------------------------
The audit's core metric: amortized total cost per generated token, as a
function of query volume N, for methods with different cost structures:

  Trained router (LayerRoute):
      total_cost(N) = train_cost_once + N * pure_inference_cost_per_query
      (search_overhead = 0 by construction)

  Heuristic/online-search (ConfLayers, SWIFT):
      total_cost(N) = 0 + N * (pure_inference_cost_per_query + search_overhead_per_query)

Prior papers (ConfLayers, SWIFT, KnapSpec, etc.) report only a single-query
wall-clock speedup, which implicitly assumes N=1 and folds search overhead
into "inference time" -- silently favoring training-free methods regardless
of deployment volume. This module makes the volume-dependence explicit and
finds the crossover N* (if any) where ranking flips.
"""

from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class MethodCostProfile:
    """Calibrated cost profile for one method on one model, from measurement."""
    name: str
    model_id: str
    train_cost_seconds: float          # one-time; 0 for training-free methods
    pure_inference_ms_per_query: float  # mean, from frozen-replay measurement
    search_overhead_ms_per_query: float  # mean, from frozen-replay measurement (0 for trained router)
    baseline_ms_per_query: float        # vanilla (no skipping) generation time, for speedup reporting
    quality_score: float                # e.g. exact-match or ROUGE, for the quality-preservation check

    def per_query_ms(self) -> float:
        return self.pure_inference_ms_per_query + self.search_overhead_ms_per_query

    def total_cost_seconds(self, n_queries: int) -> float:
        return self.train_cost_seconds + n_queries * (self.per_query_ms() / 1000.0)

    def speedup_vs_baseline(self, n_queries: int) -> float:
        """Amortized speedup at N queries (the metric missing from prior work)."""
        baseline_total = n_queries * (self.baseline_ms_per_query / 1000.0)
        my_total = self.total_cost_seconds(n_queries)
        return baseline_total / my_total if my_total > 0 else float("inf")

    def single_query_speedup(self) -> float:
        """The metric prior papers actually report: N=1, no amortization."""
        return self.speedup_vs_baseline(n_queries=1)


def find_crossover(profile_a: MethodCostProfile, profile_b: MethodCostProfile,
                   n_max: int = 1_000_000, tol: int = 1) -> Optional[int]:
    """
    Binary search for the query volume N* at which the cheaper-per-query but
    higher-fixed-cost method (typically the trained router) overtakes the
    cheaper-fixed-cost but higher-per-query method (typically the heuristic).

    Returns None if one method dominates at all N in [1, n_max] (no crossover).
    """
    def a_is_cheaper(n):
        return profile_a.total_cost_seconds(n) < profile_b.total_cost_seconds(n)

    start_state = a_is_cheaper(1)
    end_state = a_is_cheaper(n_max)
    if start_state == end_state:
        return None  # no crossover in range: one method dominates throughout

    lo, hi = 1, n_max
    while hi - lo > tol:
        mid = (lo + hi) // 2
        if a_is_cheaper(mid) == start_state:
            lo = mid
        else:
            hi = mid
    return hi


def build_report(profiles: list[MethodCostProfile], query_volumes: list[int]) -> dict:
    """
    Produce the audit's headline table: for each method, amortized speedup at
    each query volume, plus pairwise crossover points against every other
    method. This is what prior papers are missing -- a volume-dependent
    ranking instead of a single point estimate at N=1.
    """
    report = {"query_volumes": query_volumes, "methods": {}, "crossovers": []}

    for p in profiles:
        report["methods"][p.name] = {
            "model_id": p.model_id,
            "train_cost_seconds": p.train_cost_seconds,
            "pure_inference_ms_per_query": p.pure_inference_ms_per_query,
            "search_overhead_ms_per_query": p.search_overhead_ms_per_query,
            "search_overhead_fraction": (
                p.search_overhead_ms_per_query / p.per_query_ms()
                if p.per_query_ms() > 0 else 0.0
            ),
            "single_query_speedup_AS_REPORTED_BY_PRIOR_PAPERS": round(p.single_query_speedup(), 3),
            "amortized_speedup_by_volume": {
                n: round(p.speedup_vs_baseline(n), 3) for n in query_volumes
            },
            "quality_score": p.quality_score,
        }

    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            n_star = find_crossover(profiles[i], profiles[j])
            report["crossovers"].append({
                "method_a": profiles[i].name,
                "method_b": profiles[j].name,
                "crossover_query_volume": n_star,
                "interpretation": (
                    f"Below N={n_star}, one method is cheaper; above it, ranking flips."
                    if n_star is not None else
                    f"{profiles[i].name if profiles[i].total_cost_seconds(1) < profiles[j].total_cost_seconds(1) else profiles[j].name}"
                    " dominates at all tested volumes (no crossover)."
                ),
            })
    return report


def save_report(report: dict, path: str):
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
