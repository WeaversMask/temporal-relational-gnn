"""Summaries of paired online inference trials at one query batch size.

Repeats of one query batch share a cluster. Confidence intervals resample whole
query batches, retaining all repeats and execution modes within each cluster.
They describe variation across the measured query batches, not independent GPU
sessions, devices, training seeds, or data snapshots.
"""

import math
import random
import statistics
from numbers import Real

MODES = ("reference", "hgam", "blocks")
STAGES = ("sampling", "feature_gather", "transfer", "encoding", "message_passing", "readout")
MEMORY_FIELDS = (
    "baseline_allocated_bytes",
    "baseline_reserved_bytes",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "extra_peak_allocated_bytes",
)


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _quantile(values, probability):
    """Linearly interpolate between adjacent ordered observations."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_profiles(records, batch_size, *, bootstrap_samples=2000, seed=42):
    """Summarize raw ``profile_batch`` records for a single scheduled cohort.

    Each record must also contain ``batch_index`` and actual ``query_count``.
    The caller establishes that these query batches cover its cohort without
    overlap. A final incomplete batch is allowed. Equal repeat counts prevent
    unequal weighting from partially collected records.

    The paired reduction estimand is ``100 * (1 - sum(blocks_ms) / sum(baseline_ms))``
    across all online trials. Bootstrap intervals retain the covariance of mode
    timings by resampling query batches together. Stage timings come from separate
    synchronized replays and are never added to the online pipeline timings.
    """
    _integer(batch_size, "batch_size", minimum=1)
    _integer(bootstrap_samples, "bootstrap_samples", minimum=1)
    _integer(seed, "seed")
    records = list(records)
    if not records:
        raise ValueError("At least one query batch is required")

    online = {mode: [] for mode in MODES}
    stages = {mode: {stage: [] for stage in STAGES} for mode in MODES}
    memory_max = {mode: {field: 0 for field in MEMORY_FIELDS} for mode in MODES}
    cluster_totals = []
    batch_indices = set()
    remainder_indices = []
    repeats = None
    query_count = 0
    memory_present = None
    try:
        for record in records:
            index = _integer(record["batch_index"], "batch_index")
            if index in batch_indices:
                raise ValueError("batch_index values must be unique")
            batch_indices.add(index)
            count = _integer(record["query_count"], "query_count", minimum=1)
            if count > batch_size or record.get("batch_size", batch_size) != batch_size:
                raise ValueError("Records must belong to the same batch_size")
            if count < batch_size:
                remainder_indices.append(index)
            query_count += count
            trials = record["trials"]
            if not isinstance(trials, list) or not trials:
                raise ValueError("Every query batch must have a nonempty trials list")
            if repeats is None:
                repeats = len(trials)
            if len(trials) != repeats:
                raise ValueError("All query batches must have equal repeat counts")
            totals = {mode: 0.0 for mode in MODES}
            for trial in trials:
                for mode in MODES:
                    elapsed = _positive(trial["pipeline_ms"][mode], f"pipeline_ms.{mode}")
                    online[mode].append(elapsed)
                    totals[mode] += elapsed
                    for stage in STAGES:
                        stages[mode][stage].append(
                            _positive(trial["stage_ms"][mode][stage], f"stage_ms.{mode}.{stage}")
                        )
                    memory = trial["cuda_memory"][mode]
                    present = memory is not None
                    if memory_present is None:
                        memory_present = present
                    if present != memory_present:
                        raise ValueError("Do not mix CPU and CUDA memory records")
                    if present:
                        for field in MEMORY_FIELDS:
                            value = _integer(memory[field], f"cuda_memory.{mode}.{field}")
                            memory_max[mode][field] = max(memory_max[mode][field], value)
                        if (
                            memory["peak_allocated_bytes"] < memory["baseline_allocated_bytes"]
                            or memory["peak_reserved_bytes"] < memory["baseline_reserved_bytes"]
                            or memory["extra_peak_allocated_bytes"]
                            != memory["peak_allocated_bytes"] - memory["baseline_allocated_bytes"]
                        ):
                            raise ValueError("Inconsistent CUDA allocator memory record")
            cluster_totals.append(totals)
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing or malformed profile field: {error}") from error
    if remainder_indices and (
        len(remainder_indices) != 1 or remainder_indices[0] != max(batch_indices)
    ):
        raise ValueError("Only the final query batch may be smaller than batch_size")

    # Sorting ensures seeded intervals do not depend on sidecar discovery order.
    cluster_totals = [
        total
        for _, total in sorted(
            zip((record["batch_index"] for record in records), cluster_totals, strict=True)
        )
    ]
    totals = {mode: math.fsum(online[mode]) for mode in MODES}
    if any(not math.isfinite(value) for value in totals.values()):
        raise ValueError("Aggregate pipeline times must remain finite")
    reduction = {
        baseline: 100 * (1 - totals["blocks"] / totals[baseline])
        for baseline in ("reference", "hgam")
    }
    intervals = {baseline: None for baseline in reduction}
    if len(records) > 1:
        rng = random.Random(seed)
        samples = {baseline: [] for baseline in reduction}
        for _ in range(bootstrap_samples):
            clusters = rng.choices(cluster_totals, k=len(cluster_totals))
            sampled_totals = {
                mode: math.fsum(cluster[mode] for cluster in clusters) for mode in MODES
            }
            for baseline in reduction:
                samples[baseline].append(
                    100 * (1 - sampled_totals["blocks"] / sampled_totals[baseline])
                )
        for baseline in reduction:
            intervals[baseline] = [
                _quantile(samples[baseline], 0.025),
                _quantile(samples[baseline], 0.975),
            ]

    return {
        "batch_size": batch_size,
        "query_batches": len(records),
        "query_count": query_count,
        "repeats_per_batch": repeats,
        "measured_query_count": query_count * repeats,
        "latency_scope": "pooled uninstrumented whole online pipeline trials, per query batch",
        "percentile_method": "linear interpolation between ordered observations",
        "throughput_method": "all measured query occurrences / sum of online pipeline seconds",
        "stage_timing_method": "separate synchronized replay; never summed into online latency",
        "cuda_memory_scope": (
            "PyTorch allocator maxima; excludes driver and other GPU users; "
            "reserved memory includes the warm cache from previous trials and replays"
        ),
        "modes": {
            mode: {
                "batch_latency_ms": {
                    "median": statistics.median(online[mode]),
                    "p95": _quantile(online[mode], 0.95),
                },
                "total_pipeline_ms": totals[mode],
                "throughput_queries_per_second": query_count * repeats * 1000 / totals[mode],
                "stage_replay_median_ms": {
                    stage: statistics.median(stages[mode][stage]) for stage in STAGES
                },
                "cuda_memory_max_bytes": memory_max[mode] if memory_present else None,
            }
            for mode in MODES
        },
        "comparisons": {
            baseline: {
                "blocks_latency_reduction_percent": reduction[baseline],
                "cluster_bootstrap_95_percent_ci": intervals[baseline],
                "estimand": "100 * (1 - sum(blocks online ms) / sum(baseline online ms))",
                "limitation": (
                    "Only one query batch; a between-batch confidence interval is unavailable"
                    if len(records) == 1
                    else "Interval covers query-batch variation within this measured run only"
                ),
            }
            for baseline in reduction
        },
        "bootstrap": {
            "method": "paired query-batch cluster bootstrap, percentile 95% interval",
            "resampling_unit": "whole query batch with every repeat and mode preserved",
            "samples": bootstrap_samples,
            "seed": seed,
        },
    }
