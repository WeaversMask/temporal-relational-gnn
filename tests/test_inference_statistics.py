import copy
import math

import pytest

from trgnn.inference_statistics import MODES, STAGES, summarize_profiles


def profile(index, query_count, reference, hgam, blocks, *, cuda=False):
    record = {"batch_index": index, "query_count": query_count, "trials": []}
    for repeat, timings in enumerate(zip(reference, hgam, blocks, strict=True)):
        trial = {
            "sampler_seed": 42 + repeat,
            "execution_order": list(MODES),
            "pipeline_ms": dict(zip(MODES, timings, strict=True)),
            "stage_ms": {mode: {stage: 100.0 for stage in STAGES} for mode in MODES},
            "cuda_memory": {mode: None for mode in MODES},
        }
        if cuda:
            trial["cuda_memory"] = {
                mode: {
                    "baseline_allocated_bytes": 10,
                    "baseline_reserved_bytes": 20,
                    "peak_allocated_bytes": 30 + index + repeat,
                    "peak_reserved_bytes": 50 + index + repeat,
                    "extra_peak_allocated_bytes": 20 + index + repeat,
                }
                for mode in MODES
            }
        record["trials"].append(trial)
    return record


def test_paired_ratio_and_whole_pipeline_statistics():
    records = [
        profile(0, 32, [10, 20], [8, 16], [4, 8]),
        profile(1, 32, [30, 40], [24, 32], [12, 16]),
    ]
    result = summarize_profiles(records, 32, bootstrap_samples=100)
    reference = result["modes"]["reference"]
    assert reference["batch_latency_ms"] == {"median": 25, "p95": pytest.approx(38.5)}
    assert reference["total_pipeline_ms"] == 100
    assert result["query_count"] == 64
    assert result["measured_query_count"] == 128
    assert reference["throughput_queries_per_second"] == 1280
    assert result["comparisons"]["reference"]["blocks_latency_reduction_percent"] == 60
    assert result["comparisons"]["hgam"]["blocks_latency_reduction_percent"] == 50
    assert result["comparisons"]["hgam"]["cluster_bootstrap_95_percent_ci"] == [50, 50]
    assert reference["stage_replay_median_ms"] == dict.fromkeys(STAGES, 100)
    assert reference["total_pipeline_ms"] != sum(reference["stage_replay_median_ms"].values())
    assert reference["cuda_memory_max_bytes"] is None


def test_throughput_uses_total_queries_and_time_with_incomplete_final_batch():
    records = [
        profile(0, 32, [1, 3], [1, 3], [1, 3]),
        profile(1, 2, [10, 30], [10, 30], [10, 30]),
    ]
    result = summarize_profiles(records, 32)
    assert result["query_count"] == 34
    assert result["measured_query_count"] == 68
    assert result["modes"]["blocks"]["throughput_queries_per_second"] == pytest.approx(68000 / 44)
    assert result["modes"]["blocks"]["throughput_queries_per_second"] != 32000 / 6.5


def test_bootstrap_resamples_clusters_preserving_all_repeat_pairs():
    # Resampling trials independently would create intermediate ratios. Whole
    # batches have only the 0%, 50%, or 100% reduction possibilities here.
    records = [
        profile(0, 4, [10, 90], [10, 90], [1, 99]),
        profile(1, 4, [10, 90], [10, 90], [0.01, 0.01]),
    ]
    result = summarize_profiles(records, 4, bootstrap_samples=400, seed=9)
    interval = result["comparisons"]["hgam"]["cluster_bootstrap_95_percent_ci"]
    assert interval == pytest.approx([0, 99.98])
    assert result == summarize_profiles(records, 4, bootstrap_samples=400, seed=9)
    assert result == summarize_profiles(records[::-1], 4, bootstrap_samples=400, seed=9)


def test_single_cluster_cannot_estimate_between_batch_interval():
    result = summarize_profiles([profile(0, 32, [2, 2], [2, 2], [1, 1])], 32)
    comparison = result["comparisons"]["hgam"]
    assert comparison["blocks_latency_reduction_percent"] == 50
    assert comparison["cluster_bootstrap_95_percent_ci"] is None
    assert "Only one query batch" in comparison["limitation"]


def test_memory_maxima_are_allocator_measurements():
    records = [
        profile(0, 4, [3, 4], [2, 3], [1, 2], cuda=True),
        profile(1, 4, [5, 6], [4, 5], [2, 3], cuda=True),
    ]
    result = summarize_profiles(records, 4)
    for mode in MODES:
        assert result["modes"][mode]["cuda_memory_max_bytes"] == {
            "baseline_allocated_bytes": 10,
            "baseline_reserved_bytes": 20,
            "peak_allocated_bytes": 32,
            "peak_reserved_bytes": 52,
            "extra_peak_allocated_bytes": 22,
        }
    assert "excludes driver" in result["cuda_memory_scope"]


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, -math.inf, True, "1"])
@pytest.mark.parametrize("field", ["pipeline_ms", "stage_ms"])
def test_invalid_timings_rejected(value, field):
    record = profile(0, 2, [1], [1], [1])
    destination = record["trials"][0][field]
    if field == "stage_ms":
        destination["blocks"]["encoding"] = value
    else:
        destination["blocks"] = value
    with pytest.raises(ValueError, match="finite positive"):
        summarize_profiles([record], 2)


@pytest.mark.parametrize("field", ["pipeline_ms", "stage_ms", "cuda_memory"])
def test_missing_mode_rejected(field):
    record = profile(0, 2, [1], [1], [1])
    del record["trials"][0][field]["hgam"]
    with pytest.raises(ValueError, match="Missing or malformed"):
        summarize_profiles([record], 2)


def test_incomplete_records_and_mixed_sizes_rejected():
    full = profile(0, 8, [1, 2], [1, 2], [1, 2])
    short_repeats = profile(1, 8, [1], [1], [1])
    with pytest.raises(ValueError, match="equal repeat counts"):
        summarize_profiles([full, short_repeats], 8)
    with pytest.raises(ValueError, match="unique"):
        summarize_profiles([full, full], 8)
    oversized = copy.deepcopy(full)
    oversized["query_count"] = 9
    with pytest.raises(ValueError, match="same batch_size"):
        summarize_profiles([oversized], 8)
    declared_other_size = copy.deepcopy(full)
    declared_other_size["batch_size"] = 16
    with pytest.raises(ValueError, match="same batch_size"):
        summarize_profiles([declared_other_size], 8)
    early_partial = profile(0, 4, [1, 2], [1, 2], [1, 2])
    later_full = profile(1, 8, [1, 2], [1, 2], [1, 2])
    with pytest.raises(ValueError, match="Only the final"):
        summarize_profiles([early_partial, later_full], 8)
    with pytest.raises(ValueError, match="Only the final"):
        summarize_profiles([early_partial, profile(1, 4, [1, 2], [1, 2], [1, 2])], 8)


def test_invalid_memory_and_cpu_cuda_mixture_rejected():
    record = profile(0, 2, [1], [1], [1], cuda=True)
    record["trials"][0]["cuda_memory"]["blocks"]["extra_peak_allocated_bytes"] = 21
    with pytest.raises(ValueError, match="Inconsistent CUDA"):
        summarize_profiles([record], 2)
    record["trials"][0]["cuda_memory"]["blocks"] = None
    with pytest.raises(ValueError, match="Do not mix CPU and CUDA"):
        summarize_profiles([record], 2)


@pytest.mark.parametrize(
    "kwargs", [{"batch_size": 0}, {"batch_size": True}, {"bootstrap_samples": 0}, {"seed": -1}]
)
def test_invalid_options_rejected(kwargs):
    options = {"batch_size": 2} | kwargs
    with pytest.raises(ValueError):
        summarize_profiles([profile(0, 2, [1], [1], [1])], **options)


def test_empty_and_missing_batch_metadata_rejected():
    with pytest.raises(ValueError, match="At least one"):
        summarize_profiles([], 2)
    record = profile(0, 2, [1], [1], [1])
    del record["query_count"]
    with pytest.raises(ValueError, match="Missing or malformed"):
        summarize_profiles([record], 2)
