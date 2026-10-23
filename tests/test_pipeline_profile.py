"""Exercise the real encoders and serial pipeline without a compiled sampler."""

import copy
import json

import pandas as pd
import pytest
import torch
from relbench.base import Database, Table
from relbench.modeling.graph import AttachTargetTransform, make_pkey_fkey_graph
from torch_geometric.loader import NodeLoader
from torch_geometric.sampler import BaseSampler, HeteroSamplerOutput
from torch_geometric.typing import WITH_PYG_LIB

from trgnn.execution import MODES
from trgnn.pipeline_profile import STAGES, profile_batch, sampler_fingerprint
from trgnn.real_batch import RealBatchModel


class OccurrenceSampler(BaseSampler):
    """Two temporal queries share a global user and sampled event IDs."""

    def sample_from_edges(self, inputs, neg_sampling=None):
        raise NotImplementedError("This fixture only samples node queries")

    def sample_from_nodes(self, inputs):
        selected = torch.randint(2, (2,))
        relation = ("events", "f2p_UserId", "users")
        reverse = ("users", "rev_f2p_UserId", "events")
        local = torch.arange(2)
        return HeteroSamplerOutput(
            node={"users": torch.tensor([0, 0]), "events": selected},
            row={relation: local, reverse: local},
            col={relation: local, reverse: local},
            edge={relation: selected, reverse: selected},
            batch={"users": local, "events": local},
            num_sampled_nodes={"users": [2, 0, 0], "events": [0, 2, 0]},
            num_sampled_edges={relation: [2, 0], reverse: [0, 2]},
            metadata=(inputs.input_id, inputs.time),
        )


@pytest.fixture
def pipeline():
    torch.set_num_threads(1)
    database = Database(
        {
            "users": Table(
                pd.DataFrame({"Id": [0], "Created": pd.to_datetime(["2020-01-01"])}),
                {},
                pkey_col="Id",
                time_col="Created",
            ),
            "events": Table(
                pd.DataFrame(
                    {
                        "Id": [0, 1],
                        "UserId": [0, 0],
                        "Created": pd.to_datetime(["2020-02-01", "2020-02-02"]),
                    }
                ),
                {"UserId": "users"},
                pkey_col="Id",
                time_col="Created",
            ),
        }
    )
    graph, stats = make_pkey_fkey_graph(database, {name: {} for name in database.table_dict})
    loader = NodeLoader(
        graph,
        OccurrenceSampler(),
        input_nodes=("users", torch.tensor([0, 0])),
        input_time=torch.tensor([1583020800, 1601510400]),
        transform=AttachTargetTransform("users", torch.tensor([0.0, 1.0])),
        num_workers=0,
        batch_size=2,
        filter_per_worker=False,
    )
    model = RealBatchModel(graph, stats, channels=8).eval()
    return loader, model


def test_pipeline_pairs_occurrences_and_matches_staged_forward(pipeline):
    loader, model = pipeline
    before = copy.deepcopy(model.state_dict())
    record = profile_batch(loader, [0, 1], model, "users", repeats=3, warmup=1)
    assert len(record["trials"]) == 3
    for trial in record["trials"]:
        assert set(trial["execution_order"]) == set(MODES)
        assert trial["visibility_audit"]["repeated_global_id_occurrences"]["users"] == 1
        assert trial["seed_times_unix_seconds"] == [1583020800, 1601510400]
        for mode in MODES:
            assert trial["pipeline_ms"][mode] > 0
            assert set(trial["stage_ms"][mode]) == set(STAGES)
            assert all(value >= 0 for value in trial["stage_ms"][mode].values())
            assert trial["cuda_memory"][mode] is None
    assert max(record["max_absolute_logit_error"].values()) < 1e-5
    torch.testing.assert_close(model.state_dict(), before, atol=0, rtol=0)
    assert loader.data["users"].tf.device.type == "cpu"
    json.dumps(record, allow_nan=False)


def test_fingerprint_includes_occurrence_ownership_and_query_times(pipeline):
    loader, _ = pipeline
    sample = loader.node_sampler.sample_from_nodes(loader.input_data[[0, 1]])
    baseline = sampler_fingerprint(sample)
    changed = copy.deepcopy(sample)
    changed.batch["events"] = torch.tensor([1, 0])
    assert sampler_fingerprint(changed) != baseline
    changed = copy.deepcopy(sample)
    changed.metadata = (sample.metadata[0], sample.metadata[1] + 1)
    assert sampler_fingerprint(changed) != baseline


def test_profile_rejects_unpaired_sampler_results(pipeline):
    loader, model = pipeline
    original = loader.node_sampler.sample_from_nodes
    calls = 0

    def unstable(inputs):
        nonlocal calls
        result = original(inputs)
        result.metadata = (result.metadata[0], result.metadata[1] + calls)
        calls += 1
        return result

    loader.node_sampler.sample_from_nodes = unstable
    with pytest.raises(AssertionError, match="different sampled occurrences"):
        profile_batch(loader, [0, 1], model, "users", repeats=3, warmup=0)


@pytest.mark.skipif(not WITH_PYG_LIB, reason="Requires the compiled temporal sampler")
def test_profile_pairs_actual_temporal_sampler(pipeline):
    from torch_geometric.loader import NeighborLoader

    fixture_loader, model = pipeline
    loader = NeighborLoader(
        fixture_loader.data,
        input_nodes=("users", torch.tensor([0, 0])),
        input_time=torch.tensor([1583020800, 1601510400]),
        transform=fixture_loader.transform,
        time_attr="time",
        num_neighbors=[1, 1],
        disjoint=True,
        subgraph_type="directional",
        batch_size=2,
        num_workers=0,
        filter_per_worker=False,
    )
    record = profile_batch(loader, [0, 1], model, "users", repeats=3, warmup=0)
    assert len(record["trials"]) == 3
    assert all(trial["visibility_audit"]["status"] == "passed" for trial in record["trials"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA runtime")
def test_pipeline_transfer_and_cuda_memory(pipeline):
    loader, model = pipeline
    model = model.cuda()
    record = profile_batch(loader, [0, 1], model, "users", repeats=3, warmup=1, device="cuda")
    for trial in record["trials"]:
        for mode in MODES:
            memory = trial["cuda_memory"][mode]
            assert memory["peak_allocated_bytes"] >= memory["baseline_allocated_bytes"] > 0
            assert memory["extra_peak_allocated_bytes"] > 0
    assert loader.data["users"].tf.device.type == "cpu"
