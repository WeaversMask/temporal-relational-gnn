"""Reference optimization, stochastic parity, and exact CPU checkpoint resume."""

import copy

import pandas as pd
import pytest
import torch
from relbench.base import Database, Table
from relbench.modeling.graph import AttachTargetTransform, make_pkey_fkey_graph
from torch_geometric.loader import NodeLoader
from torch_geometric.sampler import BaseSampler, HeteroSamplerOutput

from trgnn.real_batch import RealBatchModel
from trgnn.training import evaluate, train_pilot

CONTRACT = {"dataset_sha256": "fixture", "features": ["constant", "relative_time"]}


class TrainingSampler(BaseSampler):
    """Use repeated temporal users, stochastic events, and a zero-node type."""

    def sample_from_edges(self, inputs, neg_sampling=None):
        raise NotImplementedError

    def sample_from_nodes(self, inputs):
        count = inputs.node.numel()
        local = torch.arange(count)
        selected = torch.randint(2, (count,))
        empty = torch.empty(0, dtype=torch.long)
        relation = ("events", "f2p_UserId", "users")
        reverse = ("users", "rev_f2p_UserId", "events")
        absent = ("tags", "f2p_UserId", "users")
        absent_reverse = ("users", "rev_f2p_UserId", "tags")
        edges = {relation: local, reverse: local, absent: empty, absent_reverse: empty}
        return HeteroSamplerOutput(
            node={"users": torch.zeros(count, dtype=torch.long), "events": selected, "tags": empty},
            row=edges,
            col=dict(edges),
            edge={relation: selected, reverse: selected, absent: empty, absent_reverse: empty},
            batch={"users": local, "events": local, "tags": empty},
            num_sampled_nodes={"users": [count, 0, 0], "events": [0, count, 0], "tags": [0, 0, 0]},
            num_sampled_edges={
                relation: [count, 0],
                reverse: [0, count],
                absent: [0, 0],
                absent_reverse: [0, 0],
            },
            metadata=(inputs.input_id, inputs.time),
        )


@pytest.fixture
def training_setup():
    torch.set_num_threads(1)
    torch.manual_seed(17)
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
            "tags": Table(
                pd.DataFrame(
                    {
                        "Id": [0],
                        "UserId": pd.Series([pd.NA], dtype="Int64"),
                        "Created": pd.to_datetime(["2020-01-01"]),
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
        TrainingSampler(),
        input_nodes=("users", torch.tensor([0, 0, 0])),
        input_time=torch.tensor([1583020800, 1601510400, 1583020800]),
        transform=AttachTargetTransform("users", torch.tensor([0.0, 1.0, 0.0])),
        num_workers=0,
        filter_per_worker=False,
        batch_size=2,
    )
    model = RealBatchModel(graph, stats, channels=8)
    return loader, model


def run(model, loader, path, steps, **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    result = train_pilot(
        model,
        optimizer,
        loader,
        "users",
        steps=steps,
        batch_size=kwargs.pop("batch_size", 2),
        seed=31,
        checkpoint_path=path,
        contract=kwargs.pop("contract", CONTRACT),
        checkpoint_every=1,
        **kwargs,
    )
    return result, optimizer


def test_training_handles_zero_and_single_node_types(training_setup, tmp_path):
    loader, model = training_setup
    initial = copy.deepcopy(model.state_dict())
    callbacks = []
    report, _ = run(
        model, loader, tmp_path / "single.pt", 2, batch_size=1, on_step=callbacks.append
    )
    assert report["completed_steps"] == 2
    assert model.training
    assert any(not torch.equal(value, initial[key]) for key, value in model.state_dict().items())
    assert [row["step"] for row in callbacks] == [1, 2]
    for row in report["history"]:
        assert row["gradient_finite"] and row["gradient_tensor_count"] > 0
        assert row["step_ms"] > 0 and row["loss"] > 0
        assert row["cuda_memory"] is None
        assert row["sampled_counts"]["nodes"]["tags"] == [0, 0, 0]
        assert row["sampled_counts"]["nodes"]["events"] == [0, 1, 0]
    checkpoint = torch.load(tmp_path / "single.pt", weights_only=True)
    assert checkpoint["completed_steps"] == 2
    assert not list(tmp_path.glob(".checkpoint-*"))


def test_resume_matches_continuous_with_dropout_and_optimizer(training_setup, tmp_path):
    loader, initial = training_setup
    assert any(
        isinstance(module, torch.nn.Dropout) and module.p > 0 for module in initial.modules()
    )
    continuous = copy.deepcopy(initial)
    expected, expected_optimizer = run(continuous, loader, tmp_path / "continuous.pt", 4)
    interrupted = copy.deepcopy(initial)
    first, _ = run(interrupted, loader, tmp_path / "resume.pt", 2)
    resumed = copy.deepcopy(initial)
    actual, actual_optimizer = run(resumed, loader, tmp_path / "resume.pt", 4, resume=True)
    torch.testing.assert_close(resumed.state_dict(), continuous.state_dict(), atol=0, rtol=0)
    torch.testing.assert_close(
        actual_optimizer.state_dict(), expected_optimizer.state_dict(), atol=0, rtol=0
    )
    assert actual["history"][:2] == first["history"]
    for left, right in zip(actual["history"], expected["history"], strict=True):
        assert left["loss"] == right["loss"]
        assert left["sampler_sha256"] == right["sampler_sha256"]
        assert left["query_indices"] == right["query_indices"]


def test_resume_rejects_changed_contract_and_architecture(training_setup, tmp_path):
    loader, model = training_setup
    path = tmp_path / "resume.pt"
    run(model, loader, path, 1)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="contract mismatch"):
        run(model, loader, path, 2, resume=True, contract={"dataset_sha256": "changed"})
    with pytest.raises(ValueError, match="architecture mismatch"):
        changed = copy.deepcopy(model)
        changed.head = torch.nn.Linear(8, 2)
        run(changed, loader, path, 2, resume=True)
    with pytest.raises(FileExistsError):
        run(model, loader, path, 2)
    assert path.read_bytes() == original


def test_resume_rejects_changed_optimizer_hyperparameters(training_setup, tmp_path):
    loader, model = training_setup
    path = tmp_path / "resume.pt"
    run(model, loader, path, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    with pytest.raises(ValueError, match="training settings"):
        train_pilot(
            model,
            optimizer,
            loader,
            "users",
            steps=2,
            batch_size=2,
            seed=31,
            checkpoint_path=path,
            resume=True,
            contract=CONTRACT,
        )


def test_restore_callback_reports_prior_history_before_resumed_failure(
    training_setup, tmp_path, monkeypatch
):
    loader, model = training_setup
    initial = copy.deepcopy(model)
    path = tmp_path / "resume.pt"
    previous, _ = run(model, loader, path, 2)
    events = []

    def restored(payload):
        events.append(("restored", payload))
        torch.testing.assert_close(initial.state_dict(), model.state_dict(), atol=0, rtol=0)

    def failed_sampler(inputs):
        events.append(("sampling", None))
        raise RuntimeError("Simulated next-batch failure")

    monkeypatch.setattr(loader.node_sampler, "sample_from_nodes", failed_sampler)
    with pytest.raises(RuntimeError, match="next-batch failure"):
        run(initial, loader, path, 3, resume=True, on_restore=restored)
    assert [kind for kind, _ in events] == ["restored", "sampling"]
    payload = events[0][1]
    assert payload["completed_steps"] == 2
    assert payload["history"] == previous["history"]
    assert payload["resumed"] is True
    assert payload["checkpoint_path"] == str(path)
    events.clear()
    with pytest.raises(ValueError, match="contract mismatch"):
        run(
            initial,
            loader,
            path,
            3,
            resume=True,
            on_restore=restored,
            contract={"dataset_sha256": "changed"},
        )
    assert events == []


def test_failed_checkpoint_write_preserves_last_complete_checkpoint(
    training_setup, tmp_path, monkeypatch
):
    loader, model = training_setup
    path = tmp_path / "atomic.pt"
    run(model, loader, path, 1)
    original = path.read_bytes()

    def interrupted_save(payload, stream):
        stream.write(b"partial checkpoint")
        raise OSError("Simulated interrupted write")

    monkeypatch.setattr("trgnn.training.torch.save", interrupted_save)
    with pytest.raises(OSError, match="interrupted write"):
        run(model, loader, path, 2, resume=True)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".checkpoint-*"))


def test_evaluation_preserves_mode_rng_and_query_label_order(training_setup):
    loader, model = training_setup
    model.train()
    state = torch.get_rng_state().clone()
    result = evaluate(model, loader, "users", batch_size=2, seed=71)
    assert model.training
    assert torch.equal(torch.get_rng_state(), state)
    assert result["labels"] == [0.0, 1.0, 0.0]
    assert len(result["predictions"]) == 3
    assert result["metrics"]["count"] == 3
    assert all(0 < value < 1 for value in result["predictions"])
    assert len(result["sampler_sha256"]) == 2


def test_training_rejects_invalid_labels_before_optimizer_step(training_setup, tmp_path):
    loader, model = training_setup
    before = copy.deepcopy(model.state_dict())
    loader.transform.target[:] = 0.5
    with pytest.raises(ValueError, match="binary query labels"):
        run(model, loader, tmp_path / "invalid.pt", 1)
    torch.testing.assert_close(model.state_dict(), before, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["hgam", "blocks"])
def test_full_model_training_gradients_and_adam_match_with_aligned_dropout(training_setup, mode):
    loader, model = training_setup
    sample = loader.node_sampler.sample_from_nodes(loader.input_data[[0, 1]])
    batch = loader.filter_fn(sample)
    for frame in batch.tf_dict.values():
        for key, value in frame.feat_dict.items():
            frame.feat_dict[key] = value.double()
    initial = model.double()
    reference, candidate = copy.deepcopy(initial), copy.deepcopy(initial)
    states, gradients, losses = [], [], []
    for actual_model, schedule in [(reference, "reference"), (candidate, mode)]:
        actual_model.train()
        optimizer = torch.optim.Adam(actual_model.parameters(), lr=0.001)
        torch.manual_seed(53)
        logits = actual_model(batch, "users", schedule)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, batch["users"].y.double()
        )
        loss.backward()
        losses.append(loss.detach())
        gradients.append(
            {
                name: parameter.grad.clone() if parameter.grad is not None else None
                for name, parameter in actual_model.named_parameters()
            }
        )
        optimizer.step()
        states.append(copy.deepcopy(actual_model.state_dict()))
    torch.testing.assert_close(losses[0], losses[1], atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(gradients[0], gradients[1], atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(states[0], states[1], atol=1e-9, rtol=1e-8)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires NVIDIA CUDA training validation"
)
def test_cuda_training_and_validation_smoke(training_setup, tmp_path):
    loader, model = training_setup
    model.cuda()
    report, _ = run(model, loader, tmp_path / "cuda.pt", 2)
    assert report["history"][-1]["cuda_memory"]["peak_allocated_bytes"] > 0
    result = evaluate(model, loader, "users", batch_size=2, seed=71, device="cuda")
    assert result["metrics"]["count"] == 3
    assert all(row["gradient_finite"] for row in report["history"])
