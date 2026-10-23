import copy

import pytest
import torch

from trgnn.execution import MODES, RelationalGraphSAGE, seed_outputs, validate_sampled_inputs
from trgnn.synthetic import EDGE_TYPES, EMPTY_RELATION, NODE_TYPES, make_batch


@pytest.mark.parametrize("fanouts", [(2,), (2, 1), (2, 1, 1), (0, 0)])
@pytest.mark.parametrize("mode", ["hgam", "blocks"])
def test_seed_outputs_match_reference(fanouts, mode):
    torch.manual_seed(19)
    batch = make_batch(fanouts=fanouts).to("cpu", torch.float64)
    model = RelationalGraphSAGE(NODE_TYPES, EDGE_TYPES, 8, len(fanouts)).double()
    reference = seed_outputs(model(*batch.inputs(), mode="reference"), batch.node_counts)
    actual = seed_outputs(model(*batch.inputs(), mode=mode), batch.node_counts)
    torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("mode", ["hgam", "blocks"])
def test_gradients_and_optimizer_step_match(mode):
    torch.manual_seed(7)
    batch = make_batch().to("cpu", torch.float64)
    reference = RelationalGraphSAGE(NODE_TYPES, EDGE_TYPES, 8, 2).double()
    candidate = copy.deepcopy(reference)
    grads, states, losses, feature_grads = [], [], [], []
    for model, schedule in [(reference, "reference"), (candidate, mode)]:
        features = {k: x.clone().requires_grad_() for k, x in batch.x_dict.items()}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        outputs = model(features, *batch.inputs()[1:], mode=schedule)
        seeds = seed_outputs(outputs, batch.node_counts)["user"]
        target = torch.arange(seeds.numel(), dtype=torch.float64).reshape_as(seeds) / 100
        loss = (seeds - target).square().mean()
        loss.backward()
        losses.append(loss.detach())
        grads.append(
            {
                k: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                for k, p in model.named_parameters()
            }
        )
        feature_grads.append(
            {
                k: x.grad.clone() if x.grad is not None else torch.zeros_like(x)
                for k, x in features.items()
            }
        )
        optimizer.step()
        states.append(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(losses[0], losses[1], atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(grads[0], grads[1], atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(feature_grads[0], feature_grads[1], atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(states[0], states[1], atol=1e-9, rtol=1e-8)


def test_empty_relation_contribution_is_preserved():
    torch.manual_seed(1)
    batch = make_batch().to("cpu", torch.float64)
    assert batch.edge_index_dict[EMPTY_RELATION].size(1) == 0
    model = RelationalGraphSAGE(NODE_TYPES, EDGE_TYPES, 8).double()
    original = model(*batch.inputs(), mode="reference")["user"][:4]
    without_empty = {k: v for k, v in batch.edge_index_dict.items() if k != EMPTY_RELATION}
    dropped = model(batch.x_dict, without_empty, batch.node_counts, batch.edge_counts)["user"][:4]
    assert not torch.allclose(original, dropped)
    blocks = model(*batch.inputs(), mode="blocks")["user"]
    torch.testing.assert_close(blocks, original, atol=1e-10, rtol=1e-9)


def test_occurrences_are_disjoint_and_time_safe():
    batch = make_batch()
    ids = batch.global_ids["user"]
    assert ids.unique().numel() < ids.numel()
    assert not torch.equal(batch.x_dict["user"][0], batch.x_dict["user"][3])
    for name in NODE_TYPES:
        assert bool((batch.node_times[name] <= batch.seed_times[batch.query_owner[name]]).all())
    for (src, _, dst), edges in batch.edge_index_dict.items():
        torch.testing.assert_close(
            batch.query_owner[src][edges[0]], batch.query_owner[dst][edges[1]]
        )


@pytest.mark.parametrize("mode", MODES)
def test_single_isolated_seed(mode):
    batch = make_batch(batch_size=1)
    model = RelationalGraphSAGE(NODE_TYPES, EDGE_TYPES, 8)
    seeds = seed_outputs(model(*batch.inputs(), mode=mode), batch.node_counts)
    assert seeds["user"].shape == (1, 8)
    assert bool(torch.isfinite(seeds["user"]).all())


def test_validation_rejects_invalid_destination_prefix():
    batch = make_batch()
    relation = EDGE_TYPES[0]
    batch.edge_index_dict[relation][1, 0] = batch.x_dict[relation[2]].size(0)
    with pytest.raises(ValueError, match="BFS prefix"):
        validate_sampled_inputs(*batch.inputs(), 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA runtime")
def test_cuda_fp32_seed_equivalence():
    batch = make_batch().to("cuda")
    model = RelationalGraphSAGE(NODE_TYPES, EDGE_TYPES, 8).cuda().eval()
    with torch.inference_mode():
        reference = seed_outputs(model(*batch.inputs(), mode="reference"), batch.node_counts)
        for mode in ["hgam", "blocks"]:
            outputs = seed_outputs(model(*batch.inputs(), mode=mode), batch.node_counts)
            torch.testing.assert_close(outputs, reference, atol=1e-5, rtol=1e-4)
