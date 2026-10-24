"""Bounded reference training with deterministic query/sampler/dropout schedules.

This module consumes an already prepared graph and official query loader. It
does not download data, choose historical features, or certify their provenance.
"""

import copy
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from trgnn.activity_baseline import binary_metrics
from trgnn.pipeline_profile import host_memory, sampler_fingerprint, validate_batch

CHECKPOINT_VERSION = 1
SEED_SCHEME = "CPU/CUDA torch seed = base seed + zero-based optimizer step"
QUERY_SCHEME = "one fixed CPU-generator permutation, traversed circularly in query batches"
TIMING_BOUNDARY = (
    "sampling + gathering + transfer + reference forward + BCE + backward + finite-gradient "
    "checks + optimizer step; excludes query scheduling, visibility/fingerprint audits, "
    "checkpoint writes, and reporting; step_ms is the sum of the two timed work regions"
)


def _device(model):
    return next(model.parameters()).device


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _seed(seed, device):
    # Avoid changing CUDA state in a CPU-only development run.
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _rng_state():
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def _restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state["cuda"])


def _input_signature(loader):
    digest = hashlib.sha256()
    for name in ("input_type", "node", "time", "input_id"):
        value = getattr(loader.input_data, name)
        digest.update(name.encode())
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
        else:
            digest.update(repr(value).encode())
        digest.update(b"\0")
    target = getattr(loader.transform, "target", None)
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().contiguous()
        digest.update(str((target.dtype, tuple(target.shape))).encode())
        digest.update(target.numpy().tobytes())
    return digest.hexdigest()


def _architecture(model):
    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "module_sha256": hashlib.sha256(repr(model).encode()).hexdigest(),
        "state": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in model.state_dict().items()
        },
    }


def _optimizer_groups(optimizer):
    groups = []
    for group in optimizer.param_groups:
        values = {}
        for name, value in group.items():
            if name == "params":
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError("Only scalar tensor optimizer hyperparameters are supported")
                value = value.detach().cpu().item()
            values[name] = value
        groups.append({"parameter_count": len(group["params"]), "hyperparameters": values})
    return json.loads(json.dumps(groups, sort_keys=True, allow_nan=False))


def _validate_loader(loader, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if loader.num_workers != 0:
        raise ValueError("Reference training requires a serial loader with num_workers=0")
    count = loader.input_data.node.numel()
    if count < 1:
        raise ValueError("The query loader is empty")
    return count


def _memory_start(device):
    if device.type != "cuda":
        return None
    baseline = {
        "baseline_allocated_bytes": torch.cuda.memory_allocated(device),
        "baseline_reserved_bytes": torch.cuda.memory_reserved(device),
    }
    torch.cuda.reset_peak_memory_stats(device)
    return baseline


def _memory_finish(device, baseline):
    if baseline is None:
        return None
    peak = torch.cuda.max_memory_allocated(device)
    return {
        **baseline,
        "peak_allocated_bytes": peak,
        "extra_peak_allocated_bytes": peak - baseline["baseline_allocated_bytes"],
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "scope": "PyTorch allocator; reserved memory includes the existing cache",
    }


def _counts(batch):
    return {
        "nodes": copy.deepcopy(batch.num_sampled_nodes_dict),
        "edges": {
            "|".join(relation): list(counts)
            for relation, counts in batch.num_sampled_edges_dict.items()
        },
    }


def _labels(batch, entity):
    labels = batch[entity].y
    count = batch[entity].seed_time.numel()
    if labels.ndim != 1 or labels.numel() != count:
        raise ValueError("Expected one binary label per seed occurrence")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("Expected finite binary query labels")
    return labels


def _atomic_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".checkpoint-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def train_pilot(
    model,
    optimizer,
    loader,
    entity,
    *,
    steps,
    batch_size,
    seed,
    checkpoint_path,
    resume=False,
    contract,
    checkpoint_every=10,
    on_step=None,
    on_restore=None,
):
    """Train only reference execution up to a *total* optimizer-step target.

    All stochastic work in a step is seeded from its absolute step number. The
    fixed query permutation, model/optimizer state, input signature, and caller's
    data/feature contract are checked when resuming. The step target may increase.
    ``on_restore`` receives the restored progress before any new batch is sampled.
    """
    count = _validate_loader(loader, batch_size)
    if steps < 1 or checkpoint_every < 1 or seed < 0:
        raise ValueError("Use positive steps/checkpoint interval and a non-negative seed")
    # Reject non-portable/non-finite metadata before doing any training work.
    contract = json.loads(json.dumps(contract, sort_keys=True, allow_nan=False))
    device = _device(model)
    path = Path(checkpoint_path)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    query_order = torch.randperm(count, generator=generator)
    settings = {
        "seed": seed,
        "batch_size": batch_size,
        "query_count": count,
        "input_sha256": _input_signature(loader),
        "entity": entity,
        "architecture": _architecture(model),
        "optimizer": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "optimizer_groups": _optimizer_groups(optimizer),
        "seed_scheme": SEED_SCHEME,
        "query_scheme": QUERY_SCHEME,
    }
    history = []
    completed = 0
    if resume:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != CHECKPOINT_VERSION:
            raise ValueError("Unsupported training checkpoint format")
        if checkpoint.get("contract") != contract:
            raise ValueError("Checkpoint data/feature contract mismatch")
        if checkpoint.get("settings") != settings:
            raise ValueError("Checkpoint training settings or model architecture mismatch")
        if not torch.equal(checkpoint["query_order"], query_order):
            raise ValueError("Checkpoint query order mismatch")
        completed = checkpoint["completed_steps"]
        history = checkpoint["history"]
        if completed != len(history) or [row["step"] for row in history] != list(
            range(1, completed + 1)
        ):
            raise ValueError("Checkpoint history does not match completed steps")
        if steps < completed:
            raise ValueError(
                "The total step target is smaller than the checkpoint's completed count"
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        _restore_rng(checkpoint["rng_state"])
        if on_restore is not None:
            on_restore(
                {
                    "completed_steps": completed,
                    "history": copy.deepcopy(history),
                    "resumed": True,
                    "checkpoint_path": str(path),
                }
            )
    elif path.exists():
        raise FileExistsError("Checkpoint already exists; use resume=True or a new checkpoint path")

    def save():
        _atomic_checkpoint(
            path,
            {
                "format_version": CHECKPOINT_VERSION,
                "contract": contract,
                "settings": settings,
                "completed_steps": completed,
                "query_order": query_order,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "rng_state": _rng_state(),
                "device": str(device),
            },
        )

    if not resume:
        save()
    model.train()
    layers = len(model.gnn.convs)
    for step in range(completed, steps):
        indices = query_order[(torch.arange(batch_size) + step * batch_size) % count].tolist()
        _seed(seed + step, device)
        optimizer.zero_grad(set_to_none=True)
        _synchronize(device)
        memory = _memory_start(device)
        started = time.perf_counter_ns()
        sample = loader.node_sampler.sample_from_nodes(loader.input_data[indices])
        batch = loader.filter_fn(sample)
        sample_ms = (time.perf_counter_ns() - started) / 1e6

        audit_started = time.perf_counter_ns()
        visibility = validate_batch(batch, entity, layers)
        _labels(batch, entity)
        fingerprint = sampler_fingerprint(sample)
        counts = _counts(batch)
        audit_ms = (time.perf_counter_ns() - audit_started) / 1e6

        started = time.perf_counter_ns()
        batch = batch.to(device)
        logits = model(batch, entity, "reference")
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, batch[entity].y.to(logits.dtype)
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite training loss at step {step + 1}")
        loss.backward()
        gradients = [
            parameter.grad for parameter in model.parameters() if parameter.grad is not None
        ]
        if not gradients or not bool(
            torch.stack([torch.isfinite(g).all() for g in gradients]).all()
        ):
            raise FloatingPointError(f"Missing or non-finite gradients at step {step + 1}")
        gradient_count = len(gradients)
        del gradients
        optimizer.step()
        _synchronize(device)
        train_ms = (time.perf_counter_ns() - started) / 1e6
        row = {
            "step": step + 1,
            "sampler_seed": seed + step,
            "query_indices": indices,
            "query_count": len(indices),
            "loss": float(loss.detach().cpu()),
            "gradient_finite": True,
            "gradient_tensor_count": gradient_count,
            "sampler_sha256": fingerprint,
            "sampled_counts": counts,
            "visibility_audit": visibility,
            "sampling_and_gather_ms": sample_ms,
            "transfer_forward_backward_optimizer_ms": train_ms,
            "step_ms": sample_ms + train_ms,
            "audit_ms_excluded": audit_ms,
            "cuda_memory": _memory_finish(device, memory),
            "host_memory": host_memory(),
        }
        del batch, sample, logits, loss
        history.append(row)
        completed = step + 1
        if completed % checkpoint_every == 0 or completed == steps:
            save()
        if on_step is not None:
            on_step(copy.deepcopy(row))
    return {
        "status": "passed",
        "completed_steps": completed,
        "history": history,
        "checkpoint_path": str(path),
        "resumed": resume,
        "seed_scheme": SEED_SCHEME,
        "query_scheme": QUERY_SCHEME,
        "timing_boundary": TIMING_BOUNDARY,
    }


def evaluate(model, loader, entity, *, batch_size, seed, device="cpu"):
    """Evaluate every supplied query once, preserving model mode and torch RNG.

    Predictions and labels follow the loader's query order. The caller chooses
    the official validation cohort; this function never selects it using labels.
    """
    count = _validate_loader(loader, batch_size)
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if _device(model) != device:
        raise ValueError("Evaluation device must match the model's device")
    state = _rng_state()
    was_training = model.training
    logits, labels, fingerprints, counts, audits = [], [], [], [], []
    try:
        model.eval()
        with torch.inference_mode():
            for index, start in enumerate(range(0, count, batch_size)):
                _seed(seed + index, device)
                indices = list(range(start, min(start + batch_size, count)))
                sample = loader.node_sampler.sample_from_nodes(loader.input_data[indices])
                batch = loader.filter_fn(sample)
                audits.append(validate_batch(batch, entity, len(model.gnn.convs)))
                target = _labels(batch, entity)
                fingerprints.append(sampler_fingerprint(sample))
                counts.append(_counts(batch))
                labels.extend(target.detach().cpu().tolist())
                batch = batch.to(device)
                prediction = model(batch, entity, "reference")
                if prediction.ndim != 1 or prediction.numel() != len(indices):
                    raise ValueError("Expected one validation logit per query occurrence")
                if not bool(torch.isfinite(prediction).all()):
                    raise FloatingPointError("Non-finite validation logits")
                logits.extend(prediction.detach().cpu().tolist())
                del sample, batch, target, prediction
    finally:
        model.train(was_training)
        _restore_rng(state)
    probabilities = torch.tensor(logits, dtype=torch.float64).sigmoid().tolist()
    return {
        "metrics": binary_metrics(labels, probabilities),
        "predictions": probabilities,
        "logits": logits,
        "labels": labels,
        "sampler_sha256": fingerprints,
        "sampled_counts": counts,
        "visibility_audits": audits,
        "scope": "reference validation on the supplied official query cohort",
    }
