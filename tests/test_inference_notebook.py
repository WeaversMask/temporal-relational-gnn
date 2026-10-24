"""Run inference notebook controls without Colab, GPUs or network requests."""

import ast
import hashlib
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((PROJECT / "notebooks/06_trained_inference.ipynb").read_text())


def execute_cell(index, namespace, content_root, **settings):
    class LocalControls(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value.startswith("/content"):
                return ast.copy_location(
                    ast.Constant(str(content_root) + node.value[len("/content") :]), node
                )
            return node

        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in settings:
                    node.value = ast.parse(repr(settings[name]), mode="eval").body
            return self.generic_visit(node)

    tree = LocalControls().visit(ast.parse("".join(NOTEBOOK["cells"][index]["source"])))
    exec(compile(ast.fix_missing_locations(tree), "notebook06", "exec"), namespace)


@pytest.fixture
def notebook_vm(tmp_path, monkeypatch):
    content = tmp_path / "content"
    content.mkdir()
    namespace = {}
    downloads = []
    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=lambda name: "test-read-only-token")
    colab.files = types.SimpleNamespace(download=downloads.append)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    execute_cell(1, namespace, content, DEVICE="cpu")
    project = namespace["PROJECT_ROOT"]
    (project / "configs").mkdir(parents=True)
    for name in ("trained_inference.json", "upstream.json"):
        (project / "configs" / name).write_bytes((PROJECT / "configs" / name).read_bytes())
    report = namespace["TRAINING_REPORT"]
    report.parent.mkdir(parents=True)
    report.write_bytes(
        (PROJECT / "reports/colab/20261024T073010Z-1f9800/training-run.json").read_bytes()
    )
    return content, namespace, downloads, colab


def test_rerun_preserves_run_budget_and_checked_out_profile_source(notebook_vm, monkeypatch):
    content, namespace, _, _ = notebook_vm
    original_run = namespace["RUN_DIR"]
    original_budget = (original_run / "colab-budget.json").read_bytes()
    commands = []
    namespace["run_logged"] = lambda arguments, *args, **kwargs: commands.append(arguments)
    source = "a" * 40
    monkeypatch.setattr(namespace["subprocess"], "check_output", lambda *args, **kwargs: source)
    execute_cell(3, namespace, content)
    assert ["git", "checkout", "main"] in commands
    assert json.loads((original_run / "notebook-run.json").read_text())["source_commit"] == source
    assert "TRGNN_CLONE_TOKEN" not in namespace["clone_environment"]

    execute_cell(
        1,
        namespace,
        content,
        DEVICE="cpu",
        RESUME=True,
        COMPUTE_UNITS_REMAINING=198.0,
        PROJECT_REF="some-other-ref",
    )
    assert namespace["RUN_DIR"] == original_run
    assert (original_run / "colab-budget.json").read_bytes() == original_budget
    assert namespace["EFFECTIVE_PROJECT_REF"] == source
    commands.clear()
    namespace["run_logged"] = lambda arguments, *args, **kwargs: commands.append(arguments)
    execute_cell(3, namespace, content)
    assert ["git", "checkout", source] in commands
    assert ["git", "merge", "--ff-only", "origin/main"] not in commands
    monkeypatch.setattr(namespace["subprocess"], "check_output", lambda *args, **kwargs: "b" * 40)
    with pytest.raises(RuntimeError, match="original source commit"):
        execute_cell(3, namespace, content)


def test_frozen_parent_contract_and_complete_input_cache(notebook_vm):
    content, namespace, _, _ = notebook_vm
    execute_cell(7, namespace, content)
    assert namespace["NEEDS_DOWNLOAD"]
    frozen = namespace["frozen_config_path"].read_bytes()
    config = namespace["INFERENCE_CONFIG"]
    assert config["batch_sizes"] == [32, 128, 512]
    assert config["primary_batch_size"] == 32
    assert config["val_queries"] == 8192
    assert config["warmup"] == 5 and config["repeats"] == 20
    assert config["checkpoint_sha256"] == namespace["parent_report"]["checkpoint"]["sha256"]
    data = namespace["DATA_DIR"]
    (data / "tasks/user-badge").mkdir(parents=True)
    (data / "db").mkdir()
    (data / ".trgnn-source.json").write_text(json.dumps(namespace["pins"]["dataset"]))
    (data / "manifest.yaml").write_text("tables:\n  users: {}\n  events: {}\n")
    for relative in (
        "tasks/user-badge/manifest.yaml",
        "tasks/user-badge/train.parquet",
        "tasks/user-badge/val.parquet",
        "db/users.parquet",
        "db/events.parquet",
    ):
        (data / relative).touch()
    execute_cell(7, namespace, content)
    assert not namespace["NEEDS_DOWNLOAD"]
    (data / "db/events.parquet").unlink()
    execute_cell(7, namespace, content)
    assert namespace["NEEDS_DOWNLOAD"]
    changed = dict(config, repeats=10)
    (namespace["PROJECT_ROOT"] / "configs/trained_inference.json").write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="original frozen configuration"):
        execute_cell(7, namespace, content)
    assert namespace["frozen_config_path"].read_bytes() == frozen


def test_modified_parent_report_is_rejected_before_any_download(notebook_vm):
    content, namespace, _, _ = notebook_vm
    with namespace["TRAINING_REPORT"].open("a") as stream:
        stream.write("\n")
    with pytest.raises(RuntimeError, match="frozen report SHA256"):
        execute_cell(7, namespace, content)


def install_mock_checkpoint(namespace, colab, payload=b"own notebook 05 checkpoint"):
    namespace["INFERENCE_CONFIG"]["checkpoint_sha256"] = hashlib.sha256(payload).hexdigest()
    colab.files.upload = lambda: {"training.pt": payload}


def test_checkpoint_upload_hash_verification_and_recursive_export(notebook_vm):
    content, namespace, downloads, colab = notebook_vm
    execute_cell(7, namespace, content)
    colab.files.upload = lambda: {"pilot.pt": b"wrong smaller pilot checkpoint"}
    with pytest.raises(RuntimeError, match="Checkpoint SHA256"):
        execute_cell(9, namespace, content)
    assert not namespace["CHECKPOINT_PATH"].exists()
    install_mock_checkpoint(namespace, colab)
    execute_cell(9, namespace, content)
    checkpoint = namespace["CHECKPOINT_PATH"]
    assert checkpoint.read_bytes() == b"own notebook 05 checkpoint"
    metadata = json.loads((namespace["RUN_DIR"] / "checkpoint-input.json").read_text())
    assert metadata["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    # Cached files are checked again, not trusted merely because a previous cell passed.
    checkpoint.write_bytes(b"changed after upload")
    with pytest.raises(RuntimeError, match="Checkpoint SHA256"):
        execute_cell(9, namespace, content)
    checkpoint.write_bytes(b"own notebook 05 checkpoint")
    execute_cell(9, namespace, content)

    batch_dir = namespace["RUN_DIR"] / "batches"
    batch_dir.mkdir()
    (batch_dir / "batch-000032-000000.json").write_text('{"status": "passed"}')
    (namespace["RUN_DIR"] / "trained-profile.json").write_text('{"status": "failed"}')
    data = namespace["DATA_DIR"]
    data.mkdir(parents=True)
    (data / "private-database.parquet").write_bytes(b"not a run artifact")
    execute_cell(17, namespace, content)
    assert downloads == [namespace["archive"]]
    with zipfile.ZipFile(downloads[0]) as archive:
        assert "checkpoints/training.pt" in archive.namelist()
        assert "batches/batch-000032-000000.json" in archive.namelist()
        assert "trained-profile.json" in archive.namelist()
        assert "frozen-inference-config.json" in archive.namelist()
        assert "colab-budget.json" in archive.namelist()
        assert not any("parquet" in name for name in archive.namelist())


def test_upload_requires_exactly_one_file(notebook_vm):
    content, namespace, _, colab = notebook_vm
    execute_cell(7, namespace, content)
    colab.files.upload = lambda: {"training.pt": b"checkpoint", "data.parquet": b"database"}
    with pytest.raises(ValueError, match="single training.pt"):
        execute_cell(9, namespace, content)
    assert not namespace["CHECKPOINT_PATH"].exists()


def test_fresh_vm_restores_only_flat_batch_candidates_and_rejects_conflicts(notebook_vm, tmp_path):
    content, namespace, _, colab = notebook_vm
    execute_cell(7, namespace, content)
    install_mock_checkpoint(namespace, colab)
    restored = tmp_path / "earlier-zip/batches"
    restored.mkdir(parents=True)
    name = "batch-000032-000000.json"
    (restored / name).write_text('{"status": "passed"}')
    (restored / "other.json").write_text("{}")
    (restored / "private.parquet").write_bytes(b"database")
    (restored / "nested").mkdir()
    (restored / "nested/batch-000032-000001.json").write_text("{}")
    namespace["RESUME_ARTIFACTS_DIR"] = str(restored)
    with pytest.raises(ValueError, match="requires RESUME=True"):
        execute_cell(9, namespace, content)
    namespace["RESUME"] = True
    execute_cell(9, namespace, content)
    destination = namespace["RUN_DIR"] / "batches"
    assert [path.name for path in destination.iterdir()] == [name]
    execute_cell(9, namespace, content)  # Repeating an identical copy is safe.
    (restored / name).write_text('{"status": "changed"}')
    with pytest.raises(RuntimeError, match="Conflicting restored batch artifact"):
        execute_cell(9, namespace, content)
    assert json.loads((destination / name).read_text()) == {"status": "passed"}


def test_profiler_cli_receives_frozen_workload_paths_and_resume_flags(notebook_vm):
    content, namespace, _, _ = notebook_vm
    execute_cell(7, namespace, content)
    commands = []
    namespace["run_logged"] = lambda arguments, log: commands.append((arguments, log))
    namespace["RESUME"] = True
    execute_cell(11, namespace, content)
    arguments, log = commands[0]
    assert arguments[:3] == [sys.executable, "-m", "trgnn.trained_profile"]
    assert arguments[arguments.index("--batch-sizes") + 1 : arguments.index("--repeats")] == [
        "32",
        "128",
        "512",
    ]
    expected = {
        "--checkpoint": namespace["CHECKPOINT_PATH"],
        "--training-report": namespace["TRAINING_REPORT"],
        "--data-dir": namespace["DATA_DIR"],
        "--output": namespace["RUN_DIR"] / "trained-profile.json",
        "--artifacts-dir": namespace["RUN_DIR"] / "batches",
        "--device": "cpu",
        "--repeats": 20,
        "--warmup": 5,
        "--seed": 42,
    }
    for flag, value in expected.items():
        assert arguments[arguments.index(flag) + 1] == str(value)
    assert arguments[-2:] == ["--download", "--resume"]
    assert log == "trained-profile.log"
    assert not any("training_pilot" in argument for argument in arguments)


@pytest.mark.parametrize(
    "end,isolated,expected",
    [(198.5, True, 0.09), (198.5, False, None), (None, True, None), (199.0, True, None)],
)
def test_budget_uses_only_observed_current_interval(notebook_vm, end, isolated, expected):
    content, namespace, _, _ = notebook_vm
    budget_path = namespace["RUN_DIR"] / "colab-budget.json"
    budget_path.unlink()
    execute_cell(
        1,
        namespace,
        content,
        DEVICE="cpu",
        COMPUTE_UNITS_REMAINING=198.59,
        ISOLATED_RUNTIME_CONFIRMED=isolated,
    )
    execute_cell(15, namespace, content, ENDING_UNITS_REMAINING=end)
    budget = json.loads((namespace["RUN_DIR"] / "budget-end.json").read_text())
    assert budget["isolated_run_consumption_units"] == expected
    assert budget["scope"] == "current_runtime_interval_only"
