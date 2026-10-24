"""Execute Colab controls with a temporary VM root and no external services."""

import ast
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((PROJECT / "notebooks/05_training_run.ipynb").read_text())


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
    exec(compile(ast.fix_missing_locations(tree), "notebook05", "exec"), namespace)


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
    return content, namespace, downloads, colab


def test_rerun_preserves_run_and_budget_and_pins_checked_out_source(notebook_vm, monkeypatch):
    content, namespace, _, _ = notebook_vm
    execute_cell(
        1,
        namespace,
        content,
        DEVICE="cpu",
        COMPUTE_UNITS_REMAINING=198.63,
        COMPUTE_UNITS_PER_HOUR=1.07,
        ISOLATED_RUNTIME_CONFIRMED=True,
    )
    run_dir, run_id = namespace["RUN_DIR"], namespace["run_id"]
    original_budget = (run_dir / "colab-budget.json").read_bytes()
    commands = []
    namespace["run_logged"] = lambda arguments, *args, **kwargs: commands.append(arguments)
    source = "a" * 40
    monkeypatch.setattr(namespace["subprocess"], "check_output", lambda *args, **kwargs: source)
    execute_cell(3, namespace, content)
    assert ["git", "checkout", "main"] in commands
    assert json.loads((run_dir / "notebook-run.json").read_text())["source_commit"] == source
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
    assert namespace["RUN_DIR"] == run_dir and namespace["run_id"] == run_id
    assert (run_dir / "colab-budget.json").read_bytes() == original_budget
    assert namespace["EFFECTIVE_PROJECT_REF"] == source
    commands.clear()
    namespace["run_logged"] = lambda arguments, *args, **kwargs: commands.append(arguments)
    execute_cell(3, namespace, content)
    assert ["git", "checkout", source] in commands
    assert ["git", "merge", "--ff-only", "origin/main"] not in commands

    monkeypatch.setattr(namespace["subprocess"], "check_output", lambda *args, **kwargs: "b" * 40)
    with pytest.raises(RuntimeError, match="original source commit"):
        execute_cell(3, namespace, content)
    assert json.loads((run_dir / "notebook-run.json").read_text())["source_commit"] == source


def test_frozen_config_and_complete_input_cache(notebook_vm):
    content, namespace, _, _ = notebook_vm
    execute_cell(1, namespace, content, DEVICE="cpu")
    project, data = namespace["PROJECT_ROOT"], namespace["DATA_DIR"]
    (project / "configs").mkdir(parents=True)
    for name in ("training_run.json", "upstream.json"):
        (project / "configs" / name).write_bytes((PROJECT / "configs" / name).read_bytes())
    execute_cell(7, namespace, content)
    assert namespace["NEEDS_DOWNLOAD"]
    frozen = namespace["frozen_config_path"].read_bytes()
    config = namespace["TRAINING_CONFIG"]
    assert config["steps"] * config["batch_size"] == config["train_queries"] == 64_000
    assert config["val_queries"] == 8192

    pins = namespace["pins"]
    (data / "tasks/user-badge").mkdir(parents=True)
    (data / "db").mkdir()
    (data / ".trgnn-source.json").write_text(json.dumps(pins["dataset"]))
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
    (data / "tasks/user-badge/train.parquet").unlink()
    execute_cell(7, namespace, content)
    assert namespace["NEEDS_DOWNLOAD"]

    changed = dict(config, steps=1000)
    (project / "configs/training_run.json").write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="original frozen configuration"):
        execute_cell(7, namespace, content)
    assert namespace["frozen_config_path"].read_bytes() == frozen


def test_resume_checkpoint_and_export_include_run_artifacts_only(notebook_vm):
    content, namespace, downloads, colab = notebook_vm
    execute_cell(1, namespace, content, DEVICE="cpu")
    execute_cell(9, namespace, content)
    checkpoint = namespace["CHECKPOINT_PATH"]
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"own saved training checkpoint")
    with pytest.raises(RuntimeError, match="Set RESUME=True"):
        execute_cell(9, namespace, content)
    execute_cell(1, namespace, content, DEVICE="cpu", RESUME=True)
    execute_cell(9, namespace, content)
    assert namespace["RESUME_CHECKPOINT"] == checkpoint
    checkpoint.unlink()
    namespace["RESUME_CHECKPOINT"] = None
    colab.files.upload = lambda: {"training.pt": b"restored own training checkpoint"}
    execute_cell(9, namespace, content)
    assert checkpoint.read_bytes() == b"restored own training checkpoint"

    data = namespace["DATA_DIR"]
    data.mkdir(parents=True)
    (data / "private-database.parquet").write_bytes(b"not an exported run artifact")
    (namespace["RUN_DIR"] / "training-run.json").write_text('{"status": "failed"}')
    execute_cell(17, namespace, content)
    assert downloads == [namespace["archive"]]
    with zipfile.ZipFile(downloads[0]) as archive:
        assert "checkpoints/training.pt" in archive.namelist()
        assert "training-run.json" in archive.namelist()
        assert "colab-budget.json" in archive.namelist()
        assert not any("parquet" in name for name in archive.namelist())


@pytest.mark.parametrize(
    "end,isolated,expected",
    [(198.5, True, 0.13), (198.5, False, None), (None, True, None), (199.0, True, None)],
)
def test_budget_delta_uses_observed_isolated_runtime_interval(notebook_vm, end, isolated, expected):
    content, namespace, _, _ = notebook_vm
    execute_cell(
        1,
        namespace,
        content,
        DEVICE="cpu",
        COMPUTE_UNITS_REMAINING=198.63,
        ISOLATED_RUNTIME_CONFIRMED=isolated,
    )
    execute_cell(15, namespace, content, ENDING_UNITS_REMAINING=end)
    budget = json.loads((namespace["RUN_DIR"] / "budget-end.json").read_text())
    assert budget["isolated_run_consumption_units"] == expected
    assert budget["scope"] == "current_runtime_interval_only"
