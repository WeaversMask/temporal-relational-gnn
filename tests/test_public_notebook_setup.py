"""Exercise public and explicit private Colab setup without external services."""

import ast
import json
import os
import stat
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOKS = tuple(sorted((PROJECT / "notebooks").glob("*.ipynb")))
SOURCE_COMMIT = "a" * 40


def execute_setup(notebook, namespace, content_root, *, authenticated=False):
    cells = json.loads(notebook.read_text())["cells"]
    setup = next(
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code" and "clone_environment" in "".join(cell["source"])
    )

    class LocalSetup(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value.startswith("/content"):
                return ast.copy_location(
                    ast.Constant(str(content_root) + node.value[len("/content") :]), node
                )
            return node

        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == "USE_GITHUB_TOKEN":
                    node.value = ast.Constant(authenticated)
            return self.generic_visit(node)

    tree = LocalSetup().visit(ast.parse(setup))
    exec(compile(ast.fix_missing_locations(tree), notebook.name, "exec"), namespace)


@pytest.fixture
def setup_vm(tmp_path, monkeypatch):
    content = tmp_path / "content"
    content.mkdir()
    project = content / "temporal-relational-gnn"
    run = content / "run"
    run.mkdir()
    calls, helpers, secret_calls = [], [], []
    colab = types.ModuleType("google.colab")

    def get_secret(name):
        secret_calls.append(name)
        return "test-private-read-only-token"

    colab.userdata = types.SimpleNamespace(get=get_secret)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    # Anonymous setup must not inherit a secret-bearing authentication helper.
    monkeypatch.setenv("GIT_ASKPASS", "/tmp/inherited-auth-helper")
    monkeypatch.setenv("SSH_ASKPASS", "/tmp/inherited-ssh-helper")
    monkeypatch.setenv("TRGNN_CLONE_TOKEN", "inherited-token")

    def run_command(arguments, *, env=None, **kwargs):
        snapshot = dict(env) if env is not None else None
        calls.append((list(arguments), snapshot))
        if snapshot is not None and "GIT_ASKPASS" in snapshot:
            helper = Path(snapshot["GIT_ASKPASS"])
            assert helper.is_file()
            assert stat.S_IMODE(helper.stat().st_mode) == 0o700
            assert "test-private-read-only-token" not in helper.read_text()
            helpers.append(helper)
        failure = namespace.get("simulate_failure")
        if snapshot is not None and failure is not None:
            raise failure
        return types.SimpleNamespace(returncode=0)

    def run_logged(arguments, log_name, **kwargs):
        return run_command(arguments, **kwargs)

    namespace = {
        "os": os,
        "subprocess": subprocess,
        "sys": sys,
        "tempfile": tempfile,
        "Path": Path,
        "json": json,
        "PROJECT_ROOT": project,
        "PROJECT_REF": "main",
        "EFFECTIVE_PROJECT_REF": SOURCE_COMMIT,
        "REPO_URL": "https://github.com/WeaversMask/temporal-relational-gnn.git",
        "RUN_DIR": run,
        "run_logged": run_logged,
        "run_metadata": {"source_commit": SOURCE_COMMIT},
        "metadata_path": run / "notebook-run.json",
    }
    monkeypatch.setattr(subprocess, "run", run_command)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: SOURCE_COMMIT)
    return types.SimpleNamespace(
        content=content,
        project=project,
        run=run,
        namespace=namespace,
        calls=calls,
        helpers=helpers,
        colab=colab,
        secret_calls=secret_calls,
    )


def clone_call(vm):
    return next((arguments, env) for arguments, env in vm.calls if env is not None)


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
@pytest.mark.parametrize("existing", [False, True], ids=["clone", "fetch"])
@pytest.mark.parametrize("secret_state", ["missing", "denied"])
def test_public_setup_never_reads_missing_or_denied_colab_secret(
    setup_vm, notebook, existing, secret_state
):
    vm = setup_vm
    if existing:
        vm.project.mkdir()
    if secret_state == "missing":
        del vm.colab.userdata
    else:

        def denied(name):
            vm.secret_calls.append(name)
            raise RuntimeError("Colab secret access denied")

        vm.colab.userdata.get = denied
    execute_setup(notebook, vm.namespace, vm.content)
    arguments, env = clone_call(vm)
    assert not vm.secret_calls
    assert "clone" in arguments if not existing else "fetch" in arguments
    assert "credential.helper=" in arguments
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert not any(key in env for key in ("TRGNN_CLONE_TOKEN", "GIT_ASKPASS", "SSH_ASKPASS"))
    assert vm.namespace["token"] is None
    assert vm.namespace["USE_GITHUB_TOKEN"] is False
    assert not vm.helpers
    assert all("token" not in str(argument) for argument in arguments)
    assert any("checkout" in command for command, _ in vm.calls)


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
@pytest.mark.parametrize("existing", [False, True], ids=["clone", "fetch"])
def test_private_setup_requires_explicit_flag_and_cleans_temporary_helper(
    setup_vm, notebook, existing
):
    vm = setup_vm
    if existing:
        vm.project.mkdir()
    execute_setup(notebook, vm.namespace, vm.content, authenticated=True)
    arguments, env = clone_call(vm)
    assert vm.secret_calls == ["GITHUB_TOKEN"]
    assert "clone" in arguments if not existing else "fetch" in arguments
    assert "credential.helper=" in arguments
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["TRGNN_CLONE_TOKEN"] == "test-private-read-only-token"
    assert all("test-private-read-only-token" not in str(argument) for argument in arguments)
    assert vm.namespace["token"] is None
    assert "TRGNN_CLONE_TOKEN" not in vm.namespace["clone_environment"]
    assert "GIT_ASKPASS" not in vm.namespace["clone_environment"]
    assert vm.helpers and all(not path.exists() for path in vm.helpers)


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
@pytest.mark.parametrize("authenticated", [False, True], ids=["public", "private"])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_setup_failure_and_interrupt_always_remove_secret_and_helper(
    setup_vm, notebook, authenticated, failure
):
    vm = setup_vm
    vm.namespace["simulate_failure"] = failure("interrupted setup")
    with pytest.raises(failure, match="interrupted setup"):
        execute_setup(notebook, vm.namespace, vm.content, authenticated=authenticated)
    assert vm.namespace["token"] is None
    assert "TRGNN_CLONE_TOKEN" not in vm.namespace["clone_environment"]
    assert "GIT_ASKPASS" not in vm.namespace["clone_environment"]
    assert all(not path.exists() for path in vm.helpers)
    assert vm.secret_calls == (["GITHUB_TOKEN"] if authenticated else [])
    assert not any("checkout" in command for command, _ in vm.calls)
    if notebook.name[:2] not in {"01", "02"} and failure is RuntimeError:
        failure_record = json.loads((vm.run / "setup-failure.json").read_text())
        assert failure_record == {"stage": "clone", "error_type": "RuntimeError"}


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_explicit_private_secret_denial_stops_without_retaining_credentials(setup_vm, notebook):
    vm = setup_vm

    def denied(name):
        vm.secret_calls.append(name)
        raise RuntimeError("Colab secret access denied")

    vm.colab.userdata.get = denied
    with pytest.raises(RuntimeError, match="Colab secret access denied"):
        execute_setup(notebook, vm.namespace, vm.content, authenticated=True)
    assert vm.secret_calls == ["GITHUB_TOKEN"]
    assert not vm.calls
    assert vm.namespace["token"] is None
    assert "TRGNN_CLONE_TOKEN" not in vm.namespace["clone_environment"]
    assert "GIT_ASKPASS" not in vm.namespace["clone_environment"]
