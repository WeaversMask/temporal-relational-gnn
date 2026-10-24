import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from trgnn.gpu_telemetry import NvidiaSmiMonitor, gpu_identity


def identity_result(stdout="Tesla T4, GPU-test, 580.95.05, 15360\n", returncode=0):
    return subprocess.CompletedProcess([], returncode, stdout, "")


def test_cpu_is_explicitly_unavailable_without_commands(monkeypatch, tmp_path):
    run = Mock(side_effect=AssertionError("CPU must not invoke nvidia-smi"))
    monkeypatch.setattr("trgnn.gpu_telemetry.subprocess.run", run)
    assert gpu_identity("cpu")["status"] == "unavailable"
    with NvidiaSmiMonitor(tmp_path, "cpu") as monitor:
        assert monitor.metadata["status"] == "unavailable"
        assert "cpu" in monitor.metadata["reason"]
    assert list(tmp_path.iterdir()) == []
    run.assert_not_called()


def test_identity_records_driver_uuid_and_device_memory(monkeypatch):
    run = Mock(return_value=identity_result())
    monkeypatch.setattr("trgnn.gpu_telemetry.subprocess.run", run)
    assert gpu_identity("cuda") == {
        "status": "available",
        "name": "Tesla T4",
        "uuid": "GPU-test",
        "driver_version": "580.95.05",
        "total_memory_mib": 15360,
        "device_index": 0,
    }
    command = run.call_args.args[0]
    assert "--id=0" in command
    assert "--query-gpu=name,uuid,driver_version,memory.total" in command
    assert run.call_args.kwargs["timeout"] == 5


@pytest.mark.parametrize(
    "result",
    [
        identity_result(returncode=1),
        identity_result("Tesla T4, GPU-test, 580.95.05\n"),
        identity_result("Tesla T4, GPU-test, 580.95.05, nan\n"),
        identity_result("Tesla T4, GPU-test, 580.95.05, 0\n"),
        identity_result("Tesla T4, , 580.95.05, 15360\n"),
        identity_result("Tesla T4, GPU-test, 580.95.05, N/A\n"),
    ],
)
def test_invalid_identity_is_unavailable(monkeypatch, result):
    monkeypatch.setattr("trgnn.gpu_telemetry.subprocess.run", Mock(return_value=result))
    result = gpu_identity("cuda:0")
    assert result["status"] == "unavailable"
    assert result["reason"]
    assert "total_memory_mib" not in result


@pytest.mark.parametrize("error", [FileNotFoundError("absent"), subprocess.TimeoutExpired([], 5)])
def test_identity_command_failure_is_nonfatal(monkeypatch, error):
    monkeypatch.setattr("trgnn.gpu_telemetry.subprocess.run", Mock(side_effect=error))
    assert gpu_identity("cuda")["status"] == "unavailable"


def monitor_process(monkeypatch, *, polls=None, waits=None):
    monkeypatch.setattr("trgnn.gpu_telemetry.subprocess.run", Mock(return_value=identity_result()))
    process = Mock()
    process.poll.side_effect = polls or [None, None]
    process.wait.side_effect = waits or [-15]

    def launch(command, *, stdout, stderr):
        stdout.write("2026/10/24 08:00:00.000, 200, 12, 5, 30\n")
        stdout.flush()
        return process

    popen = Mock(side_effect=launch)
    monkeypatch.setattr("trgnn.gpu_telemetry.subprocess.Popen", popen)
    return process, popen


def test_monitor_writes_raw_unique_samples_and_terminates(monkeypatch, tmp_path):
    process, popen = monitor_process(monkeypatch)
    with NvidiaSmiMonitor(tmp_path, "run/unsafe", device="cuda") as monitor:
        assert monitor.metadata["status"] == "available"
        assert monitor.metadata["sampling_interval_ms"] == 1000
        assert monitor.metadata["units"]["memory.used"] == "MiB"
        assert monitor.metadata["units"]["utilization.gpu"] == "percent"
        assert "device-wide" in monitor.metadata["scope"]
        assert "not continuous peaks" in monitor.metadata["sampling_limitation"]
        assert "--loop-ms=1000" in popen.call_args.args[0]
        assert Path(monitor.metadata["csv_path"]).parent == tmp_path
    assert "200, 12, 5, 30" in Path(monitor.metadata["csv_path"]).read_text()
    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=5)
    process.kill.assert_not_called()
    assert popen.call_args.kwargs["stdout"].closed
    assert popen.call_args.kwargs["stderr"].closed
    assert monitor.metadata["status"] == "available"


def test_observer_files_are_unique_per_session(monkeypatch, tmp_path):
    paths = []
    for _ in range(2):
        monitor_process(monkeypatch)
        with NvidiaSmiMonitor(tmp_path, "same", device="cuda") as monitor:
            paths.append(monitor.metadata["csv_path"])
    assert paths[0] != paths[1]


def test_cleanup_kills_after_terminate_timeout(monkeypatch, tmp_path):
    process, _ = monitor_process(
        monkeypatch, waits=[subprocess.TimeoutExpired("nvidia-smi", 5), -9]
    )
    with NvidiaSmiMonitor(tmp_path, "timeout", device="cuda"):
        pass
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.wait.call_count == 2


def test_benchmark_exception_is_preserved_during_cleanup_failure(monkeypatch, tmp_path):
    process, popen = monitor_process(monkeypatch)
    process.terminate.side_effect = ProcessLookupError("already gone")
    monitor = NvidiaSmiMonitor(tmp_path, "exception", device="cuda")
    with pytest.raises(RuntimeError, match="benchmark failure"), monitor:
        raise RuntimeError("benchmark failure")
    assert monitor.metadata["status"] == "unavailable"
    assert "Could not stop" in monitor.metadata["reason"]
    assert popen.call_args.kwargs["stdout"].closed
    assert popen.call_args.kwargs["stderr"].closed


def test_subprocess_start_failure_is_nonfatal(monkeypatch, tmp_path):
    monkeypatch.setattr("trgnn.gpu_telemetry.subprocess.run", Mock(return_value=identity_result()))
    popen = Mock(side_effect=FileNotFoundError("missing command"))
    monkeypatch.setattr("trgnn.gpu_telemetry.subprocess.Popen", popen)
    with NvidiaSmiMonitor(tmp_path, "failed", device="cuda") as monitor:
        assert monitor.metadata["status"] == "unavailable"
    assert "Could not start" in monitor.metadata["reason"]
    assert popen.call_args.kwargs["stdout"].closed
    assert popen.call_args.kwargs["stderr"].closed


@pytest.mark.parametrize("polls", [[1, 1], [None, 7]])
def test_subprocess_early_or_later_failure_is_reported(monkeypatch, tmp_path, polls):
    process, _ = monitor_process(monkeypatch, polls=polls)
    with NvidiaSmiMonitor(tmp_path, "exit", device="cuda") as monitor:
        pass
    assert monitor.metadata["status"] == "unavailable"
    assert "exited" in monitor.metadata["reason"]
    process.terminate.assert_not_called()


def test_no_samples_is_explicitly_unavailable(monkeypatch, tmp_path):
    process, popen = monitor_process(monkeypatch)

    def launch(command, *, stdout, stderr):
        return process

    popen.side_effect = launch
    with NvidiaSmiMonitor(tmp_path, "empty", device="cuda") as monitor:
        pass
    assert monitor.metadata["status"] == "unavailable"
    assert "no samples" in monitor.metadata["reason"]


def test_output_directory_failure_is_nonfatal(monkeypatch, tmp_path):
    monkeypatch.setattr("trgnn.gpu_telemetry.subprocess.run", Mock(return_value=identity_result()))
    target = tmp_path / "already-a-file"
    target.write_text("existing")
    with NvidiaSmiMonitor(target, "bad-dir", device="cuda") as monitor:
        assert monitor.metadata["status"] == "unavailable"
    assert "Could not start" in monitor.metadata["reason"]
