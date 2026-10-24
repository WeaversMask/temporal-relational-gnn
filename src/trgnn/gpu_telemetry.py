"""Optional device-wide NVIDIA telemetry alongside an inference experiment.

The observer writes raw one-second samples directly to disk. These sampled
values are distinct from PyTorch allocator measurements and continuous peaks.
Its failure must not prevent an otherwise valid inference benchmark.
"""

import csv
import math
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SAMPLE_FIELDS = (
    "timestamp",
    "memory.used",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
)
SAMPLING_INTERVAL_MS = 1000


def _cuda_device_zero(device):
    return str(device) in ("cuda", "cuda:0")


def gpu_identity(device):
    """Read the physical GPU 0 identity once, outside benchmark timing."""
    if not _cuda_device_zero(device):
        return {
            "status": "unavailable",
            "reason": "NVIDIA telemetry requires CUDA device 0; this run uses " + str(device),
        }
    command = [
        "nvidia-smi",
        "--id=0",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=5)
        if result.returncode != 0:
            return {
                "status": "unavailable",
                "reason": f"nvidia-smi identity query exited with code {result.returncode}",
            }
        rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
        if len(rows) != 1 or len(rows[0]) != 4:
            raise ValueError("Expected one GPU identity row with four fields")
        name, uuid, driver, memory = (field.strip() for field in rows[0])
        total_memory = float(memory)
        if not all((name, uuid, driver)) or not math.isfinite(total_memory) or total_memory <= 0:
            raise ValueError("Missing or invalid GPU identity field")
        return {
            "status": "available",
            "name": name,
            "uuid": uuid,
            "driver_version": driver,
            "total_memory_mib": total_memory,
            "device_index": 0,
        }
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {"status": "unavailable", "reason": f"GPU identity query failed: {error}"}


class NvidiaSmiMonitor:
    """Start one optional observer before timing and clean it up on exit.

    Metadata remains available after the context exits, including failure reasons
    and raw-file paths. No statistics or substitute values are fabricated when
    the command or a sampled counter is unavailable.
    """

    def __init__(self, output_dir, session_id, *, device="cpu"):
        self.output_dir = Path(output_dir)
        self.session_id = str(session_id)
        self.device = device
        self.process = None
        self._stdout = None
        self._stderr = None
        self.metadata = {
            "status": "unavailable",
            "reason": "Monitor has not started",
            "sampling_interval_ms": SAMPLING_INTERVAL_MS,
            "observer_host_timezone": str(datetime.now().astimezone().tzinfo),
            "fields": list(SAMPLE_FIELDS),
            "units": {
                "timestamp": "nvidia-smi wall-clock timestamp",
                "memory.used": "MiB",
                "utilization.gpu": "percent",
                "utilization.memory": "percent",
                "power.draw": "W",
            },
            "scope": "sampled device-wide values, including other processes and driver memory",
            "observer_note": (
                "One nvidia-smi observer samples every second; observer overhead is present "
                "during measurement and is not subtracted from benchmark times"
            ),
            "sampling_limitation": (
                "One-second observations can miss short activity; sampled maxima are not "
                "continuous peaks and are distinct from PyTorch allocator peaks"
            ),
        }

    def _unavailable(self, reason):
        self.metadata["status"] = "unavailable"
        self.metadata["reason"] = reason

    def __enter__(self):
        identity = gpu_identity(self.device)
        self.metadata["identity"] = identity
        if identity["status"] != "available":
            self._unavailable(identity["reason"])
            return self
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            safe_session = re.sub(r"[^A-Za-z0-9_.-]", "_", self.session_id)[:80] or "session"
            self._stdout = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.output_dir,
                prefix=f"nvidia-smi-{safe_session}-",
                suffix=".csv",
                delete=False,
            )
            csv_path = Path(self._stdout.name).resolve()
            stderr_path = csv_path.with_suffix(".stderr.log")
            self.metadata["csv_path"] = str(csv_path)
            self.metadata["stderr_path"] = str(stderr_path)
            self._stderr = stderr_path.open("x", encoding="utf-8")
            command = [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=" + ",".join(SAMPLE_FIELDS),
                "--format=csv,noheader,nounits",
                f"--loop-ms={SAMPLING_INTERVAL_MS}",
            ]
            self.metadata["command"] = command
            self.metadata["observer_started_at_utc"] = datetime.now(timezone.utc).isoformat()
            self.process = subprocess.Popen(command, stdout=self._stdout, stderr=self._stderr)
            returncode = self.process.poll()
            if returncode is not None:
                self.metadata["process_returncode"] = returncode
                self._unavailable(f"nvidia-smi observer exited before measurement: {returncode}")
                self._close_files()
                return self
            self.metadata["status"] = "available"
            self.metadata.pop("reason", None)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            self._unavailable(f"Could not start nvidia-smi observer: {error}")
            self._stop()
        return self

    def _close_files(self):
        for stream in (self._stdout, self._stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError as error:
                    self._unavailable(f"Could not close nvidia-smi output: {error}")

    def _stop(self):
        try:
            if self.process is not None:
                returncode = self.process.poll()
                if returncode is None:
                    self.process.terminate()
                    try:
                        returncode = self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        returncode = self.process.wait(timeout=5)
                elif returncode != 0:
                    self._unavailable(
                        f"nvidia-smi observer exited during measurement: {returncode}"
                    )
                self.metadata["process_returncode"] = returncode
        except (OSError, subprocess.SubprocessError) as error:
            self._unavailable(f"Could not stop nvidia-smi observer: {error}")
        finally:
            self._close_files()

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop()
        if self.metadata["status"] == "available":
            try:
                if Path(self.metadata["csv_path"]).stat().st_size == 0:
                    self._unavailable("nvidia-smi observer recorded no samples")
            except OSError as error:
                self._unavailable(f"Could not inspect nvidia-smi samples: {error}")
        return False
