"""Install a binary pyg-lib wheel matching the active PyTorch/CUDA build.

Run in Colab after the core package is installed. Failure to find a compatible
wheel is a preflight failure; never silently compile extensions during a run.
"""

import subprocess
import sys

import torch


def main():
    major, minor, *_ = torch.__version__.split("+")[0].split(".")
    torch_abi = f"{major}.{minor}.0"
    platform = "cpu" if torch.version.cuda is None else "cu" + torch.version.cuda.replace(".", "")
    wheel_index = f"https://data.pyg.org/whl/torch-{torch_abi}+{platform}.html"
    print(f"torch={torch.__version__}; wheel index={wheel_index}", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--only-binary=:all:",
            "--force-reinstall",
            "--find-links",
            wheel_index,
            "pyg_lib",
        ],
        check=True,
    )
    print(
        "Restart the runtime if PyG was imported before installation, then run sampler preflight."
    )


if __name__ == "__main__":
    main()
