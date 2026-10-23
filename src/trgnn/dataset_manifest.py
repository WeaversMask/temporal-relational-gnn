"""Fetch pinned public schema/task manifests, without downloading the database."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from trgnn.artifacts import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/dataset-manifests.json"))
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    pins = json.loads((project_root / "configs/upstream.json").read_text())
    records = {}
    for name, path in {
        "database": "rel-stack/manifest.yaml",
        "task": "rel-stack/tasks/user-badge/manifest.yaml",
    }.items():
        url = (
            f"https://huggingface.co/datasets/{pins['dataset']['repository']}/resolve/"
            f"{pins['dataset']['revision']}/{path}"
        )
        with urllib.request.urlopen(url, timeout=60) as response:
            body = response.read()
        records[name] = {
            "url": url,
            "sha256": hashlib.sha256(body).hexdigest(),
            "manifest": body.decode("utf-8"),
        }
    write_json(args.output, {"dataset": pins["dataset"], "files": records})
    print(f"Saved pinned manifests to {args.output}; full database was not downloaded.")


if __name__ == "__main__":
    main()
