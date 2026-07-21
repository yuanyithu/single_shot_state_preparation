"""Freeze the diagnostic 24-hour schedule without a Python heredoc."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import (
    DEFAULT_CONFIG_RELATIVE,
    DEFAULT_REGISTRY_RELATIVE,
    freeze_schedule,
    sha256_file,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_RELATIVE))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_RELATIVE))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("fresh diagnostic schedule output already exists")
    schedule = freeze_schedule(
        args.registry, args.config, args.source_commit, args.archive_sha256,
        args.manifest_sha256, output,
    )
    print(json.dumps({
        "status": schedule["status"],
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_file_sha256": sha256_file(output),
        "started_unix": schedule["started_unix"],
        "deadlines_unix": schedule["deadlines_unix"],
        "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
