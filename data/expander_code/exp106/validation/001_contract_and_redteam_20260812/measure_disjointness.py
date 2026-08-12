"""Measure that exp106's ensemble shares no code with exp104's or exp105's.

Disjointness is meant to hold *by construction*: the candidate seed is
`sha256(master_seed : namespace : m : candidate_index)`, and exp106's master seed
is neither of theirs. But "by construction" is an argument, and the point of a
validation directory is to hold measurements. This compares the actual accepted
parity-check matrices, which is what a collision would have to show up in.

It also reports the acceptance rates, which are a property of the *rule* rather
than of the seed, so they should agree across all three experiments to within
sampling error. Agreement is evidence the port did not change the ensemble;
disjointness is evidence it did not reuse it.

Run from the repository root:

    conda run -n 12 --no-capture-output python \\
      data/expander_code/exp106/validation/001_contract_and_redteam_20260812/measure_disjointness.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from data.expander_code.exp106.exp106_pipeline.io import atomic_json, sha256_json  # noqa: E402


def _hashes(load, path):
    registry = load(str(REPO_ROOT / path))
    by_m = {}
    for row in registry["codes"]:
        by_m.setdefault(int(row["m"]), set()).add(str(row["classical_H_sha256"]))
    return registry, by_m


def main():
    from data.expander_code.exp104.exp104_pipeline.ensemble import (
        load_registry as load_exp104,
    )
    from data.expander_code.exp105.exp105_pipeline.ensemble import (
        load_registry as load_exp105,
    )
    from data.expander_code.exp106.exp106_pipeline.ensemble import (
        load_registry as load_exp106,
    )

    sources = {
        "exp104": _hashes(
            load_exp104, "data/expander_code/exp104/config/ensemble_registry.v1.json",
        ),
        "exp105": _hashes(
            load_exp105, "data/expander_code/exp105/config/ensemble_registry.v1.npz",
        ),
        "exp106_pilot": _hashes(
            load_exp106,
            "data/expander_code/exp106/config/ensemble_registry.pilot.v1.npz",
        ),
    }

    _, ours = sources["exp106_pilot"]
    overlaps = {}
    for name in ("exp104", "exp105"):
        _, theirs = sources[name]
        shared = {}
        for m, codes in ours.items():
            common = codes & theirs.get(m, set())
            if common:
                shared[str(m)] = sorted(common)
        overlaps[name] = shared

    core = {
        "schema_version": "exp106.disjointness.v1",
        "question": (
            "does any code accepted into an exp106 panel also appear in exp104's "
            "or exp105's frozen registries?"
        ),
        "registry_sha256": {
            name: sources[name][0]["registry_sha256"] for name in sources
        },
        "codes_compared": {
            name: sum(len(v) for v in sources[name][1].values()) for name in sources
        },
        "exp106_codes_by_m": {
            str(m): len(codes) for m, codes in sorted(ours.items())
        },
        "shared_codes": overlaps,
        "status": (
            "DISJOINT" if not any(overlaps.values()) else "OVERLAP_DETECTED"
        ),
    }
    report = dict(core, report_sha256=sha256_json(core))
    atomic_json(Path(__file__).resolve().parent / "disjointness.json", report)
    print(json.dumps(
        {k: report[k] for k in ("status", "codes_compared", "shared_codes")},
        sort_keys=True, indent=2,
    ))
    return 0 if report["status"] == "DISJOINT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
