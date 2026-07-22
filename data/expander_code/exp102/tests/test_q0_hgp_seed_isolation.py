from pathlib import Path

from data.expander_code.exp102.exp102_pipeline.io import canonical_json, sha256_json
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import (
    HGP_SCREEN_IS_ROOT,
    HGP_SCREEN_PREFLIGHT_DIGEST_ROOT,
    HGP_SCREEN_PREFLIGHT_IS_ROOT,
    HGP_SCREEN_RUNTIME_IS_ROOT,
    HGP_SCREEN_RUNTIME_TIMED_ROOT,
    HGP_SCREEN_RUNTIME_WARMUP_ROOT,
    HP_METHODS,
    INIT_FAMILIES,
    MAP_METHOD_ID,
    SCREEN_METHODS,
    TRAJECTORIES_PER_FAMILY,
    _aux_seed_identity,
    _map_cells,
    _map_is_seed,
    _method_cells,
    _seed_identity,
    load_hgp_screen_config,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_registry


EXP102_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"
CONFIG_PATH = EXP102_ROOT / "config/q0_hgp_global.screen.v2.json"
SOURCE_COMMIT = "c" * 40
ARCHIVE_SHA256 = "a" * 64
SOURCE_MANIFEST_SHA256 = "b" * 64


def _sampler_streams(seed_identity, method):
    if method in HP_METHODS:
        replicas = int(method[2:])
        for stage in ("burn", "measurement"):
            for rung in range(replicas):
                yield {
                    "stage": stage,
                    "role": "replica",
                    "index": rung,
                    "seed": seed_identity.seed(stage, "replica", rung),
                }
            yield {
                "stage": stage,
                "role": "observation",
                "index": 0,
                "seed": seed_identity.seed(stage, "observation"),
            }
    else:
        assert method == MAP_METHOD_ID
        for stage in ("burn", "measurement"):
            yield {
                "stage": stage,
                "role": "stream",
                "index": 0,
                "seed": seed_identity.seed(stage),
            }


def _append_identity_streams(rows, *, domain, purpose, case, seed_identity,
                             method, engine=None):
    identity_json = canonical_json(seed_identity.as_dict())
    for stream in _sampler_streams(seed_identity, method):
        rows.append({
            "domain": domain,
            "purpose": purpose,
            "case": case,
            "engine": engine,
            "seed_identity_json": identity_json,
            **stream,
        })


def _artifact_descriptor(cell):
    return {
        "artifact_content_sha256": sha256_json({
            "purpose": "seed_collision_regression",
            "cell": cell,
        }),
    }


def test_trajectory_and_is_63bit_streams_are_isolated_for_fixture_identity():
    registry = load_registry(REGISTRY_PATH)
    config = load_hgp_screen_config(CONFIG_PATH, registry)
    rows = []
    formal_task_count = 0

    # Enumerate every trajectory stream used by all 384 measurement tasks.
    # Disorder, character, B-character and MAP-anchor generation have separate
    # identity tests and are intentionally outside this trajectory/IS catalog.
    for method in SCREEN_METHODS:
        for cell in _method_cells(config, method):
            for family in INIT_FAMILIES:
                for trajectory in range(TRAJECTORIES_PER_FAMILY):
                    identity = _seed_identity(
                        config, registry, SOURCE_COMMIT, ARCHIVE_SHA256,
                        SOURCE_MANIFEST_SHA256, method, "T3", cell, family,
                        trajectory,
                    )
                    formal_task_count += 1
                    case = sha256_json({
                        "method": method, "cell": cell, "family": family,
                        "trajectory": trajectory,
                    })
                    if family == "U":
                        rows.append({
                            "domain": "formal",
                            "purpose": "uniform_hard_coset_initializer",
                            "case": case,
                            "engine": None,
                            "seed_identity_json": canonical_json(
                                identity.as_dict(),
                            ),
                            "stage": "initialize",
                            "role": "hard_coset",
                            "index": 0,
                            "seed": identity.seed(
                                "initialize", "hard_coset",
                            ),
                        })
                    _append_identity_streams(
                        rows, domain="formal", purpose="measurement_task",
                        case=case, seed_identity=identity, method=method,
                    )
    assert formal_task_count == 384

    descriptors = {
        sha256_json(cell): _artifact_descriptor(cell)
        for cell in _map_cells(config)
    }
    for cell in _map_cells(config):
        rows.append({
            "domain": "formal",
            "purpose": "importance_sampling",
            "case": sha256_json(cell),
            "engine": None,
            "seed_identity_json": None,
            "stage": "iid_proposal_draws",
            "role": "stream",
            "index": 0,
            "seed": _map_is_seed(
                SOURCE_COMMIT, ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256,
                config, registry, cell, descriptors[sha256_json(cell)],
                HGP_SCREEN_IS_ROOT,
            ),
        })

    # The tiny digest deliberately executes an identical identity once in the
    # reference engine and once in Numba.  These are the only duplicate uses.
    tiny_cell = {
        "code_id": "tiny", "p": 0.10, "disorder_index": 0,
        "disorder_source": "preflight",
    }
    for oracle_index in range(2):
        identity = _aux_seed_identity(
            config, registry, SOURCE_COMMIT, ARCHIVE_SHA256,
            SOURCE_MANIFEST_SHA256, "HP32", "T1", tiny_cell, "P",
            oracle_index, HGP_SCREEN_PREFLIGHT_DIGEST_ROOT,
        )
        case = f"tiny_oracle_{oracle_index}"
        for engine in ("reference", "numba"):
            _append_identity_streams(
                rows, domain="auxiliary", purpose="tiny_digest",
                case=case, seed_identity=identity, method="HP32",
                engine=engine,
            )

    benchmark_cell = next(
        cell for cell in _method_cells(config, "HP32")
        if cell["code_id"] == "m08_c06"
    )
    for method in SCREEN_METHODS:
        identity = _aux_seed_identity(
            config, registry, SOURCE_COMMIT, ARCHIVE_SHA256,
            SOURCE_MANIFEST_SHA256, method, "T1", benchmark_cell, "P", 0,
            HGP_SCREEN_PREFLIGHT_DIGEST_ROOT,
        )
        _append_identity_streams(
            rows, domain="auxiliary", purpose="hard_cell_digest",
            case=f"hard_digest_{method}", seed_identity=identity,
            method=method,
        )

    for cell in _map_cells(config):
        rows.append({
            "domain": "auxiliary",
            "purpose": "importance_sampling_preflight",
            "case": sha256_json(cell),
            "engine": None,
            "seed_identity_json": None,
            "stage": "iid_proposal_draws",
            "role": "stream",
            "index": 0,
            "seed": _map_is_seed(
                SOURCE_COMMIT, ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256,
                config, registry, cell, descriptors[sha256_json(cell)],
                HGP_SCREEN_PREFLIGHT_IS_ROOT,
            ),
        })

    for method in SCREEN_METHODS:
        for purpose, namespace in (
            ("runtime_warmup", HGP_SCREEN_RUNTIME_WARMUP_ROOT),
            ("runtime_timed", HGP_SCREEN_RUNTIME_TIMED_ROOT),
        ):
            identity = _aux_seed_identity(
                config, registry, SOURCE_COMMIT, ARCHIVE_SHA256,
                SOURCE_MANIFEST_SHA256, method, "T1", benchmark_cell, "P", 1,
                namespace,
            )
            _append_identity_streams(
                rows, domain="auxiliary", purpose=purpose,
                case=f"{purpose}_{method}", seed_identity=identity,
                method=method,
            )
    for cell in _map_cells(config):
        rows.append({
            "domain": "auxiliary",
            "purpose": "importance_sampling_runtime",
            "case": sha256_json(cell),
            "engine": None,
            "seed_identity_json": None,
            "stage": "iid_proposal_draws",
            "role": "stream",
            "index": 0,
            "seed": _map_is_seed(
                SOURCE_COMMIT, ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256,
                config, registry, cell, descriptors[sha256_json(cell)],
                HGP_SCREEN_RUNTIME_IS_ROOT,
            ),
        })

    assert len([row for row in rows if row["domain"] == "formal"]) == 31_682
    assert all(
        isinstance(row["seed"], int) and 0 <= row["seed"] < 2**63
        for row in rows
    )
    formal_seeds = {
        row["seed"] for row in rows if row["domain"] == "formal"
    }
    assert len(formal_seeds) == 31_682
    auxiliary_seeds = {
        row["seed"] for row in rows if row["domain"] == "auxiliary"
    }
    assert formal_seeds.isdisjoint(auxiliary_seeds)

    uses_by_seed = {}
    for row in rows:
        uses_by_seed.setdefault(row["seed"], []).append(row)
    collisions = [uses for uses in uses_by_seed.values() if len(uses) > 1]
    assert len(collisions) == 132
    for uses in collisions:
        assert len(uses) == 2
        assert {row["engine"] for row in uses} == {"reference", "numba"}
        assert {row["purpose"] for row in uses} == {"tiny_digest"}
        reference, accelerated = sorted(uses, key=lambda row: row["engine"])
        assert {
            key: value for key, value in reference.items() if key != "engine"
        } == {
            key: value for key, value in accelerated.items() if key != "engine"
        }
