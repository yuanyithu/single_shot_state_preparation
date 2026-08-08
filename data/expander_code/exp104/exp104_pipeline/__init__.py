"""Fail-closed ensemble Monte Carlo pipeline for exp104.

exp104 measures the q=0 code-capacity block logical failure rate of the frozen
exp103 BP+OSD-0 decoder over a large, randomly generated expander-code ensemble.
The budget is spent on codes rather than on trials per code: exp103 showed that
shot noise never binds (largest cell standard error 0.0018 against a largest
between-code standard deviation of 0.3245), so the estimand is limited by how
many codes are drawn, not by how often each one is decoded.
"""

CONFIG_SCHEMA = "exp104.config.v1"
REGISTRY_SCHEMA = "exp104.registry.v1"
CENSUS_SCHEMA = "exp104.census.v1"
RAW_SCHEMA = "exp104.raw.v1"
AGGREGATE_SCHEMA = "exp104.aggregate.v1"
EXPERIMENT_ID = "exp104.ensemble_mc.v1"

__all__ = [
    "AGGREGATE_SCHEMA",
    "CENSUS_SCHEMA",
    "CONFIG_SCHEMA",
    "EXPERIMENT_ID",
    "RAW_SCHEMA",
    "REGISTRY_SCHEMA",
]
