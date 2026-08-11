"""Fail-closed ensemble Monte Carlo pipeline for exp105.

exp105 measures the block logical failure rate of the frozen exp103/exp104
BP+OSD-0 decoder at readout error rate q = 0.05, over a large randomly generated
expander-code ensemble, and asks where that rate crosses as the code grows.

Two things differ from exp104 and both follow from q > 0. The decoder is given
the augmented matrix [H_Z | I] with a mixed error channel, because the readout
error is part of what has to be inferred. And a trial is scored through the
exp101 absolute logical label phi_r rather than through a residual pairing,
because the residual data error no longer has zero syndrome.

The requested observable q_top is not measurable at m >= 4 with the frozen
certified instrument; see EXPERIMENT_CONTRACT.md section 2. What exp105 delivers
towards it is a certified one-sided lower bound, plus a transport-free q_top
anchor at m = 2, 3.
"""

CONFIG_SCHEMA = "exp105.config.v1"
PILOT_CONFIG_SCHEMA = "exp105.config.pilot.v1"
REGISTRY_SCHEMA = "exp105.registry.v1"
CENSUS_SCHEMA = "exp105.census.v1"
RAW_SCHEMA = "exp105.raw.v1"
AGGREGATE_SCHEMA = "exp105.aggregate.v1"
ANCHOR_RAW_SCHEMA = "exp105.anchor.raw.v1"
ANCHOR_AGGREGATE_SCHEMA = "exp105.anchor.aggregate.v1"
EXPERIMENT_ID = "exp105.noisy_syndrome_mc.v1"

__all__ = [
    "AGGREGATE_SCHEMA",
    "ANCHOR_AGGREGATE_SCHEMA",
    "ANCHOR_RAW_SCHEMA",
    "CENSUS_SCHEMA",
    "CONFIG_SCHEMA",
    "EXPERIMENT_ID",
    "PILOT_CONFIG_SCHEMA",
    "RAW_SCHEMA",
    "REGISTRY_SCHEMA",
]
