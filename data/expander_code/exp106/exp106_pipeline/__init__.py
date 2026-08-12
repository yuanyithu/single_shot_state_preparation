"""Fail-closed ensemble Monte Carlo pipeline for exp106.

exp106 measures the block logical failure rate of the frozen exp103/exp104
BP+OSD-0 decoder at readout error rate q = 0.01, over a large randomly generated
expander-code ensemble, and asks whether that rate still crosses as the code
grows.

It exists because exp104 and exp105 bracketed the question without answering it.
exp104 certified a crossing at p = 0.05512 with perfect readout; exp105 certified
no crossing at any p in [0.001, 0.07] with q = 0.05. The readout threshold of
this decoder on this family therefore lies strictly inside (0, 0.05), and exp106
measures one interior point of that interval.

Two things differ from exp104 and both follow from q > 0. The decoder is given
the augmented matrix [H_Z | I] with a mixed error channel, because the readout
error is part of what has to be inferred. And a trial is scored through the
exp101 absolute logical label phi_r rather than through a residual pairing,
because the residual data error no longer has zero syndrome.

The requested observable q_top is not measurable at m >= 4 with the frozen
certified instrument; see EXPERIMENT_CONTRACT.md section 2. What exp106 delivers
towards it is a certified one-sided lower bound. There is no Track B anchor here:
exp105 established that the full-sector TI instrument cannot certify one, and
permanent discipline 13 forbids extending that family of attempt.
"""

CONFIG_SCHEMA = "exp106.config.v1"
PILOT_CONFIG_SCHEMA = "exp106.config.pilot.v1"
PILOT_REMOTE_CONFIG_SCHEMA = "exp106.config.pilot.remote.v1"
REGISTRY_SCHEMA = "exp106.registry.v1"
CENSUS_SCHEMA = "exp106.census.v1"
RAW_SCHEMA = "exp106.raw.v1"
AGGREGATE_SCHEMA = "exp106.aggregate.v1"
EXPERIMENT_ID = "exp106.noisy_syndrome_mc.q001.v1"

__all__ = [
    "AGGREGATE_SCHEMA",
    "CENSUS_SCHEMA",
    "CONFIG_SCHEMA",
    "EXPERIMENT_ID",
    "PILOT_CONFIG_SCHEMA",
    "PILOT_REMOTE_CONFIG_SCHEMA",
    "RAW_SCHEMA",
    "REGISTRY_SCHEMA",
]
