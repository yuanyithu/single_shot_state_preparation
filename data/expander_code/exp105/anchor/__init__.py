"""Track B: the transport-free q_top anchor, kept out of the measured package.

`exp105_pipeline` is identity-bound: its `source_tree_sha256` is recorded in the
frozen configs and in every raw file and the published aggregate. Track B shares
no code path with Track A's measurement, so putting it here keeps that identity
intact -- adding a module to `exp105_pipeline` after the measurement would make
the live tree disagree with the artifacts that were bound to it.
"""
