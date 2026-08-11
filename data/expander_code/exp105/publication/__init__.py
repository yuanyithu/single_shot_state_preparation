"""Post-measurement presentation, kept out of the identity-bound package.

`exp105_pipeline` is frozen: its `source_tree_sha256` is recorded in the configs,
in every raw file and in the published aggregate, and the loader checks it. So
the report generator that was corrected *after* the measurement lives here
instead. It reads the published aggregate through the loader and never touches
raw, the decoder or the seeds, so it cannot change what was measured -- only how
it is displayed.

`exp105_pipeline/report.py` remains exactly as it was when the measurement ran,
carrying the three defects that only surface at report time. Those defects are
recorded in validation/006; they are not fixed in place, because fixing them in
place would orphan the published aggregate.
"""
