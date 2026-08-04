# Validation 050: exact full-column bridge between truth-free MAP B basins

Validation 049 showed that P/L share one nearly frozen B mask while exact-K0 U
remains far from it.  This no-sample structural probe asks a sharper question:
can one-column exact heatbaths connect the two frozen, non-planted,
minimum-weight MAP anchors already present in the immutable HGP-v2 artifact?

Both anchors have physical weight 62 and the same logical label, but their B
masks differ in six bits: one three-bit vector in each of columns 11 and 17.
For both bridge orders and both directions, the probe computes the exact full
conditional probability of selecting the next bridge column.  The action gate
requires at least four expected first bridge departures per anchor in the
entire 10,240-update T1 random-scan clock.  This is only a necessary structural
gate; passing would not establish mixing or a posterior.

The artifact, cell, exact mass engine, clock, and gate are frozen in
`q0_full_column_map_bridge.structure.v0.json`.  The probe uses no planted
error, chain raw, q_top, or result-dependent anchor selection.
