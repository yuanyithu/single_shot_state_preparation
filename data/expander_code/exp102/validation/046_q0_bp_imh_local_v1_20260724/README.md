# BP-IMH local hard-sentinel viability v1

This is the fresh successor to the infrastructure-failed v0 run in validation
045.  It changes only the contract/config/seed namespace and canonicalizes the
CLI output path before any file operation.  The sampler, target, P/U/L design,
fixed clock, gates, and local-diagnostic-only authority are unchanged.

No v0 task, RNG stream, raw, receipt, or result is reused.

## Terminal outcome

`LOCAL_BP_IMH_TRANSPORT_UNRESOLVED`

All 24 raw, receipt, report, deterministic runner replay, and the independent
`allow_pickle=False` raw audit completed.  The auditor independently rebuilds
the target support, proposal densities, all 55,296 MH decisions, full states,
weights, labels, and collision diagnostics without importing the sampler or
runner.  It terminates as
`INDEPENDENT_RAW_AUDIT_PASS_UNRESOLVED_CONFIRMED`.

- P and all eight distinct L trajectories make zero real burn/measurement moves.
- U makes 1--3 real burn moves and 0--2 real measurement moves per trajectory;
  every U chain reaches the same weight-62 state with the P logical label.
- P/L have `delta q_top=1` and `D2_norm=1`; U/L are `.998413` for both.
- P's best observed measurement log acceptance is at most `-53.13`; L's best
  lies between `-88.69` and `-47.79`.  Accepted U self-proposals are not motion.

Identities: config `00c36ec0c529aa9c86f85db23270a48bd204777d895fc1395d0531864240b84c`,
manifest `1292ff116d902797c3e8857c314a1398d844fdb72c017c80a1dbe23ed6ffb5fb`,
raw set `60ae69f3b829fd6037cf25979f0a55f3e74b52bc086fb988533f963ee70bc28c`,
report self-hash `62a96e7f16cbbc020f8d4e893c413bd11ec54da928893ccf23abbf6c65983c58`,
and audit self-hash `d7af8f008c500b72df512a546a051b53e1c049de5fc29a92b428cb9a35fd2ce0`.

This exact kernel and fixed budget do not proceed to HARD2 or remote testing.
Nothing here is a reportable posterior `q_top`, formal, held-out, or production
result, and the failure is not a proof that q=0 is mathematically impossible.
