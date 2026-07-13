# PT and aggregation integration evidence

- Real four-instance PT task: `INVALID` as designed.
- Forced fresh execution: `computed=1`, `reused=0`.
- Round trips per instance: `[0, 0, 0, 0]`.
- Burn-in/measurement round trips: `[0, 0, 0, 0]` / `[0, 0, 0, 0]`.
- Failure includes `pt_instance_round_trips_insufficient`.
- Invalid-only PT mean: `NaN` (serialized as `null`).
- Complete ladder/swap/round-trip/cold/gate schema: PASS.
- Synthetic valid/invalid aggregate counts: `{'valid': 2, 'invalid': 1, 'missing': 0}`.
- Valid-only mean/SEM: `0.30000000000000004` / `0.1` (invalid value `0.95` excluded).
- Crossing input: `[0.2, 0.4, nan]` (invalid entry is NaN).
- Task/source fingerprints: `d46c6c0c4cd8f5c95e43e2ff06d0538f0551bd9a90d8625679db022acc8a782b` / `bc3867f359bdc14e2dee10535e21064c75df21d0ebef5b5102d973bd5d688ae2`.
