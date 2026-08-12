# Validation 004: nd-3 qualification and resource gate

Status: **`BLOCKED_REMOTE_RESOURCE_PREFLIGHT`**. Controlled category:
`REMOTE_RESOURCE_GATE`.

Qualification passed. The resource projection did not. **No production task has
run**, and per contract section 6 the experiment stops here and reports rather
than reducing the panel to fit.

## Environment qualification: `PASS`

`environment_qualification.json`, run on nd-3 through the verified-source chain.

| group | passed | expected | skipped | xfailed | xpassed | deselected |
|---|---:|---:|---:|---:|---:|---:|
| exp106 | 223 | 223 | 0 | 0 | 0 | 0 |
| exp105 | 166 | 166 | 0 | 0 | 0 | 0 |
| exp104 | 131 | 131 | 0 | 0 | 0 | 0 |
| exp101 | 58 | 58 | 0 | 0 | 0 | 0 |
| exp102 | 17 | 17 | 0 | 0 | 0 | 0 |

Source tree bytecode-clean before and after. Decoder binary hashes to the pinned
`3a5a7dc2...`. Host: 96 logical CPUs on 48 physical cores, 503 GiB, against a
frozen worker count of 72.

**First attempt, no retries.** exp105's first qualification failed and took four
separate defects to clear; exp106 inherited all four fixes — the
`EXP10x_TEST_CONFIG_PATH` override for every carried suite, the registry tracked
rather than gitignored, pass counts measured rather than guessed, and the two
preflight entry points agreeing on their benchmark grid.

The preflight's first invocation refused with "remote environment qualification
is not present in the deployed pushed source". That is the gate working: the
report has to be committed and re-deployed so the run-root copy can be compared
byte for byte against the archive. It was, and the second invocation proceeded.

## Resource preflight: `BLOCKED_RESOURCE_PREFLIGHT`

`remote_resource_preflight_BLOCKED.json`. Retained as evidence; a failed gate is
not deleted and is not rerun in place.

| quantity | projected | cap | verdict |
|---|---:|---:|---|
| reserved core-hours | **2001.95** | 1800 | **FAIL** |
| predicted wall hours | 15.87 | 20 | pass |
| projected peak RSS | 55.5 GiB | 128 | pass |

Composition: generation `904.68` + committed replay `94.30` + analysis `1` +
overhead `1`, times the discipline-11 reserve multiplier of 2.

Only the core-hour cap fails, by 11.2 percent.

## Why, and whose fault it is

**The cap was set too tight, and that is a defect in the contract I wrote.**
Section 10 derived it as `2 x (800 + 80 + 1 + 1) = 1764` and I rounded to 1800.
That leaves 2 percent of slack. But the allocation rule *spends the entire frozen
generation budget by construction* — it solved for 799.8 of 800 — so the
projection starts at the cap and any upward drift in re-measured cost fails. A
2 percent margin was never going to survive an independent re-benchmark.

**The drift is benchmark noise, not a real cost increase.** The preflight
re-measures the same quantities the Validation 003 benchmark measured, minutes
apart, on a machine another user holds ten cores on. The per-`m` ratios between
the two measurements:

| m | preflight `c_m` | Validation 003 `c_m` | ratio |
|---|---:|---:|---:|
| 3 | 0.1300 | 0.0825 | 1.576 |
| 4 | 0.4441 | 0.4226 | 1.051 |
| 5 | 0.7279 | 1.0564 | **0.689** |
| 6 | 2.1311 | 2.2251 | 0.958 |
| 7 | 4.1507 | 4.2668 | 0.973 |
| 8 | 6.8431 | 5.8403 | 1.172 |

They scatter from 0.69 to 1.58. Two of the six moved by more than 15 percent in
*opposite* directions, which is what sample noise on a contended host looks like,
not a systematic change. The aggregate effect was +13 percent on generation.

**None of that makes the gate wrong.** The projection is a deliberate upper bound
compared against a preregistered ceiling, and it did exactly what it is for. The
question it raises is whether the ceiling was right, and that is not a question
this experiment may answer for itself: contract section 6 says the run stops and
reports rather than reducing the panel, and section 10's caps are preregistered.
**A failed gate does not authorize its own relaxation.**

## What the projection implies about real wall time

`15.87` predicted wall hours is an upper bound built from per-`m` maxima over the
benchmarked grid points. exp105's equivalent projection was `7.01` hours against
`2.20` measured — conservative by a factor of 3.2. On that precedent the real
scan is nearer 5 hours than 16, and the 20-hour wall cap already covers the worst
case. The binding constraint is the core-hour ceiling alone.

## Evidence in this directory

- `environment_qualification.json` — `PASS`, five groups, exact counts
- `remote_resource_preflight_BLOCKED.json` — the failed projection, retained
- `local_suite_after_freeze.txt` — 223 passed / 0 skipped locally on the frozen
  plan, the number the qualification gate was then checked against

## What has to happen next

The decision is the user's, and there is no version of it this directory may
make. Either the section 10 core-hour cap is amended — a resource-only change
that touches no scientific protocol, the same form exp103's approved remote
amendment took — or the generation budget is lowered and the section 6 allocation
rule re-evaluated at the cost of precision, or exp106 stops here.
