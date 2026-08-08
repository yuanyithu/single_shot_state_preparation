"""Power analysis for exp104 using the real bootstrap and classification code.

Synthetic per-code failure rates are drawn from exp103's measured per-distance
rates (mean and within-stratum spread) and assigned using the exp104 registry's
actual distance composition. Nothing here touches the frozen pipeline; it only
answers whether N=2000 codes at T=4 trials can certify the crossing.
"""

import csv
import collections
import sys
import time
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parents[5])
sys.path.insert(0, ROOT)

from data.expander_code.exp104.exp104_pipeline.config import load_config
from data.expander_code.exp104.exp104_pipeline.crossing import (
    classify_crossing,
    cluster_bootstrap,
    crossing_location,
)
from data.expander_code.exp104.exp104_pipeline.ensemble import load_registry

DIAG = f"{ROOT}/data/expander_code/exp103/validation/010_final_crossing_20260807/code_diagnostics.csv"
P_TOKENS = [f"0.{v:02d}" for v in range(2, 11)]

rows = list(csv.DictReader(open(DIAG)))
by_md = collections.defaultdict(list)
by_d = collections.defaultdict(list)
for r in rows:
    key = (int(r["m"]), int(r["classical_distance"]), r["p"])
    by_md[key].append(float(r["rate"]))
    by_d[(int(r["classical_distance"]), r["p"])].append(float(r["rate"]))


def stratum_stats(m, d, token):
    values = by_md.get((m, d, token))
    if values is None or len(values) < 2:
        values = by_d.get((d, token))
    if values is None:
        # d=12 never appears in exp103; extrapolate from the best stratum seen.
        values = by_d.get((10, token)) or by_d.get((8, token))
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.05
    return mean, max(std, 0.01)


config = load_config(f"{ROOT}/data/expander_code/exp104/config/ensemble_mc.v1.json")
registry = load_registry(f"{ROOT}/data/expander_code/exp104/config/ensemble_registry.v1.json")
distance_by_m = collections.defaultdict(list)
for row in registry["codes"]:
    distance_by_m[row["m"]].append(row["classical_distance"])

print("exp104 registry composition (2000 codes per m):")
for m in range(3, 9):
    counter = collections.Counter(distance_by_m[m])
    total = sum(counter.values())
    print(f"  m={m}: " + "  ".join(
        f"d={d}:{c / total:.3f}" for d, c in sorted(counter.items())
    ))

rng = np.random.Generator(np.random.PCG64(20260808))
trials = 4
failures_by_m = []
for m in range(3, 9):
    distances = np.asarray(distance_by_m[m])
    counts = np.zeros((len(distances), len(P_TOKENS)), dtype=np.int64)
    for p_index, token in enumerate(P_TOKENS):
        rates = np.empty(len(distances))
        for d in np.unique(distances):
            mask = distances == d
            mean, std = stratum_stats(m, int(d), token)
            drawn = rng.normal(mean, std, size=int(mask.sum()))
            rates[mask] = np.clip(drawn, 0.0, 1.0)
        counts[:, p_index] = rng.binomial(trials, rates)
    failures_by_m.append(counts)

start = time.perf_counter()
boot = cluster_bootstrap(failures_by_m, trials, config, "power_check")
print(f"\nbootstrap: {config['bootstrap']['replicates']} replicates in "
      f"{time.perf_counter() - start:.1f}s")

p_values = np.asarray([float(t) for t in P_TOKENS])
print("\n  p     " + "".join(f"{m:>9d}" for m in range(3, 9)) + "     Delta38    band")
for i, token in enumerate(P_TOKENS):
    line = f"  {token} " + "".join(f"{boot['point'][j, i]:9.4f}" for j in range(6))
    mark = ""
    if boot["endpoint_high"][i] < 0:
        mark = "  CERT-"
    elif boot["endpoint_low"][i] > 0:
        mark = "  CERT+"
    line += f"  {boot['endpoint'][i]:+9.4f}  [{boot['endpoint_low'][i]:+.4f},{boot['endpoint_high'][i]:+.4f}]{mark}"
    print(line)

print(f"\nsimultaneous half-width : {boot['half_width']:.4f}   (exp103 was 0.2601)")
per_m_sem = [
    float(np.std(boot['point'][j])) for j in range(6)
]
decision = classify_crossing(
    p_values, boot["endpoint"], boot["endpoint_low"], boot["endpoint_high"],
)
print(f"terminal status         : {decision['status']}")
print(f"certified bracket       : {decision['bracket']}")
location = crossing_location(
    p_values, boot["endpoint"], boot["endpoint_replicates"], decision,
)
print(f"p_cross                 : {location['p_cross']:.5f}  "
      f"95% CI [{location['p_cross_low']:.5f}, {location['p_cross_high']:.5f}]  "
      f"defined {location['defined_fraction']:.3f}")
