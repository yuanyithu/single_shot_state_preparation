import hashlib

import numpy as np

from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
from data.expander_code.exp102.exp102_pipeline.q0_pt import Q0PtConfig, run_q0_pt_instance
from data.expander_code.exp102.exp102_pipeline.worker import build_model

load_exp101()
from exp101_certified_src.graphs import cycle_parity_check_matrix

model, frame = build_model(cycle_parity_check_matrix(2))
error = np.zeros(model.num_qubits, dtype=np.uint8)
error[0] = 1
syndrome = (model.H_check.astype(np.int64) @ error.astype(np.int64) % 2).astype(np.uint8)
config = Q0PtConfig(0.45, 4, 1.0, 5, 20)
results = [run_q0_pt_instance(model, frame, syndrome, 0.1, config, 12345, 0, engine=engine)
           for engine in ("reference", "numba")]
fields = ("labels", "swap_attempts", "swap_accepts", "logical_attempts", "logical_accepts",
          "hot_arrival_labels", "hot_departure_labels")
digest = hashlib.sha256()
for field in fields:
    assert np.array_equal(results[0][field], results[1][field]), field
    value = np.ascontiguousarray(results[1][field])
    digest.update(field.encode("ascii")); digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(value.tobytes())
for field in ("round_trips", "sector_changing_round_trips", "max_hard_coset_residual"):
    assert results[0][field] == results[1][field], field
    digest.update(field.encode("ascii")); digest.update(np.int64(results[1][field]).tobytes())
print(digest.hexdigest())
