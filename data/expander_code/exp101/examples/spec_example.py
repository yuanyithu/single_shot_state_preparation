"""expander_code.md「Expected final deliverable」示例脚本。

运行：cd "project D" && conda run -n 12 python data/expander_code/exp101/examples/spec_example.py
输出同时写入本目录 spec_example_output.txt（G1.7 证据）。
"""

import sys
from pathlib import Path

EXP101_ROOT = Path(__file__).resolve().parents[1]
if str(EXP101_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP101_ROOT))

from src.instance import build_quantum_expander_code_instance  # noqa: E402


def main():
    instance = build_quantum_expander_code_instance(
        m=2,
        d_A=3,
        d_B=4,
        seed=12345,
        gamma="1/10",
        delta="1/16",
        verify_expansion=True,
        compute_logicals=True,
        compute_distance=True,
    )
    lines = [
        f"seed = {instance.seed}",
        f"n_A, n_B = {instance.graph.n_A}, {instance.graph.n_B}",
        f"n = {instance.parameters.n}",
        f"k = {instance.parameters.k}",
        f"d_X, d_Z, d = {instance.parameters.d_X}, {instance.parameters.d_Z}, "
        f"{instance.parameters.d}  (method: {instance.distance_method})",
        f"CSS commutation holds: {instance.css_commutation_ok}",
        f"expansion verification passed: {instance.expansion_result.passed} "
        f"(vacuous_left={instance.expansion_result.vacuous_left}, "
        f"vacuous_right={instance.expansion_result.vacuous_right})",
        f"construction_attempts = {instance.graph.construction_attempts}",
        f"classical rank = {instance.classical_rank} (full={instance.graph.n_B})",
        f"fingerprint = {instance.fingerprint()}",
        f"notes = {instance.notes}",
    ]
    text = "\n".join(lines)
    print(text)
    output_path = Path(__file__).resolve().parent / "spec_example_output.txt"
    output_path.write_text(text + "\n", encoding="utf-8")
    instance.save_json(Path(__file__).resolve().parent / "spec_example_instance.json")
    return instance


if __name__ == "__main__":
    main()
