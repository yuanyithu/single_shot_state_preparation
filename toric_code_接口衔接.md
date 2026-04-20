# Toric code 示例与主框架的衔接

`build_toric_code_examples.py` 提供两个构造函数，用来产生架构设计里
`run_disorder_average_simulation` 的前两个输入参数。dtype / shape 已经严格
按照接口约定准备好，不需要任何后处理。

## 构造函数

- `build_2d_toric_code(lattice_size) -> (parity_check_matrix, dual_logical_z_basis)`
- `build_3d_toric_code(lattice_size) -> (parity_check_matrix, dual_logical_z_basis)`

两者返回值的约定（与 `架构设计.md` 第 1–15 行的“输入约定”一致）：

- `parity_check_matrix`: `np.ndarray[bool]`，shape `(num_checks, num_qubits)`，即 H_Z。
- `dual_logical_z_basis`: `np.ndarray[bool]`，shape `(num_logical_qubits, num_qubits)`，
  每行一个逻辑 Z 代表链，**已经**落在 ker H_X 中（构造函数内部保证）。

`num_logical_qubits` 即论文中的 k：2D toric code → k = 2，3D toric code → k = 3。

## 典型调用

```python
from build_toric_code_examples import build_3d_toric_code
from <主框架模块> import run_disorder_average_simulation

parity_check_matrix, dual_logical_z_basis = build_3d_toric_code(lattice_size=3)

result = run_disorder_average_simulation(
    parity_check_matrix,
    dual_logical_z_basis,
    syndrome_error_probability=q,
    data_error_probability=p,
    num_disorder_samples=...,
    num_burn_in_sweeps=...,
    num_sweeps_between_measurements=...,
    num_measurements_per_disorder=...,
    rng=np.random.default_rng(0),
)
```

## 给 Codex 的几条注意事项

1. **不要**把 H_X 传进主框架。主框架只消费 H_Z 和 dual_logical_z_basis。
   文件里的 `build_2d_toric_x_check_matrix` / `build_3d_toric_x_check_matrix`
   以及 `verify_and_report` 只是 sanity check 工具，不要放进主流程。
2. **不要**事先去掉 H_Z 的冗余行。3D toric code 的 H_Z 有 O(L³) 个冗余，
   这是 3D 单次制备 (single-shot preparation) 物理性质的来源，必须保留。
   架构里的 `build_linear_section` 会以 rank ρ < num_checks 的方式自动处理。
3. 两种 code 共用同一条代码路径，**不**需要为 2D / 3D 分别写主流程分支。
   差别完全体现在 H_Z 的稀疏图案上：
   - 2D：每列 weight = 2（每条边在 2 个 plaquette 里），k = 2，Z 冗余 = 1
   - 3D：每列 weight = 4（每条边在 4 个 plaquette 里），k = 3，Z 冗余 = L³ + 2
4. `lattice_size` 建议 ≥ 2。3D 规模增长很快（n = 3L³），跑 MCMC 时先用 L = 2 或 3
   做 smoke test，再上更大的格子。
5. 要换成开放边界的平面 surface code（2D k = 1、3D 带自由边界），只需在构造函数
   内部把 `% lattice_size` 的地方在边界处省掉对应边 / plaquette 即可，
   **接口本身不变**。
