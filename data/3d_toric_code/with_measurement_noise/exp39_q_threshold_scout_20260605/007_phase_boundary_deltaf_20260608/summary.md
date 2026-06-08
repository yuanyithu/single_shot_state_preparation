# exp39/007 — 3D toric code single-shot 测量错误相边界 q_c(p)

**问题**：带测量噪声 q + Pauli-X 噪声 p 的 3D toric code 单次态制备，相边界 q_c(p)（可纠错↔不可纠错）长什么样。

**方法**：固定 p 扫 q，对 L=3,4,5 算**扇区自由能 gap Δf**（主导逻辑扇区→最近竞争扇区的自由能差 = 逻辑保护/翻错能垒，**不饱和**，优于在有序相饱和到 1 的 q_top）。Δf(q) 的 L=3,4,5 曲线在 q_c 处换序（有序侧 q<q_c 大 L 更高、无序侧反），即有限尺寸 crossing。q_c(p) 拼成相边界。全程 TI / `projection_mode=linear`（正确观测量，见 `qtop_vs_deltaf_math.md` 与记忆 `exp37-decoder-sector-bug`）。

**运行**：6 个新 p∈{0.02,0.04,0.09,0.14,0.16,0.20} 各 48 disorder；复用 p=0.06(96 dis)/p=0.12(96 dis)。grid129 / m8192 / burn512，`--common-disorder-across-q --disorder-seed-scope disorder_index`。node↔p 并行（每节点跑完整 L=3,4,5 的一次 `run`，worker 池自动均衡 L×q×disorder），`launch_boundary_remote.sh`；3 计算节点 80/80/96 核。p=0.16/0.20 因初始 q 网格按错误的(更低)预期 q_c 设计、crossing 落扫描上边缘，已补高 q 点（同 seed 合并到 `collected/p*/supp/`）。

## 结果

| p | q_c (Δf L3–L5) | CI95 | q_top crossing | boot_frac |
|---|---|---|---|---|
| 0.02 | 0.055 | [0.049,0.060] | 0.032 | 1.0 |
| 0.04 | 0.058 | [0.052,0.069] | — | 1.0 |
| 0.06 | 0.056 | [0.050,0.063] | 0.032 | 1.0 |
| 0.09 | 0.053 | [0.049,0.060] | 0.043 | 1.0 |
| 0.12 | 0.052 | [0.048,0.057] | 0.031 | 1.0 |
| 0.14 | 0.053 | [0.048,0.056] | 0.046 | 1.0 |
| 0.16 | 0.062 | [0.054,0.072] | 0.041 | 1.0 |
| 0.20 | 0.050 | [0.048,0.055] | 0.027 | 1.0 |
| 端点 q=0 | — | — | p_c≈0.227 | (锚点) |

**核心结论**：相边界 q_c(p) 在 p∈[0.02,0.20] **近乎平坦 ≈0.05–0.06**，随 p 仅缓降；q_top crossing（饱和量）给更低的 ~0.03–0.045，两个有限尺寸估计量**夹住真阈值**。物理图像：测量错误阈值主要由时间向 syndrome 耦合 K_q 定，在 p≪p_c 时几乎与 p 无关，直到 p 逼近自身阈值 p_c≈0.227 才陡降到 0。

**caveat**：p∈[0.20,0.227] 的陡降段未解析（图中虚线 schematic，需 p≈0.21–0.22 + 很小 q 的近轴点，成本高）；p=0.16 略高(0.062)，CI 宽、与平坦趋势在误差内一致。q_c 是有限尺寸 crossing 估计（L=3,4,5），非热力学极限外推。

## 产物
- `phase_boundary.png` — q_c vs p（Δf 主线 + CI、q_top 下界线、ordered/disordered 阴影、q=0 端点）
- `deltaf_crossings_grid.png` — 每 p 的 Δf-vs-q（L=3,4,5）crossing，8 panel
- `boundary_summary.json` — 每 p 的 q_c/CI/三对 pairwise/grid 信息
- `launch_boundary_remote.sh`、`analyze_boundary.py` — 运行与分析代码
- `qtop_vs_deltaf_math.md` — Δf 与 q_top 的数学定义 + 程序对齐
