# exp41/003 — P1：p=0.11 加 L=7,q_c 误差减半（L=3..7 × 384 disorder）

日期：2026-06-22。延续 exp40,目标压低 q_c(p=0.11) 误差限。**复用 exp40/004 的 L=3,4,5,6×384,仅新跑 L=7**,合并得 L=3..7 完整集。

## 运行链

| 步骤 | run id | 规格 | 墙钟 | 结论 |
|---|---|---|---|---|
| G1 smoke | `exp41_g1smoke_20260620_155118` | nd-1, L3-7, q=0.040, 64dis | ~3.2h | 管线 OK,workers=76/numba/linear,L7 跑通 |
| G2 kp gate | `exp41_g2kp129/257_20260620_191408` | nd-1/nd-2, L7, q=0.034/0.040/0.048, 64dis 配对 | 4.3h/8.3h | **通过**:crossing 区 kp129≈kp257(|Δ|<0.004) |
| P1 生产 | `exp41_p011L7_20260621_034148` | nd-1(192,seed910000)+nd-2(192,seed911000), L7, exp40 十点 q | ~42–43h | ALL CELLS OK,seed 无重,核心区 pass=1.0 |

P1 固定参数（与 exp40/004 逐项一致以保证合并）：projection=linear, kp=129, burn=512, max_eff_burn=512, meas=8192, stride=2, block=128, boot=800, winding=1, seed_scope=disorder_index, realization=rng_stream, use_numba。核时 P1 ~6450（两节点）。

## G2 收敛 gate（投产前验证 kp=129 在 L=7 不偏）

L=7 配对 kp129 vs kp257（同 seed 901000，逐 disorder）：

| q | w0 Δ(129−257) | q_W Δ(129−257) | |Δ|/配对SEM |
|---|---|---|---|
| 0.034（近 w0 crossing） | +0.0023 | +0.0000 | ≤1.3 |
| 0.040（近 q_W crossing） | −0.0036 | +0.0019 | ≤0.15 |
| 0.048（无序尾） | +0.0386 | +0.0274 | ~1.0–1.6 |

**crossing 区 kp129≈kp257（<0.004，配对 SEM 内）**；唯一差异在无序尾 q=0.048（<1.6σ，且不决定 crossing）。kp=129 对 L=7 足够，且与复用的 L3-6（kp=129）同参——用 kp=257 only-L7 反而会引入估计量不一致。⇒ **P1 用 kp=129**。

## 主结果：q_c(p=0.11) ≈ 0.033 ± 0.002（sign-aware w0），误差较 exp40 减半

估计量全部从 `delta_f_per_disorder` softmax 重算（`q_W ≡ exp40 旧 q_top`；stored 标量字段未用）。NBOOT=10000。

**sign-aware w0（headline）**：

| L 对 | q_c | CI95 | 半宽 |
|---|---|---|---|
| **L3-L7（headline）** | **0.0338** | [0.0314, 0.0358] | **0.0022** |
| L4-L7 | 0.0337 | [0.0307, 0.0371] | 0.0032 |
| L5-L7 | 0.0325 | [0.0270, 0.0358] | 0.0044 |
| （L3-L6, exp40 复核） | 0.0331 | [0.0311, 0.0409] | 0.0049 |

- **加 L=7 把 headline 半宽从 exp40 的 0.0049(L3-L6) 收到 0.0022,误差≈减半。** 达成本阶段目标（plan §6：≤0.005，理想 ≤0.004）。
- 与 L7 配对的三个 crossing（L3-L7/L4-L7/L5-L7）一致落在 **0.033**,随小 L 伙伴增大轻微下漂（0.0338→0.0337→0.0325）→ 渐近 q_c ≲ 0.032。
- msigned（第二 sign-aware 量）给出一致结果：L3-L7=0.0330 [0.0311,0.0355]。

**even-moment 交叉验证**：q_W L3-L7=0.0400 [0.0327,0.0420]，q_purity L3-L7=0.0407。比 w0 系统性高 ~0.007,符合已知 even-moment 偏差排序（memory `qtop-saturates-use-deltaf-for-crossing`）。方向一致,无矛盾。

## 曲线形态（图 `p011_L34567_curves.png`）

- 有序侧 q≤0.030：所有 L 的 w0/q_W 饱和到 ~1.0。
- 无序侧 q≥0.040：干净扇出 L3>L4>L5>L6>L7（w0 q=0.048：0.976→0.824，L7 掉最快）。
- L3/L7 在 q≈0.034 分离 = headline crossing。

## Caveats

- **两个 L 对简并,不可用**:**L3-L4**(小 L 双饱和)与 **L6-L7**(两条大 L 曲线在有序侧都饱和到 1.000,微小噪声制造假首变号 → q_c=0.022 贴边)。headline 用小-vs-大 杠杆(L3/L4/L5 对 L7),不用这两对。
- pass_fraction:crossing 核心区(q≤0.034)=1.0;无序尾 q=0.058/0.070 降到 0.5–0.66(L7 比 L≤6 更低,因大系统在深无序侧 kp 网格更吃紧)。**不影响 crossing**(crossing 在 0.033–0.040,该区 pass=1.0);深无序尾点仅作锚,不参与定值。

## 验收（plan §6）—— 全部达成

1. G1/G2 通过,L=7×384 合并成 L3..7 ✓
2. crossing 核心 pass_fraction≥0.9（=1.0）✓
3. headline w0 最大 L 对 CI 半宽 ≤0.005 → **0.0022** ✓✓
4. w0 与 q_W 无矛盾（w0~0.033 < q_W~0.040，符合偏差排序）✓
5. L 对 crossing 显示有限尺寸下漂 ✓
6. 结论：**误差主导是 finite-size geometry**（加一个 L 即减半,加 disorder 不会）;384 disorder 足够,**不需 1024,不需 q 细化**;L=7 应作最终 headline。

## 对第二阶段的指导

- p=0.11 单点定型:**q_c = 0.033 ± 0.002（w0）**。
- 第二阶段不必每点堆 disorder;**算力应投到「每个 p 点加到 L=7」**,尤其近 p_c≈0.227 区(plan §7)。
- 平台中段可复用 exp40/005(L3-5)+本结构,只补大 L。

## 产物
- `p011_L34567_summary.json`（全 L 对 × 4 估计量 crossing + CI）
- `p011_L34567_curves.png`（四联图）
- `analyze_exp41_p011.py`（合并+重算,通用)、`plot_p011_L34567.py`
- 原始 NPZ:`nd{1,2}/collected/p0p11/`(本地,不入 git);L3-6 复用 `exp40/004`
