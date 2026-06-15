# exp36/018 partial summary: early sector gate

更新时间：2026-06-01

更新说明：018 已完成全 q-grid final NPZ 和 10000-bootstrap final gate。
当前权威汇总请看 `018_final_qgrid_summary.md`；本文件仅保留早期滚动诊断脉络。

## 范围

本摘要只分析已经从远端同步到本地的 chunks，不是 018 的完整生产结论。

- 远端 run root: `/home/DATA1/users/yuany/.single_shot/exp36/018_production64_full_q_grid_20260531`
- 本地同步目录: `remote_partial/`
- 已完整生成 final NPZ 的 q：`0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.20,0.21,0.22,0.23`，均覆盖 `L=3/4/5/6` 各 `64` disorder
- 当前 partial 还包括：`q=0.18,L=6/5/4` 已同步 `64/64` disorder，
  `q=0.18,L=3` 已同步 `2/64` disorder；`q=0.19` 尚待 018 继续推进
- 诊断脚本: `src/summarize_exp36_sector_gate.py`
- 详细 gate 报告: `018_partial_sector_gate.md`
- `q=0.08` final gate 报告: `018_q0p08_final_sector_gate.md`
- `q=0.11` final gate 报告: `018_q0p11_final_sector_gate.md`
- 新增高精度复核：`018_q0p13_final_sector_gate_boot10000.md`、`018_q0p15_final_sector_gate_boot10000.md`、`018_q0p20_final_sector_gate_boot10000.md`、`018_q0p21_final_sector_gate_boot10000.md`、`018_q0p22_final_sector_gate_boot10000.md`、`018_q0p23_final_sector_gate_boot10000.md`
- `q=0.16` final gate 报告: `018_q0p16_final_sector_gate_boot10000.md`
- `q=0.16,L=6` checkpoint: `018_q0p16_L6_sector_gate_boot10000.md`
- `q=0.16,L=5` checkpoint: `018_q0p16_L5_sector_gate_boot10000.md`
- `q=0.16,L=4` checkpoint: `018_q0p16_L4_sector_gate_boot10000.md`
- `q=0.16,L=3` checkpoint: `018_q0p16_L3_sector_gate_boot10000.md`
- `q=0.17` final gate 报告: `018_q0p17_final_sector_gate_boot10000.md`
- `q=0.17,L=6` checkpoint: `018_q0p17_L6_sector_gate_boot10000.md`
- `q=0.17,L=5` checkpoint: `018_q0p17_L5_sector_gate_boot10000.md`
- `q=0.17,L=4` checkpoint: `018_q0p17_L4_sector_gate_boot10000.md`
- `q=0.17,L=3` checkpoint: `018_q0p17_L3_sector_gate_boot10000.md`
- `q=0.18` final gate 报告: `018_q0p18_final_sector_gate_boot10000.md`
- `q=0.18,L=6` checkpoint: `018_q0p18_L6_sector_gate_boot10000.md`
- `q=0.18,L=5` checkpoint: `018_q0p18_L5_sector_gate_boot10000.md`
- `q=0.18,L=4` checkpoint: `018_q0p18_L4_sector_gate_boot10000.md`
- `q=0.18,L=3` checkpoint: `018_q0p18_L3_sector_gate_boot10000.md`
- targeted 8-start reference: `../019_targeted_8start_sector_reference_20260531/019_summary.md`

## 物理诊断定义

固定一个 disorder 后，每条链在冷端测量得到一串 Wilson-loop sector 向量
`sector_t=(W_1(t),...,W_7(t))`。正确性 gate 比较不同初态链的
`sector_t` 直方图，而不是只比较压缩后的 `q_top`。

具体统计量是：同一 disorder 内，4 个不同 `sector` 初态链的冷端
sector 直方图两两 total-variation distance 的最大值。当前生产覆盖的
start labels 是 `000,100,010,110`，即 3 个 winding 生成元的 `4/8` 个组合；
这能检查初态依赖，但不能当作“全 8 个拓扑初态都已覆盖”的证明。为了避免把有限
measurement 噪声误判成不收敛，使用 pooled sector 直方图做 parametric
bootstrap，得到同样 sample count 下的 TV p99 参照；若观测 TV 超过 p99，
该 disorder 被标记为不同初态不一致。

## 当前结果摘要

完整 final NPZ 中，`q=0.08..0.17,0.20,0.21,0.22,0.23` 已完成本地 gate。绝大多数
`(q,L)` 的 `TV fail=0`。用 10000-bootstrap 对 018 内的 marginal 点复核后：

- `q=0.13,L=3,d52` 是 018 final gate 内保留的一个 marginal fail：
  observed TV `0.0400`，boot p99 `0.0342`。
- `q=0.23,L=3,d15` 是另一个 high-q L3 marginal fail：
  observed TV `0.0332`，boot p99 `0.0312`。
- `q=0.15` 无 final fail。
- `q=0.22` 的 1000-bootstrap marginal fail 在 10000-bootstrap 下消失。
- `q=0.20` 因 rolling 200-bootstrap 新增一个很小边界 flag，已补做
  10000-bootstrap final gate；所有 `L=3/4/5/6` 的 64 disorder 均通过，
  TV fail `0`。
- `q=0.16` 已生成 final NPZ，并完成 10000-bootstrap final sector gate。
  `L=3/4/5/6` 各 64 disorder 均通过，TV fail `0`：
  `L=3` 为 `q_top=0.949977±0.016291`、start-TV max `0.0605`、
  boot p99 max `0.0811`；`L=4` 为 `0.995024±0.001783`、
  start-TV max `0.0166`、boot p99 max `0.0283`；`L=5` 为
  `0.998015±0.000304`、start-TV max `0.0068`、boot p99 max `0.0107`；
  `L=6` 为 `0.996571±0.000390`、start-TV max `0.0088`、
  boot p99 max `0.0107`。此前 rolling 200-bootstrap partial gate 中
  `q=0.16,L=6,d19` 的标红已被 final 10000-bootstrap gate 清除。
- `q=0.17` 已生成 final NPZ，并完成 10000-bootstrap final sector gate。
  `L=3/4/5/6` 各 64 disorder 均通过，TV fail `0`：
  `L=3` 为 `q_top=0.948914±0.014898`、start-TV max `0.0498`、
  boot p99 max `0.0791`；`L=4` 为 `0.991572±0.003326`、
  start-TV max `0.0127`、boot p99 max `0.0391`；`L=5` 为
  `0.997885±0.000335`、start-TV max `0.0068`、boot p99 max `0.0117`；
  `L=6` 为 `0.995955±0.000525`、start-TV max `0.0078`、
  boot p99 max `0.0137`。rolling 200-bootstrap partial gate 中
  `q=0.17,L=3,d35` 和 `q=0.17,L=6,d35` 的边界标红已被 final
  10000-bootstrap gate 清除。
- `q=0.18,L=6` 已完整达到 `64/64` disorder，并完成 10000-bootstrap
  checkpoint。TV fail `0`：`q_top=0.995018±0.000623`，start-TV max
  `0.0127`，boot p99 max `0.0156`，`q_top spread max=0.019767`。
  物理判断：当前没有不同初态长期驻留在不同 Wilson-loop sector 的信号。
- `q=0.18,L=5` 已完整达到 `64/64` disorder，并完成 10000-bootstrap
  checkpoint。TV fail `0`：`q_top=0.997589±0.000361`，start-TV max
  `0.0068`，boot p99 max `0.0127`，`q_top spread max=0.015475`。
  物理判断：当前没有不同初态长期驻留在不同 Wilson-loop sector 的信号。
- `q=0.18,L=4` 已完整达到 `64/64` disorder，并完成 10000-bootstrap
  checkpoint。TV fail `0`：`q_top=0.993041±0.002744`，start-TV max
  `0.0166`，boot p99 max `0.0381`，`q_top spread max=0.028205`。
  物理判断：当前没有不同初态长期驻留在不同 Wilson-loop sector 的信号。
- `q=0.18,L=3` 已完整达到 `64/64` disorder，并完成 10000-bootstrap
  checkpoint/final gate。该尺寸有 1 个稳定边界 fail：
  `disorder=10`，observed TV `0.0195`，boot p99 `0.0176`，
  `q_top=0.964710`，`q_top spread=0.038982`。全尺寸 `q=0.18`
  final gate 因此也为 TV fail `1`。物理判断：这不是系统性失败，
  但说明该固定 disorder 的四个生产初态 histogram 有小幅可分辨差异，
  需要 targeted 8-start 长链 reference 复核。

针对 `q=0.13,L=3,d52`、`q=0.23,L=3,d15` 以及若干 high-q 对照点，
019 做了 `8/8` 初态、`m=8192` targeted reference，全部通过 bootstrap
p99 gate。

## 暂时结论

这些结果总体没有出现“不同初态长时间后驻留在不同 sector，
但 `q_top` 恰好接近”的系统性信号。`q=0.13,L=3,d52` 的 018 marginal
fail 在 019 的 8-start 长链 reference 中消失：start-TV 从 `0.0400`
降到 `0.0035`，`q_top spread` 从 `0.079274` 降到 `0.006438`，top
sector 概率仍与 018 相容。`q=0.23,L=3,d15` 也在 019 reference 中通过：
start-TV 从 `0.0332` 降到 `0.0056`，`q_top spread` 从 `0.067668`
降到 `0.009727`。

这只能说明已完成的 chunks 通过了当前 sector-histogram gate。
它不能替代完整的 `L=3,4,5,6`、`q=0.08..0.23` 全部 disorder 的最终检查。
018 完整结束后还需要对 final NPZ 重新运行同一 gate，并检查每个 `(L,q)`
的 disorder 平均、方差和 block 漂移。
