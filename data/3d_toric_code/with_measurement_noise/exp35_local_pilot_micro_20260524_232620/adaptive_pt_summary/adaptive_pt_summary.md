# exp35 adaptive PT 参数搜索汇总

- 输入 NPZ 数量: 3
- allow_partial: True
- 覆盖尺寸 L: [3]
- 覆盖 q: 0.1500

## 轮数汇总

| adaptive轮数 | NPZ数 | disorder数 | mean | max | local墙时比 | local接受率 | min swap | mean swap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 0.0000 | 0.0000 | 0.9348 | 0.0061 | 0.0000 | 0.5000 |
| 3 | 1 | 1 | 0.1667 | 0.5000 | 0.9361 | 0.0061 | 0.5000 | 0.7500 |
| 5 | 1 | 1 | 0.1667 | 0.5000 | 0.9613 | 0.0061 | 0.0000 | 0.5000 |

## 参数指导

1/3/5 轮差别较小；生产可用 3 轮作为稳健默认。

## 图像

- 总体 f 柱状图: `adaptive_pt_f_final_overall.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p1500.png`

## 说明

- mean/max 是最终一轮 monotone flow `f_mono` 到线性目标 `f_target` 的绝对误差。
- local墙时比按 ordinary local update / (ordinary + PT swap + observable) 估算。
- local接受率使用冷端链的平均总接受率，包含 single-bit 与 zero-syndrome local 更新。
