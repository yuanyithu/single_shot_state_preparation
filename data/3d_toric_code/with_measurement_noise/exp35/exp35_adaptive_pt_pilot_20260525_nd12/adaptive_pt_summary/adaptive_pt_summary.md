# exp35 adaptive PT 参数搜索汇总

- 输入 NPZ 数量: 54
- allow_partial: False
- 覆盖尺寸 L: [3, 4, 5]
- 覆盖 q: 0.0800, 0.1500, 0.2300

## 轮数汇总

| adaptive轮数 | NPZ数 | disorder数 | mean | max | local墙时比 | local接受率 | min swap | mean swap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18 | 432 | 0.1845 | 0.8750 | 0.7383 | 0.0060 | 0.1485 | 0.5314 |
| 3 | 18 | 432 | 0.1866 | 0.8750 | 0.7366 | 0.0060 | 0.1588 | 0.5287 |
| 5 | 18 | 432 | 0.1841 | 0.8750 | 0.7351 | 0.0060 | 0.1624 | 0.5252 |

## 参数指导

1/3/5 轮差别较小；生产可用 3 轮作为稳健默认。

## 图像

- 总体 f 柱状图: `adaptive_pt_f_final_overall.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p0800.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p1500.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p2300.png`
- 典型点 f 柱状图: `adaptive_pt_f_L4_q0p0800.png`
- 典型点 f 柱状图: `adaptive_pt_f_L4_q0p1500.png`
- 典型点 f 柱状图: `adaptive_pt_f_L4_q0p2300.png`
- 典型点 f 柱状图: `adaptive_pt_f_L5_q0p0800.png`
- 典型点 f 柱状图: `adaptive_pt_f_L5_q0p1500.png`
- 典型点 f 柱状图: `adaptive_pt_f_L5_q0p2300.png`

## 说明

- mean/max 是最终一轮 monotone flow `f_mono` 到线性目标 `f_target` 的绝对误差。
- local墙时比按 ordinary local update / (ordinary + PT swap + observable) 估算。
- local接受率使用冷端链的平均总接受率，包含 single-bit 与 zero-syndrome local 更新。
