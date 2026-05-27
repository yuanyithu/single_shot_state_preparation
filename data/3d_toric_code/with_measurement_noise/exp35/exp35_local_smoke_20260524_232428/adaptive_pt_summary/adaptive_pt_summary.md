# exp35 adaptive PT 参数搜索汇总

- 输入 NPZ 数量: 1
- allow_partial: True
- 覆盖尺寸 L: [3]
- 覆盖 q: 0.0800

## 轮数汇总

| adaptive轮数 | NPZ数 | disorder数 | mean | max | local墙时比 | local接受率 | min swap | mean swap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 0.1667 | 0.5000 | 0.9321 | 0.0049 | 0.0000 | 0.3333 |

## 参数指导

当前 pilot 缺少 3 轮结果；建议至少补 3 轮后再定生产轮数。

## 图像

- 总体 f 柱状图: `adaptive_pt_f_final_overall.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p0800.png`

## 说明

- mean/max 是最终一轮 monotone flow `f_mono` 到线性目标 `f_target` 的绝对误差。
- local墙时比按 ordinary local update / (ordinary + PT swap + observable) 估算。
- local接受率使用冷端链的平均总接受率，包含 single-bit 与 zero-syndrome local 更新。
