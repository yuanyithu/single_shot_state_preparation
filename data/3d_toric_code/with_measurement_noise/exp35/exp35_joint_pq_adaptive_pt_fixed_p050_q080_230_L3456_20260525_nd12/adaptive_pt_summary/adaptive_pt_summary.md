# exp35 adaptive PT 参数搜索汇总

- 输入 NPZ 数量: 192
- allow_partial: True
- 覆盖尺寸 L: [3, 4, 5, 6]
- 覆盖 q: 0.0800, 0.0900, 0.1000, 0.1100, 0.1200, 0.1300, 0.1400, 0.1500, 0.1600, 0.1700, 0.1800, 0.1900, 0.2000, 0.2100, 0.2200, 0.2300

## 轮数汇总

| adaptive轮数 | NPZ数 | disorder数 | mean | max | local墙时比 | local接受率 | min swap | mean swap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 192 | 262144 | 0.2053 | 0.8750 | 0.7510 | 0.0064 | 0.1249 | 0.4739 |

## 参数指导

当前 pilot 缺少 1 轮基线；先补齐 1/3/5 轮对比再定生产轮数。

## 图像

- 总体 f 柱状图: `adaptive_pt_f_final_overall.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p0800.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p0900.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p1000.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p1100.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p1200.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p1300.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p1400.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p1500.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p1600.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p1700.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p1800.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p1900.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p2000.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p2100.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p2200.png`
- 典型点 f 柱状图: `adaptive_pt_f_L3_q0p2300.png`
- 典型点 f 柱状图: `adaptive_pt_f_L4_q0p0800.png`
- 典型点 f 柱状图: `adaptive_pt_f_L4_q0p0900.png`
- 典型点 f 柱状图: `adaptive_pt_f_L4_q0p1000.png`
- 典型点 f 柱状图: `adaptive_pt_f_L4_q0p1100.png`
- 典型点 f 柱状图: `adaptive_pt_f_L4_q0p1200.png`
- 典型点 f 柱状图: `adaptive_pt_f_L4_q0p1300.png`
- 典型点 f 柱状图: `adaptive_pt_f_L4_q0p1400.png`
- 典型点 f 柱状图: `adaptive_pt_f_L4_q0p1500.png`

## 说明

- mean/max 是最终一轮 monotone flow `f_mono` 到线性目标 `f_target` 的绝对误差。
- local墙时比按 ordinary local update / (ordinary + PT swap + observable) 估算。
- local接受率使用冷端链的平均总接受率，包含 single-bit 与 zero-syndrome local 更新。
