非主报告，仅保留备忘。正式引用请优先看 [实验报告_2D_toric_code.md](实验报告_2D_toric_code.md) 与 [README.md](README.md)。

目前不能“确保”很小 p 区域绝对没有 threshold，只能说现有数据不支持存在一个有限的正阈值。

现有依据是三层：

q=0.001 扩到 L=11 后，在整个 p ∈ [0.075, 0.100] 里，L3-L5、L5-L7、L7-L9、L9-L11 的 pair gap 全都没过零，只能给出
p_cross < 0.075。
这说明如果 threshold 存在，它已经被推到更小 p 了。

对固定正 p，比如 p=0.075, 0.0875, 0.10，q_top(L) 都随 L 单调下降，没有出现“大尺寸趋于平台”的迹象。
这更像是 thermodynamic limit 下性能继续变差，而不是收敛到某个有限 p_c(q) 两侧的分界。

从 q=0.0025, 0.005, 0.01 到更小的 q=0.001，pseudo-threshold proxy 没有稳定在某个正 p，而是继续往左漂。

但这三点的逻辑含义是：

它们支持“没有有限正 threshold 的强数值迹象”
它们不等于“已经直接排除了 p < 0.075 里某个极小但非零的 threshold”
如果你要专门回答“怎么确保很小 p 区域也没有 threshold”，那最后还需要一轮更直接的检查：

固定 q=0.001
把窗口继续左移到例如 p ∈ [0.01, 0.08]
尺寸至少保留 L=5,7,9,11
看相邻尺寸 crossing 上界是否还继续随 L 左移
看固定 p 的 q_top(L) 在这些更小 p 上是否仍单调下降
一句话说，现在的数据是在“把 threshold 一直往 0 挤”，而不是已经对 p→0^+ 做了数学上的封口。论文里更稳妥的表述应是：

strong numerical evidence that no finite positive threshold is observed for any fixed q>0 in 2D

而不是“严格证明在所有很小 p 上都不存在 threshold”。