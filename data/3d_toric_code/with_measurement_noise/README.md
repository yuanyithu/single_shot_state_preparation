# 3D Toric Code With Measurement Noise

- [`exp05_q005_local_precheck_after_fix`](exp05_q005_local_precheck_after_fix/)：修复后本地轻量预检，只验证 `q>0` 生产路径和诊断字段。
- [`exp06_zero_disorder_quick_scan`](exp06_zero_disorder_quick_scan/)：全零 disorder 单样本快速扫描，不作正式 threshold 结论。
- [`exp07_q005_broad_scan`](exp07_q005_broad_scan/)：正式 `q=0.005` broad scan，给出 `p≈0.2077` deep-window 线索，但 CI 较宽。
- [`exp08_q005_oneday_deep_scan`](exp08_q005_oneday_deep_scan/)：一天内快速 deep；convergence 失败较多，不宣称 threshold。
- [`exp11_q001_oneday_deep_partial`](exp11_q001_oneday_deep_partial/)：停止过慢 `q=0.0100` one-day deep 后回收的 partial 数据；只有 `L=3,4`，不能作三尺寸 threshold。
- [`exp12_q005_fine_20260425_nd1`](exp12_q005_fine_20260425_nd1/)：Numba 后的 `q=0.0050` fine run；`528/528` chunks 完成，自动推荐窗口 `p≈0.2164~0.2200`。
- [`exp13_q001_coarse_20260425_nd2`](exp13_q001_coarse_20260425_nd2/)：`q=0.0100` coarse right-side check；说明 `p>0.24` 已偏右，不应继续把算力放在更大 `p`。
- [`exp14_q001_fine_20260425_nd3`](exp14_q001_fine_20260425_nd3/)：`q=0.0100` `0.225~0.270` fine scan；多数 gap 已偏正，支持回到左侧加样本。
- [`exp15_q001_left_denseA_20260425_nd1`](exp15_q001_left_denseA_20260425_nd1/) / [`exp16_q001_left_denseB_20260425_nd2`](exp16_q001_left_denseB_20260425_nd2/)：`q=0.0100` 左侧 dense 独立复本，各 `64` disorder，用于池化。
- [`exp17_q001_left_fine_20260425_nd3`](exp17_q001_left_fine_20260425_nd3/)：`q=0.0100` 左侧 `0.0025` fine grid，作为 dense 池化的局部对照。
- [`exp18_q001_left_combined_summary`](exp18_q001_left_combined_summary/)：`q=0.0100` 左侧综合图和 JSON；当前最重要结论是窗口约 `p≈0.22~0.235`，不应继续向 `p>0.24` 扩展。
