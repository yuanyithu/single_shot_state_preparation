# STATUS.md — exp38 分步进度（执行 agent 接力用）

> 当前阶段 = 下表中第一个状态不是 `PASS` 的阶段，只在该阶段上工作。
> 状态取值：`TODO` / `DOING` / `PASS` / `FAIL`。判据细节见 `detail_plan.md`。
> 全部 PASS 且通过 Definition of Done 后，把下一行改写为：`ALL DONE`。

ALL DONE

| Stage | 名称 | 状态 | 闸门达成数字（贴关键对比值） | 交付目录 | 更新日期 |
|-------|------|------|------------------------------|----------|----------|
| P0 | 回归锚点（本地，防代码漂移） | PASS | ids=0,1,2: max TV=0.004020, max \|δq_top\|=0.006496, CI miss=0, max grid TV=0.004113 | `001_p0_regression_anchor_20260604/` | 2026-06-04 |
| P1 | 本地配对差分 de-risk + 算力标定 | PASS | P1a PASS production-candidate grid max TV=0.018144, max dq=0.015909；P1b PASS under recorded production-power criterion：strict pilot gate 仍为 audit failure，但 same-seed rng_stream q=0.21,N=12 paired SEM=0.069376 vs unpaired=0.081221, ratio=0.854<1, CI=[-0.294715,-0.030315], projected paired SEM @N=32 =0.042484<=0.05；q=0.20 projected @N=32=0.043838；coordinate_hash rejected corr=-0.381, ratio=1.175, L5 d9 grid dq=0.021202 WARN；P1c PASS wall L3/4/5=51.5/107.3/211.9s, budget with crossing N=32: 4.81/10.02/19.78h per L-node batch | `002_p1_paired_demo_20260604/` | 2026-06-05 |
| P2 | 服务器生产网格（强档 TI，跨 L 公共 disorder） | PASS | run `exp38_p2_ti_grid_20260605_0145`：远端 preflight PASS（nd-1/2/3 conda11+Numba）；L3/L4/L5 shards 全 success；merged grid shape=(3,13,32)，same-seed across L=True；P2a coverage=True、missing=0、rows=1248、points=39；P2b unresolved_tail_fail=0、pass_violations=0；P2c PASS-disorder grid failures=0；common disorder mismatches=0；point statuses PASS:20/WARN:19/FAIL:0，disorder statuses PASS:1195/WARN:53/FAIL:0；max total SEM=0.043558 | `003_p2_production_grid_20260605/` | 2026-06-05 |
| P3 | 第二法抽样交叉验证（退火 + 双向 BAR） | PASS | run `exp38_p3_second_method_20260605_0610`：subset `3:0.22:0,4:0.22:0,5:0.22:0`；bridge 读取 P2 `disorder_seed_per_disorder`，seed 均为 639000；P3a checks=3, max TV=0.003319<=0.03, max \|dq_top\|=0.005144<=0.02；P3b max full-path bidirectional gap=0.046359<=0.20, max BAR residual=8.185e-12<=1e-8；coverage L=[3,4,5] PASS | `004_p3_second_method_subset_20260605/` | 2026-06-05 |
| P4 | 验收 + 失败地图 + 配对差分表 | PASS | P4a PASS：P2 gates P2a/P2b/P2c/common_disorder 全 True，point statuses PASS:20/WARN:19/FAIL:0，disorder statuses PASS:1195/WARN:53/FAIL:0，NPZ seed mismatches=0；P4b PASS：P3 checks=3, max TV=0.003319, max \|dq_top\|=0.005144, max full-path gap=0.046359；P4c PASS：paired rows=39/39，min PASS-only paired N=21，crossing-region CI excludes zero rows=5 at q=0.20/0.21/0.22/0.23；L4-L5 has 0 excluding-zero CIs and remains unresolved for P5 | `005_p4_acceptance_20260605/` | 2026-06-05 |
| P5 | 生产曲线 + crossing 判定 + 绘图 | PASS | G1 PASS：P0 exact replay max TV=0.004020、max \|dq_top\|=0.006496、CI misses=0；G2 PASS：paired CI excludes zero rows=5（L3-L5 at q=0.20/0.21/0.22/0.23，L3-L4 at q=0.22），common three-size crossing resolved=False because L4-L5 has 0 excluding-zero crossing-region CIs；G3 PASS：q_top reconstructed from w_g[8] max abs diff=0，uncertainty includes disorder bootstrap + TI stderr；red line PASS：unresolved tail FAIL present=False；point statuses PASS:20/WARN:19/FAIL:0 | `006_p5_production_curve_20260605/` | 2026-06-05 |
