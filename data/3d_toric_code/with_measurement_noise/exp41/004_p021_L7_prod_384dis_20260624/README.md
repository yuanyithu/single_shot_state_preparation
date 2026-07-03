# exp41/004 — 第二阶段 step1：p=0.21 加 L=7（近 p_c 收口测试）恢复手册

目的：p=0.21（逼近 p_c≈0.227）。exp40 的 L≤5 在此给 q_c≈0.042、看不到向 (0.227,0) 收口（小 L 系统性高估可纠错性）。本步加 L=7，检验大 L 是否把 q_c 拉下来 = 收口证据。复用 exp40/005 p=0.21 的 L=3,4,5×384。

## 配置（与 exp40/005 + exp41/003 同参以便合并）
- L=7 only，kp=129，burn=512，max_eff=512，meas=8192，stride=2，block=128，boot=800，winding=1，projection=linear，seed_scope=disorder_index，realization=rng_stream，use_numba。
- q 网格 = exp40 十点：0.012,0.018,0.022,0.026,0.030,0.034,0.040,0.048,0.058,0.070。
- 384 disorder = nd-1 192（seed 920000）+ nd-2 192（seed 921000）。nd-3 仍占用 → 两节点（用户决定 a）。
- 预计 ~42h、~6400 核时。
- MASTER_RUN_ID：`exp41_p021L7_20260624_103221`（TS=20260624_103221；screens exp41_20260624_103221_nd{1,2}）。注：本次经 ssh ControlMaster 多路复用启动（Jumper 间歇性断连，launcher 已加 SSH_CTL 选项）。

## 复用
exp40/005 `nd{1,2,3}/collected/p0p21/`（L3,4,5×384，seed 879xxx，同 q 网格）。

## 合并分析
```
cd ..  # exp41/
E5=exp40_qtop_phase_boundary_20260610/005_boundary_highstats_20260612
python analyze_exp41_p011.py --nboot 10000 --out 004_p021_L7_prod_384dis_20260624/p021_L34567_summary.json \
  --globs "$E5/nd1/collected/p0p21" "$E5/nd2/collected/p0p21" "$E5/nd3/collected/p0p21" \
          004_p021_L7_prod_384dis_20260624/nd1 004_p021_L7_prod_384dis_20260624/nd2
```
注意：analyze 脚本递归找 sector_ti_results.npz，exp40/005 的 p0p21 子目录里也有 p0p02/05/17 同级，故 glob 要精确到 `.../collected/p0p21`。

## 判据
若 w0 的大 L 对 crossing（L3-L7/L4-L7/L5-L7）明显低于 exp40/005 的 L3-L5 值（≈0.042）→ 大 L 把 q_c 拉向收口。验证 L=7 在 p=0.21 的 pass_fraction（近 p_c 可能更吃紧；若 crossing 区<0.9 需补 kp 或 burn）。

## seed 台账
phase-1：G1=900000/G2=901000/P1=910000,911000。**phase-2 step1：p=0.21 = 920000(nd1),921000(nd2)。** exp40 复用：879xxx(005,p21)、86xxxx(004,p11)。
