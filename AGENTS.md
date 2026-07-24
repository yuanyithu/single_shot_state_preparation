本文是开发本数值模拟程序的操作契约与避坑清单；物理图像与核心概念见下方「物理图像与 L·T·S 分解」一节。

使用python开发，本地使用已有的名为`12`的conda环境 本地运行和验证统一使用名为`12`的conda环境，不要默认切到别的环境

维护本文档的规则（每次完成复杂开发后执行）：

- 优先**就地改写**已有条目，而不是在底部不断追加新条目；同一个坑有了新认识就改原条目，避免各节无限增长、前后矛盾。
- 清理判据是「这条坑是否还会再次绊到 agent」：只要某个坑在后续 run 里仍可能复现、需要主动规避，就保留在本文档——无论它是通用规律还是带具体参数实例的个案（如 `exp35 p=0.05,q_hot=0.44 可行`）。只有已被代码或流程修掉、不可能再触发的历史记录，才删除或下沉到 `笔记/实验报告.md`。
- 冻结或启动复杂实验前必须做一次**科学 red-team**：逐项核对目标分布与支持集、坐标含义和合法且有对抗性的初态、真正的慢变量与“接受但未移动”的自环、估计量是否就是用户要交付的量、门禁的假阳性/假阴性与共同失效模式、exact/独立确认以及结果权限边界。上述问题没有说清前，不得先围绕 wall time、acceptance、ESS 或实现细节做局部优化。

## 当前主线：expander code（exp101 起，2026-07-07~）

项目从 3D toric 转向 quantum expander code（(3,4)-biregular 随机图 HGP）单发制备 q_top。
**exp101 的 `exp101.physics.v2` 已由 `validation/014_paper_alignment_20260713/` 认证；
`exp101.scan.v3` 已由 `validation/015_aggregation_safety_20260714/` 认证，可用于严格门禁后的正式
publication/FSS。014 中的 scan v2 聚合只作历史审计；旧 259 tests 与 V1–V6 仍全部是
`PRE_ALIGNMENT`。exp102 复用当前管线时必须遵守下列生产约束。**接手先读
`data/expander_code/exp101/PHYSICS_CONTRACT.md`（唯一物理权威）、`status.md` 和
`validation/README.md`。关键硬约束：

**exp102 当前为 `Q=0 HGP HARD-PAIR DIAGNOSTIC UNRESOLVED / PRE-PILOT`，不是已有物理结果。** 正式历史契约仍为
`exp102.physics.v1 / exp102.q0_pt.v1 / exp102.scan.v1`。固定 Q32 + multi-swap PT-v2 已因 96 条轨迹
认证往返总数为 0 而 `EXHAUSTED`；不得追加 S128、延长轮数或复用 raw。随后
`exp102.q0_pa.discovery.v1` 的四个 transport autopsy 因条件 attempts<200 均为 `INCONCLUSIVE`，
`C192-2/B96-1/B192-1/B96-2` 又在两个 hard cells 上因 genealogy 塌缩全部失败；PA 零通过分支同样
`EXHAUSTED`，禁止 B384-2 rescue。

`exp102.q0_global.discovery.v1` 已实现 logical catalog、hard-coset cluster/joint heatbath、独立
defect trace、m3 full-sector TI、三节点 digest/runtime、72h schedule 与 control freeze。其第三个且
终止性的 immutable run `exp102_q0_global_20260721_204b37d`（source
`204b37d8e00e7d11ffa2b6766b90d947892e179d`）三节点 worker 与 canonical digest 全过，所有 hard/defect
候选也都可用 T3；但必需的 TI contingency 在 nd-2/nd-3 投影为 116275/251241 秒，超过冻结的
79200 秒窗口，故专用 worst-node consensus 为 `RUNTIME_EXHAUSTED`，在 bias/screen 前终止。**节点
worker 的 SUCCESS 只表示测试与报告生成成功，不等于 preflight PASS；后续 stage 必须看到 aggregate
runtime/preflight status=PASS。** 旧 combiner 对合法 exhausted report 抛异常的问题已修为持久化
`RUNTIME_EXHAUSTED`，但下游 PASS 门槛未变。完整证据见
`validation/010_q0_global_runtime_exhausted_20260721/`；前两次基础设施失败审计见 008/009，不得原地
重跑。当前全范围结论只能是 `UNRESOLVED_WITHIN_ALGORITHM_AND_72H_BUDGET`，不能写 `IMPOSSIBLE`，
也不能外推为某个参数点的物理失败。该 discovery 自身的 screen/HARD2/confirmation/resolution/TI
sampler raw 均不存在。

用户随后批准了独立 `exp102.q0_global.screen_diagnostic.v1` 的 `HARD2+EASY3` 测试，最高权限仅为
`DIAGNOSTIC_SCREEN_PAIR_FOUND`。修复后的 fresh run
`exp102_q0_screen_diagnostic_20260721_342dd5b`（source `342dd5bc0fb2c7694dbc58a8d0f2d92689c24991`）
已通过三节点 preflight/digest/runtime，选择 T3，并完整运行、逐位 replay 15/15 bias 与 1280/1280
measurement raw（`reused=0`）。终态为 **`UNRESOLVED_NO_HARD_COSET_PASS`**、`selected_pair=null`：
`RC8-QC1/QC4/J08/J12/J16` 和 `DT16/DT32/DT64` 均为 0/5。hard-coset 的 25/25 cell summaries
均超过 P/U `q_top` 差的绝对 0.04 门槛，全部 U family 均 `Rhat>1.05` 且 ESS<400；480 条 defect-trace chains 的
fixed-clock D=0 observation 与完整 excursion 均为 0。故这是冻结 T3 预算内的 sampler 收敛失败，
不是基础设施失败、`IMPOSSIBLE` 或正式参数点结论；不得追加链长、改 gate 或直接进入 full-range。

本地 conda-12 verified-archive replay 的 raw/state/label/counter 仍逐位一致，所有门禁与终态一致；仅
62 个派生 `core_seconds` 和 18 个派生 ESS 跨平台相差最多 4 ULP。证据比较必须分别验证 report/package
self-hash，并只对这两个白名单字段允许已审计的 4-ULP 上限，不能把整个 report 做 byte equality，
更不能对 gamma/raw replay 使用 ULP 容差。4096 项 Decimal gamma 的 versioned SHA 仍固定为
`a2c459ec9438e23f863c44528ac093c5b93d891b6a8bec0278b873fe47f2459a`；禁止恢复平台 `libm`
fractional power。首个 `5e1f5aa` run 保持 `CONFLICT_CROSS_NODE_GAMMA_LIBM` 审计，15 个旧 bias raw
永不复用。接手先读 `GLOBAL_SCREEN_DIAGNOSTIC_CONTRACT.md`、`validation/011_*` 和 `status.md`。

2026-07-22 隔离 HGP v2 诊断已经完成，fresh immutable run 为
`exp102_q0_hgp_screen_v2_20260722_4d134ee`（source
`4d134ee7ca25125d341eb11cbfa34d6856514101`、archive
`ad72d2c7039192be721b87ce7c96c5da577af05acd37cacd9167e26a773d9027`、manifest
`5bafae76b06ff46557ae8315bb281a42256e7e4e50ed2e9dae868695114b8ff8`）。三 Linux 节点 full exact
consensus PASS 并选中 T3，本地 conda-12 对 portable projection 与四条 MAM acceptance-decision probes
逐位复核为 `PORTABLE_PASS`；12 个 full mismatch 全部且仅是预注册的 nonportable float。固定 ownership
下 384/384 measurement 与 2/2 IS 完成，nd-3 full replay 和本地 terminal audit 验证 386/386 raw；终态为
**`UNRESOLVED_MAP_MIXTURE_FAIL`**，不是 infrastructure `CONFLICT` 或 `IMPOSSIBLE`。terminal package
identity 为 `233e31e599180153f979a30dc971e8e8128be64505fd0572d68bc1ae87a64041`。

方法结果：HP64 为 5/5，是明确的 promising candidate；HP32 为 3/5，其中 m3 只是单个 B-character
`0.0404396>.04` 的边缘 fail，m5 则是明确 B 慢模态（U max Rhat `1.1552`、min ESS `327.0`、pooled
Rhat `1.1172`）。MAM-IMH8 仅 1/2，m8 的普通 P family 与 P/U B family 均未过 Rhat/ESS，16 个 B
characters 初始化族不一致。HP64/MAM 的四个 HARD2 family-cell comparison 全为 0/4：m6 的
`q_top=.14587/.16241` 绝对差虽小于 .04 但约 30 SE，m8 为 `.91317/.99273`，绝对差约 .0795。
因此一个 sentinel `(m,p)` 也未认证，HP32/HP64 属同一机制，不能互相当独立确认。

本 run 暴露的新关键坑是：**真实 state change 不等于 logical transport**。m8 的每条 MAM 链虽至少有
1947 次 measurement state changes、rate 至少 `.0594`，P/U 合计分别有 39899/40735 次，但其中只有
330/288 次改变 logical label（约 `.827%/.707%`）；典型链只见 3 个 label。两个不同的最小权重 m8
MAP anchors 的 64-bit logical coordinate 完全相同，`theta_logical=.08/.25/.5` 的 proposal components
在每族 524288 次总 attempts 中零接受，global IS ESS/总 acceptance 主要测到同扇区运动。后续 MAM
viability 必须预注册 anchor signature coverage、逐 component 的 accepted cross-signature moves 和
logical-character mixing；不得直接加长 T3 或只优化总 acceptance/state changes。

五个 sentinel syndrome 权重为 `83,160,39,58,125`，物理全零 bit string 全不在目标 hard coset；只有
显式定义 `x=e xor epsilon_true` 时，`x=0` 才对应现有 P 初态，并不是新的起点。16 条 P 链共享该合法
planted state 但 RNG 独立，16 条 U 链使用独立 exact-K0 hard-coset states。不得用“所有链从 0 开始”
制造表面收敛，也不得删掉 P/U；若扩充初态，应增加合法、结果无关、按 logical signature 分层的对抗
初态。HP 每轮精确重抽 `A|B` 仍可能用条件噪声掩盖 B，故 B-bit/row-column/dense-character 与 full
logical/energy/weight 门都必须保留。有限 characters 和 16 个 U 仍可能共同漏 basin，HP64 通过也不是
混合证明。接手先读
`HGP_GLOBAL_SCREEN_CONTRACT.md`、`validation/013_q0_hgp_global_screen_20260722/` 与 `status.md`。

v1 `exp102_q0_hgp_screen_20260722_2e6ba2a` 的 Linux preflight 虽 PASS，但本地发现 MAM float/IS full
digest 漂移而终止 `CONFLICT`，从未产生 measurement raw；仍不得续跑或复用。v2 最高权限本就只有
diagnostic，当前也没有 `READY_FOR_FORMAL`；EASY3 独立确认、fresh T/2T、扩展 panels、正式 tuning、
held-out 与 production 全部仍缺失。

2026-07-23 的 local-only `CSMC64-B8-S1-N128` 已终止为
`LOCAL_COLLAPSED_SMC_WEIGHT_OR_GENEALOGY_NOT_VIABLE`：它在 m8 hard cell 上从数学正确的
`lambda=0` iid Bernoulli B base 出发，八个 N128 populations 均通过 exact small-HGP / reference-Numba、
完整 seed replay 和独立 raw-only audit，但 63 次**无条件** systematic resampling 使 root ESS 在 stage 31
已降至 median `1.22/128`、终态仅 1--5 roots (ESS 1.00--2.74)。其 fresh 的无 resampling 后继
`CAIS64-B8-S1-N128` 也已终止为 `LOCAL_COLLAPSED_AIS_PATH_WEIGHT_NOT_VIABLE`：八个 exact-base
paths 虽无 clone、且 replay/不调用 AIS engine 的 raw-only audit 均通过，终态 full-path ESS/N 仍只有
`.0078125--.0100431`（门槛 `.25`）、最大权重 `.872760--1`（门槛 `.10`），cold endpoint median ESS
约 `1.000002/128`。关键新坑是：**逐级 CESS 看似约 .9N 不保证独立根，去掉 resampling 也不保证完整
AIS 路径权重不塌缩；必须看 full-path ESS/最大权重及独立重算的 AIS 时序公式。** 两份 raw 均不得补长、
合并、作 q_top 或送 HARD2；不能据此说全部 collapsed SMC/AIS 不可能。后续若评估新 annealing，必须
另立 fresh config/seeds/raw，并同时报告 full-path weights、root ancestry（若有重采样）和为何其 AIS/SMC
weight formula 对所用可逆 mutation kernel 正确；不要用 P/U/L 或物理零态替换这个 exact-base initializer。

2026-07-24 的 fresh local IID-MIS m8 诊断终止为
`LOCAL_IID_IS_EMPIRICAL_FEASIBILITY_UNRESOLVED`：每个 block 对多个 proposal 等量抽样时，目标
mixture density 必须是同样的**均匀** mixture，不能事后改 mixture weights。旧
MAM/T05/mixture 的 min block ESS 仅 `22.09/28.91/28.78`（门槛 `50`），max weight 为
`.1522/.1629/.1574`（门槛 `.10`）。其 component-provenanced 后继
`LOCAL_BP_SYSTEMATIC_IID_FEASIBILITY_UNRESOLVED` 又表明一个更细的契约坑：BP-SYS-F64/R64
单独都通过 ESS/最大权重 gate，但预冻结的三源等权 mixture 因 MAM-source blocks 只有 ESS `23.48`
和最大权重 `.14695` 而失败。**只要 stress proposal 被放进 estimator 的 mixture，mixture gate 就会让它
实际拥有 veto；不得在看见结果后剔除它并把 BP-only 说成通过。** 后继若真要评估 BP-only，必须另立
fresh contract/seeds/raw，明确 stress source 是仅报告还是 estimator 的一部分。

更根本的坑是：**full support、cross-proposal agreement、低 jackknife SE 甚至看似很高的 collision 值
都不能认证未观测 target tail；反过来，低温链未在 measurement 内遍历全部 logical directions 也不是
purity 正确性的数学必要条件。** 不得加样、放宽门或把诊断数报为 q_top；任何 successor 都必须保存
source 之外的 anchor/component provenance，并有独立 tail/normalizer 证据或确认方法。不要以所有链从
P/物理零态开始来规避这个问题；非零 syndrome 的物理零态仍不在支持上。

2026-07-24 的 strict depth-two collapsed-B envelope 又确认一个容易忽略的区别：**更快地算出一个
global upper bound，不等于该 bound 对交付目标足够紧。** 在 m8 hard sentinel 上，width-25 contraction
虽只用 `3.14s`、峰值约 `2.50GB`，但两个非 planted retained B marginals 的 tail/retained 上界仍是
`9.40e84`（目标 `<=.01`）。不要把 runtime PASS、更多 MCMC clocks、换成共同 P/零初态或漂亮的局部
ESS 当作缩小这 87 个数量级 global-mass 缺口的理由；要继续必须另立能说明 tightness 与资源的契约，且
depth-two 的负结果只排除该 factorized envelope，不可泛化成数学不可能。

2026-07-24 的 local HCA 调查也已完成。035 的 single-copy character WMC 在两种编码下 min-degree
width 都为 `378`，min-fill 在 120s 前已经超过 width `102`；036 的最佳 linear-code trellis exponent
为 `584`，均不能作为当前 exact-normalizer 路线。037/038 的 tensor-logical Houdayer 坐标在真实低能对上
只产生整对 replica exchange；039 唯一预注册的 canonical-reduced 坐标虽在 120 个 L/L 对中有 102 个
产生真实新 unordered pair（例如 `67+67 -> 63+71`），但 P/L 仍只是 whole swap。随后的精确
HCA-RHB1 pair kernel（每 replica 832 random-scan coordinate heatbath 加一 HCA；small-HGP 完整
stationarity、replay、raw-only audit 均过，固定 `128+1024` clocks runtime 约 22s）在 fresh PP/UU/LL/PL
八对各族屏幕终止为 `LOCAL_HOUDAYER_PAIR_TRANSPORT_UNRESOLVED`：PP/LL/PL 在 normalized pair weight
约 `.03886` 一致，LL 有 1091 个真新 pair event；但 exact-K0 U/U 为 `.1486354`，相对 PP 差
`.1097799`、basis gap `1.4603271`、early/late fail，且 8 条 U/U 的 1024 个 measurement HCA 每条都只
是 whole-pair exchange、零真新 pair。**可见的低能 L/L recombination 不能替代从均匀 hard coset 到目标
低能区的运输。** 不得删除 U/U、延长/合并 041 raw、改阈值或改成共同 P/物理零初态；非零 syndrome 下物理
零非法、平移坐标零就是 P。该结果只排除这个 HCA kernel/budget，不是 `IMPOSSIBLE`，也不授权 remote、
formal、held-out 或 production。接手需读 `validation/035_*`--`042_*`、尤其 041 raw audit 与
`status.md`；下一候选必须先证明其如何解决 U transport，并保持独立 tail/normalizer 或独立 confirmation
路径，不能只优化 HCA event 数、acceptance、ESS 或低能 L/L 图像。

同日的 `043_q0_collapsed_houdayer_structure_feasibility_20260724` 进一步排除了一个表面上很自然但
其实偏离慢变量的后继：在精确 collapsed-B 边缘上，完整 factor component 的 Houdayer swap 虽代数正确，
但 16 个 P/low-energy-L、120 个 L/L 和 64 个 P/rank-complete-L 冻结对的 B masks 全部相同；物理 logical
label 的低能变化全在已被 HP64 精确热浴的 A 中。独立 U/U 对有 284 个不同 B bit，却只有一个完整 component，
仍只是 whole-pair exchange。small-HGP 的转换、不变量、involution、detailed balance 和 stationarity 都已穷举
通过，故终态是 `COLLAPSED_B_HCA_NO_LOW_ENERGY_RECOMBINATION`，不是实现失败。**“对真实 state 有变动”或
“pair 代数正确”不等于推动交付所需的 B/logical 慢变量。** 不得据此实现 HP64+B-HCA、优化其 acceptance/state
changes 或把它包装成独立 confirmation；该结构性负结果只排除该 frozen collapsed-B HCA，不表示 HP64、HCA、
q=0 或后验数学上不可能。

同日的 `044_q0_bp_dominance_witness_feasibility_20260724` 钉住了 importance/rejection 路线的另一处
容易自欺的环节：两个 BP-systematic source 的有限 ESS 看起来很好，并不自动给出 `pi/q` 的全局上界。对
1691 个预冻结合法 planted/logical/systematic-coordinate witnesses，以精确三 component mixture density 和
outward Decimal rounding 计算后，唯一无需解决原问题的 normalizer 上界
`Z<=.96^-1600` 只给出 `sup(pi/q)` 的微小下界（forward `5.53e-63`、reverse `2.54e-53`），所以终态是
`BP_MIXTURE_REJECTION_ENVELOPE_WITNESS_INCONCLUSIVE`，绝不是 BP 通过。**一个 proposal 的局部 overlap、
低 jackknife SE 或 full support 都不能替代 tight global normalizer/tail bound；用过松的 `Pr(y)<=1` 也不能
把“没有找到坏 witness”说成 coverage。** 不得据此开 BP-only IID/rejection sampler、报告 q_top 或以 P/common
start 绕过 MCMC 对抗初态；要继续此路线须先独立得到紧的 hard-coset normalizer 上界，而那正是尚未解决的
global-mass 问题。

随后 fresh `exp102.q0_bp_imh.local.v1` 直接把 BP-SYS-F64/R64 的精确 full-support mixture 用作
independence-MH proposal；small-HGP 完整 transition matrix、detailed balance/stationarity、18 项 focused
测试、24/24 raw replay 和不调用 sampler/runner 的 55296-step `allow_pickle=False` audit 均通过，但终态为
`LOCAL_BP_IMH_TRANSPORT_UNRESOLVED`。P 与 8 个不同合法低能 L 在 burn/measurement 都是零真实移动；U
虽在 burn 用 1--3 次真实移动冷却，却全部落到同一个 weight-62、P-label state，measurement 仅 0--2 次真实
移动。P 最大 measurement log acceptance 也只有 `-53.13`，L 最好为 `-47.79`（最差 `-88.69`），说明
proposal 对 high-`pi/q` 低能态严重供给不足；大量 U accepted counters 是同态 self-proposal，不是运输。
P/L 的 full-label `D2_norm=1`，U/L 为 `.998413`。注意 full-label D2 是本次 raw 前补上的必要门：相同 purity
和全部 basis means 仍可能对应互不相交的 sector supports。045 v0 仅因 relative output receipt 路径错误在
首 raw 后终止为 infrastructure failure，raw 禁用；046 v1 使用 fresh contract/config/seeds，零复用。不得把
BP 当 U 冷却器再接旧 full-row Gibbs 就宣称成功：BP 把 U 全送入 P logical label，而旧 full-row 又把 P/L
送入同一冻结 B basin，三族一致可能只是共同塌缩。后继必须有结果无关的 high-`pi/q` signature/basin coverage
与独立 B/tail 证据，不能只优化 accepted/self-loop 计数、统一初态、延长链或直接送 HARD2/remote。

2026-07-24 的 047--051 又排除了一个“共同落入低能 basin 就算收敛”的盲区。truth-free dressed logical
XOR catalog 虽代数正确且 signature rank=64，但 T3 下 BASE/P 可达 rank 仅 `4/1`，并会把全部低能 L
向同一 label 拉回，故终止为 `LOCAL_CENTER_PRESERVING_STRUCTURE_NOT_VIABLE`。exact random-scan
full-B-column Gibbs 的 small-HGP detailed balance/stationarity 与 bit replay 通过；但 049 短跑中 P/L 的
B 几乎冻结，`A|B` 精确重抽仍会制造 visible logical-label changes，U 的 B weight/likelihood 仍完全分离，
所以 **logical/state change 不能替代 B 慢变量门**。050 的两个 truth-free MAP anchors 只证明 T1 下某一
两列桥有足够 expected first departures，不是 sampler pass；051 独立重算保留 047/049 失败和 050 的窄权限。

fresh `exp102.q0_random_full_column.t1_m8.v0` 已冻结并完成三节点 preflight，但没有 measurement raw。它不用物理零态（该非零
syndrome 下非法；shifted zero 就是 P），而用 P、独立 exact-K0 U、两个 B-distinct MAP 及 8 个低能
B/logical-distinct S starts 各 8 条，固定 `2048+8192` clocks。S 中故意保留一个与 MAP 同 B、不同 logical
label 的起点，以区分 A/logical redraw 与真实 B transport。三节点 clean-archive preflight 必须在固定四并发
下 exact digest 一致且 replay-inclusive 单 trajectory 投影 `<=2h` 才能启动；本地四并发超时不具有远端
判定权限。门禁必须保留 character-U-statistic q_top/D2、full/B weight、B likelihood、全部 B bit/row/column
和 dense characters、logical characters、Rhat/ESS、constant-character burn crossing、MAP 双向 basin visits。
immutable run `exp102_q0_rfcg_t1_m8_20260724_6fa489f` 的三节点 mass/transcript exact consensus 通过，
但 nd-1/2/3 replay-inclusive 单 trajectory 投影分别为 `24701.47/24812.06/29871.42s`，全部超过冻结的
7200s 上限，故 aggregate 终态为 `RUNTIME_EXHAUSTED`，measurement raw 数为 0。本地 conda-12 独立审计
复核全部 self-hash、control/schedule、40-task ownership、runtime 算术和 raw absence，audit SHA 为
`817425dbaa6a9e5d90d03d34efe16f957beb7424eddd27dcde7cf12d60d75c6d`。这不是收敛失败、物理参数点失败或
数学不可能，只说明该冻结实现/clock/replay/并发无法满足两小时资源契约；不得绕过 gate、缩短链或事后改 cap。
没有 m6/HARD2/formal/held-out/production 权限；若做性能后继必须另立 fresh contract/source/seeds/raw。

其 memory-streaming 后继 validation 053 在 macmini 上 12/12 完整 CDF byte equality、`4.9391x` speedup
和 `2432.39s` T1 投影均过，但 fresh 三节点 run
`exp102_q0_streaming_preflight_20260724_de68bbc` 终止为 **`CONFLICT` 且独立 runtime-exhausted**，没有
T1 raw。Linux 三节点都只有 `U0,column=11` 的 legacy-dense/streaming CDF byte mismatch；proposed
streaming CDF 的完整 SHA catalog 和四条 PortablePrng sampling/replay transcript 反而在 macmini/三节点
完全一致，所以不得把它误写成 streaming sampler 跨节点随机漂移，也不得事后忽略冻结的 any-mismatch
门。更独立的阻断是 nd-1/2/3 speedup 仅 `2.5911/2.5372/1.3823x`、T1 replay-inclusive 投影为
`8797.83/9144.89/17760.30s`，仍未过 `4.2x/7200s` 门。audit SHA 为
`6426a1a01c01747f474d587a10cdb6db9e53db09112193499a8f9307adb7640f`。后继若利用正质量范围改成直接
weight/fixed-block exact heatbath，必须 fresh source/contract 并重做 small exact、underflow、portable
replay 与三节点 runtime；不得复用 053、删 P/U/MAP/S、缩短 T1 或先看 q_top。

2026-07-24 的 direct-positive fixed-block 后继 validation 054 已通过。fresh immutable run
`exp102_q0_direct_block_preflight_20260724_61d605a`（source
`61d605a5e27db0970457736c72d1c45d72a12b10`、archive
`61bb87e70320f7371504ea99c320e49baf1140b4ac9d3050fc9a3b742d5a7bec`）在 macmini/三 Linux 节点精确复现
12 个 frozen block-subtotal SHA 与四条 P/M0/S0/U0 PortablePrng sampling+replay transcript；三节点
replay-inclusive T1 投影为 `4144.85/4139.52/5454.14s < 7200s`，aggregate=`PASS`。完整 `2^24`
权重检查的 worst scaled absolute/relative/TV 为 `2.020606e-14/7.290711e-14/4.148991e-15`，候选
log-weight lower bound `-221.658`，没有接近 binary64 underflow。独立审计为
`INDEPENDENT_AUDIT_PASS_DIRECT_BLOCK_PREFLIGHT_CONFIRMED`（SHA
`9646c6f92070024680728bf377e802e647b48a2b66ca6210c89c436fbd70f539`）。

该 PASS 只证明 `RFCG-C24-DPB12-S1` 的 exact conditional、portable replay 和资源可行，只授权 fresh m8
T1 diagnostic；不是混合、q_top 或参数点认证。T1 必须另立 contract/source/seeds/raw，保持
`2048+8192` fixed clocks、full replay 和 `P/U/M0/M1/S` 各 8 条独立轨迹，并保留 validation 052 的
full/B D2、weight/likelihood、B bit/row/column/dense、logical、Rhat/ESS、burn crossing、双向 MAP
basin 与 B-column/label-change 门。非零 syndrome 下物理零态不在支持集，shifted zero 已是 P；全部从
P/零态开始只会掩盖慢混合。T1 未通过前不得运行 m6/HARD2 或 formal/held-out/production。

fresh successor validation 055
`exp102.q0_random_full_column_direct_block.t1_m8.v1` 已终止为 **preflight
`RUNTIME_EXHAUSTED`，measurement raw=0**。它保持 validation 052 的 P/U/M0/M1/S 几何，但 fresh control
重抽全部 schedule seeds 与 logical/B characters；四类 seed 与 052 overlap 均为 0，并 byte-bind validation
054 的两个 sampler 源文件与 portable artifact。pre-run red-team 用真实 miniature direct raw+full replay
修复了 direct engine 身份、`state_label` import 和 B-likelihood sum-order 三个 dormant analyzer 坑。

前两个 schedule attempt 都在 control 前因提前创建 fresh run root 而基础设施失败；第三个 immutable run
`exp102_q0_direct_block_t1_m8_20260724_146ef55_r3` 才是权威证据。最终 source 的完整 054 portable/runtime
preflight 三节点 exact consensus PASS，T1 投影 `4216.16/4149.15/4549.57s`；但 055 自己冻结的 probe 只测
`2+8` updates，却把含固定初始化/runner 开销的总时间线性外推到 10240 updates 再乘 2，得到
`9272.13/8779.07/13638.99s > 7200s`，所以 schedule 正确阻止 measurement。独立审计状态为
`INDEPENDENT_AUDIT_PASS_PORTABLE_PASS_T1_RUNTIME_EXHAUSTED_CONFIRMED`（SHA
`00622194dc370a66e08a0b94a7108b324aa49322de648fda7656f2c6ed5fc665`）。这不是 sampler/参数点失败，
也不得事后解释成 PASS；后继须另立 fresh contract/source/schedule/seeds/raw，用代表 steady-state 且包含 full
replay 的 probe 或冻结 intercept/slope 估计，同时不改 T1、7200s cap、五类初态和统计门。合同与证据见
`RANDOM_FULL_COLUMN_DIRECT_BLOCK_T1_CONTRACT.md`、`validation/055_*/`。

fresh validation 056 已冻结
`exp102.q0_random_full_column_direct_block.t1_m8.v2`，目前是 **pre-schedule/pre-measurement**。它没有改
sampler、T1、7200s cap、P/U/M0/M1/S x8 或任何统计门，只替换 055 的错误 runtime estimator。每节点用
两个独立 cold 4-process batch 跑 `8+128` 与 `16+256` updates，P/M0/S0/U0 均实际执行 sampling+full bit
replay；sampling/replay 分别拟合一次 startup intercept 与 per-update slope，投影 10240 updates 后乘原 factor
2 并取最坏 family。任一 component slope 非正即 `RUNTIME_ESTIMATOR_UNSTABLE`，不得 favorable fallback。
本地真实 smoke 全部 slope 为正，最坏投影约 `1348.4s`，但不能替代三 Linux 节点。config/control/manifest
SHA 为 `70285cf7...899d / 49665fb9...9c48 / fd31f5a7...3ce7`；fresh logical/B characters 为
`347ec3cf...8467 / 4cdbaf99...03be`，四类 40-task seeds 与 055 均无 overlap。conda-12 回归 exp102
`617 passed`、exp101 `366 passed`。最终 source 仍须完整重跑 054 portable/runtime 和 056 schedule-bound
preflight，双 aggregate PASS 前禁止 measurement；详见
`RANDOM_FULL_COLUMN_DIRECT_BLOCK_T1_V2_CONTRACT.md`、`validation/056_*/PRE_RUN_RED_TEAM.md`。

若要继续正式实验，须另立经审查的新算法/科学契约与 fresh tuning/held-out，不得把重复 T3、延长链、
删困难 disorder、缩范围或改窗口包装成原 discovery 成功。
PT/PA/global discovery raw 均不得进入正式 merge/freezer。48-code registry 已冻结，但正式新
sampler config、pilot、held-out 和 6144 个生产任务均不存在。生产 worker 必须看到
`FROZEN_HELD_OUT_PASS` 才会启动，禁止手工绕过；
正式读取只用 `load_exp102_publication_q_top`，不得交给 exp101 loader。接手 exp102 先读该目录的
`EXPERIMENT_CONTRACT.md`、`GLOBAL_DISCOVERY_CONTRACT.md`、
`GLOBAL_SCREEN_DIAGNOSTIC_CONTRACT.md` 与 `status.md`。

- **生产 convention 固定**：`sector=x_error`、`H_check=H_Z`、稳定子 move=`H_X` 行、logical move=`logical_X`、observable=`logical_Z`、制备态=`|+>_L`；对偶 `z_error/H_X` 才对应 `|0>_L`。现有矩阵接线不交换。
- **跨环境 proposal artifact**：BpLSD/MILP 的多个版本可对同一 hard coset 给出不同但都代数有效的 anchor；不得要求这些重建结果逐字相同，也不得混用。`q0_logical_stratified.v0.v2` 只允许 `nd-1` 构造一次 artifact，其余节点/macmini 只审计同一冻结字节、代数和离散回放；不通过则 `CONFLICT`。
- **生产 posterior 固定**：`pi(e|y_eff) ∝ exp[-K_p|e|-K_q|H_check e xor y_eff|]`，`y_eff=H_check epsilon_data_true xor measurement_error`；真实错误不得直接进入能量、Metropolis/TI/PT 比值。正式系综名为 `true_posterior`、`legacy_delta_only`，`paper_true_posterior/repo_compat` 只作先归一化的 deprecated alias。
- **三种 section 不混用**：meta-check measurement-error decoder、preparation-chain representative、logical-sector section 的 domain/用途不同；q>0 的一般 `effective_syndrome` 不在 `im(H_check)`，禁止传给 logical-sector section。
- **统计量不混称**：同时区分 absolute/relative characters、`posterior_mass_on_planted_class`、`posterior_purity` 与 `map_success_probability`；公共 `w0` 已废弃。exact/解析端点只填 algebraic MAP bounds，普通 TI 与 sampled 只填 plug-in estimated bounds，后者统一标注 `no confidence coverage`。只保证 boundary-only section shift 不变，不再声称任意 frame change 是 gauge。
- **引擎和 disorder gate**：full-sector TI 仅 `k<=10`；large-k pairwise 只输出 free-energy-gap diagnostics，不得有 `m_u/q_top`；`k>10,q>0` 用四独立 PT observable instances，`q=0` 用 validated 8-start。任一 disorder gate 失败必须保存 raw 值并标 `INVALID`；无偏 U-statistic 在有限样本下可为负或越出物理区间，禁止裁剪。
- **参数点级 fail-closed**：正式 mean、SEM 与 crossing/FSS 输入只在所有 planned disorders 均存在且 `valid_for_aggregation=true` 时输出。一个 invalid 即 `SAMPLING_INSUFFICIENT`，一个 missing 即 `INCOMPLETE`，整点正式输出全部为 NaN；valid-only 条件均值/SEM 仅供诊断，因条件选择偏差不得用于 crossing/FSS。所有 fraction 以 planned disorders 为分母，公共 `pass_fraction` 已删除。
- **scan v3 与 loader**：输出仍为 `scan_results.npz`，但 v1/v2 chunk 和 v2 NPZ 永不复用为 v3 正式结果。publication/FSS 必须通过 `src.scan_results.load_publication_q_top` 读取；loader 只接受 `exp101.scan.v3 + true_posterior` 的 `REPORTABLE` 点，不从 v2 条件均值推断 eligibility。当前 scan v3 权威回归只认 `validation/015_aggregation_safety_20260714/`。
- **运行坑仍有效**：本地 `conda run -n 12` 加 `--no-capture-output`；多进程显式传 `--num-workers N`，不要依赖 screen 内的 `$(nproc)`。
- **clean-source tree 不得直接跑 Python**：远端 `repos/<run>/source/` 必须保持与 archive 逐文件一致；生成 control、benchmark 和 orchestrator 都要走 `run_verified_source.sh`（或至少显式禁止 bytecode）。直接运行会留下 `__pycache__`，随后所有节点会被 verified wrapper 以 exit 67 fail-closed；不得删除 FAILED marker 原地重跑，须用 fresh run/deployment 留审计链。

下方「物理图像与 L·T·S 分解」及其 Gibbs 公式描述旧 3D toric 程序，只用于 legacy 3D 工作，
不得覆盖 exp101 的 `physics.v2 / scan.v3` 契约。

## 物理图像与 L·T·S 分解（核心概念对齐）

模拟对象是 3D toric code 纠 X 错误的统计力学模型。MCMC 更新的是 edge 上的 X-error 构型 `c ∈ C_1 = F_2^n`（`n = 3L³`，逻辑比特数 `k = 3`）。固定 disorder `(s, η)`：`s` 是带测量噪声的 syndrome，`η` 是真实 data error。目标是采样该 disorder 下的 Gibbs 热态

`π(c) ∝ exp[ −K_p·|c ⊕ η| − K_q·|H_Z c ⊕ s| ]`，其中 `K_p = log((1−p)/p)`、`K_q = log((1−q)/q)`。

把 `c`（更一般地辛空间 `F_2^{2n}` 里任意 Pauli，在一组固定基下）唯一分解为 `c = T ⊕ L ⊕ S`：

- **S（stabilizer）**：`S ∈ im(H_X^T)`，`H_Z S = 0` 且逻辑平凡 = vertex star / contractible loop。改 S 不动 syndrome、不动逻辑类，只在固定 (syndrome, 逻辑类) 的 coset 内做微观重排。`dim = n − ρ − k`。
- **L（logical）**：逻辑算符代表元 = winding / Wilson loop（非平凡 1-cycle）。`H_Z L = 0`，改 L 不动 syndrome，但要翻一整条长度 `~L` 的非收缩环，data 权重变 `~K_p·L`。`dim = k`。**测的 Wilson loop 可观测量就是 L 的指示量。**
- **T（destabilizer / syndrome）**：与 stabilizer 对偶、携带 syndrome 的代表元，`H_Z|_T` 单射。改 T 会改变 `H_Z c`，`q>0` 花 `K_q` 能量，`q=0` 被硬约束 `H_Z c = s` 钉死。`dim = ρ = rank(H_Z)`。代码里的线性 section `r(σ)`（syndrome→代表链）就是 T 空间的一组代表元。

MCMC 三类 move ↔ 该分解：single-bit 翻转一般同时动 **T 与 S**（采样 syndrome 涨落 T 的主力）；contractible 零-syndrome move / cluster update 只动 **S**；**winding 零-syndrome move 只动 L，即"逻辑扇区翻转 / sector flip"**。

观测量 `O_u = (−1)^<z_u, c + η + r(H_Z c) + r(H_Z η)>`：`c + r(H_Z c)` 去掉 c 的 T 分量（落进 `S⊕L`），与逻辑代表 `z_u` 配对又滤掉 S（`<z_u, S> = 0`），故 `O_u` 读出的是 **c 相对 η 的 L 分量（逻辑类）**。`q_top = mean_u(m_u²)` = 该逻辑类分布的纯度：`→1` 有序/可纠错，`→0` 无序/不可纠错。

**「sector 不翻转」= L 被冻结**：S、T 一直在被正常采样，但链从不在不同逻辑类之间移动。这是有序相里的破缺遍历性，与低温 Ising 单自旋翻转卡在某磁化扇区同构——翻 L 要插入 `~K_p·L` 的畴壁/winding，小 p（深有序相）+ 大 L 时势垒巨大（接受率 `~(p/(1−p))^L`，p=0.05 时 L=5 约 4e-7、L=6 约 2e-8），local + PT 在有限时间跨不过去 → `winding 接受率 ≈ 0` → `q_top` 被假性钉在 1、看不到相变 crossing。因此收敛诊断必须把**冷端 winding 接受率 / 逐温度 winding 是否传到冷档**作为硬判据；"不同冻结起点互相一致"不能当收敛证据（共冻 ≠ 收敛）。

文件结构：

- `src/main.py`：主入口，单尺寸/多尺寸扫描与结果保存
- `src/mcmc.py`：MCMC 状态初始化与采样
- `src/mcmc_parallel_tempering.py`：Parallel tempering 采样
- `src/mcmc_convergence_gate.py`：收敛 gate 诊断
- `src/mcmc_diagnostics.py`：MCMC 诊断工具
- `src/cluster_update.py`：Cluster update 算法
- `src/preprocessing.py`：预处理（checks 邻接表、logical observable masks）
- `src/linear_section.py`：GF(2) 线性截面构造
- `src/exact_enumeration.py`：小规模精确枚举/校验
- `src/build_toric_code_examples.py`：构造 toric code 输入（使用说明见 `toric_code_接口衔接`）
- `src/production_chunked_scan.py`：生产扫描提交、merge、preflight
- `src/profile_3d_q_positive.py`：3D q>0 性能/诊断 profiler
- `src/`：其余分析、绘图、诊断代码统一放这里
- `笔记/实验报告.md`：实验记录，按时间戳增量更新，务必中文、简洁清晰

运行规则：

- 运行完整实验之后要更新 `笔记/实验报告.md`
- 版本大改并完成必要验证之后，应只提交相关文件，使用清晰 commit message，并 push 到 GitHub；不要用 `git add .` 混入无关数据或临时产物。
- 使用服务器规则:
  - 使用命令`ssh yuany`可以登录到存储节点(nd-0)，文件传输可以在这个节点与本地实现
  - 当登录到nd-0之后进一步使用命令`ssh nd-3`或`ssh nd-1`或`ssh nd-2`可以登录到计算节点，计算节点与存储节点共享存储，计算应在这个节点开展
  - 运行python请使用名为`11`的conda环境，运行请开启`screen`后台运行
  - **服务器根目录唯一为 `~/.single_shot/`（launcher 的 `REMOTE_BASE`），是 Project D 在服务器上的唯一落脚点**；其它项目（N01 / A14 / QEDC / BP_OSD / QEM_QEC 等）与本项目无关，不要读写或混入。根目录下只长期保留两个子目录：`runs/`（唯一的实验产物文件夹，每次实验一个带编号子目录 `expNN_…`，与本地 `data/` 编号一一对应）和 `logs/`（launcher 日志）。索引见 `~/.single_shot/SERVER_README.md`。
  - **保持有序、用完即清**：`repos/`（每次 launch tar 过去的 src 副本）、`mpl-cache/`、`*code*` 快照、profile/smoke/benchmark 等都是 scratch，由 launcher 自动重建——把一个 run 的结果 tar 回本地 `data/` 并校验后，应删掉该 run 的 `repos/` 副本和临时 cache，长期只留 `runs/` 下最终 NPZ/产物。不要再在根目录散放代码快照或散乱命名的 run。
  - 本地 `data/` + git 是 single source of truth；服务器 `runs/` 只是 NPZ 的异地备份，不在服务器上做分析。（2026-06-15 已清理：删除 ~30G repos/代码快照/profile/smoke/早期摸底 scratch，59 个编号实验归拢到 `runs/`。）
- 快速测试技巧：可以不做disorder sample，只看一个disorder固定为0的内部有没有相变
- 每次实验放在一个文件夹下，文件夹命名规则类似`exp1_极简实验内容_20260501_日内时间戳`。每次实验按顺序编号命名
- exp36 起的迭代优化实验统一使用带序号子文件夹，例如 `001_cluster_stage_diag_smoke_20260528/`、`002_cluster_stage_repeats_20260528/`；同一轮远端多节点结果放在该编号目录下的不同 run 子目录。
- 每次实验如果分多节点分批次运行，请全部放在一个文件夹下的不同子文件夹，用来保存每个节点的运行代码，每个节点的运行数据
- 本地实验、smoke、profile 和 sanity check 只要会产生需要查看或留档的数据，都必须输出到本项目的 `data/` 目录下；不要把结果放到项目外的 `/tmp`、系统临时目录或其他不可见位置。临时 cache 可以例外，但最终 JSON/NPZ/PNG/MD/CSV 产物必须在 `project D/data/` 内。

实验参数陷阱：

- threshold 方向判读：当 `p < threshold` 时，code 越大错误应该越小，也就是 `q_top` 越大；当 `p > threshold` 时反过来，code 越大错误率越大，也就是 `q_top` 越小。
- `q=0` 生产扫描不能传 `--pt-*` 参数；parallel tempering 只支持 `q>0`，否则会在 preflight 阶段报 `parallel tempering is only supported for q>0`。
- `q=0` 多起点扫描必须确认实际使用 `q0_num_start_chains=8` 或等价 `num_start_chains=8`；若误传 `--num-start-chains 1`，会覆盖 `q0_num_start_chains`，导致多起点 spread 诊断失效。
- 新 run 若曲线明显偏离，应先检查 manifest 中的 `common_random_disorder_across_p`、`num_start_chains`、`q0_num_start_chains`、`pt_num_temperatures`、burn-in 设置、Numba 是否启用和 commit SHA。
- `r` 的数学定义已经从"线性截面"修正为"任意 section"；代码里优先走 `ldpc.BpLsdDecoder` 风格的 syndrome-to-chain 代表元映射，失败再回退到高斯消元。
- 逻辑观测量必须按修正公式 `(-1)^<z_u, c + eta + r(H_Z c) + r(H_Z eta)>` 计算；不要恢复旧的 `r^T`/factorized mask 线性化路径。
- 旧的 `q!=0` 数值结果基于错误的线性化 observable 公式，已经失效，后续分析不要沿用；exp37 的 `020/028` 还用了旧 `x+r(Hx)` sector 标签，也只保留作审计。`p=0.05,q=0.08..0.23,L=3,4,5` 的 corrected AIS 最终表以 `exp37/032_final_corrected_qgrid_20260603` 为准。早期 AIS NPZ 可能没有 `delta_f_stderr_per_disorder`，优先用 `032` 或重新跑修正后的聚合代码。

服务器/性能运行坑：

- 远端 `conda run -n 11 python - <<'PY' ...` 可能吞掉 stdin 或产生空输出；复杂脚本优先写到临时 `.py` 文件再 `conda run -n 11 python script.py`。
- 远端 conda/镜像解析不稳定；安装或假设依赖前先检查 `conda run -n 11 python -c "import ..."`，不要在生产任务中临时装包。
- 3D L=5 的默认 burn-in 会按 `num_qubits/18` 放大，`1200` 会变成约 `25000`；若只是侦察或 PT 已稳定，考虑显式设置 `--max-effective-num-burn-in-sweeps` 并记录诊断。
- `production_chunked_scan.py` 已优先调度大 L chunk；若 L=5 仍有长尾，可进一步减小 `chunk_size`。
- Numba 是可选加速依赖：有 `numba` 时 3D 主路径会走 JIT fast path，没有时自动回退；远端节点升级/换环境后要先用小 benchmark 确认确实启用。
- `profile_3d_q_positive.py` 是 opt-in 诊断 runner，不是生产 threshold 扫描；对比 config 时应确认 manifest 中 `disorder_seed_scope=lattice_size,p_value,q_value,disorder_index`，避免不同 config 使用不同 disorder 造成 A/B 噪声。
- exp35 的同步放大 PT 已接入 production scan：`--pt-ladder-mode sync_enlarge --pt-q-hot ... --adaptive-pt-rounds ...` 会同时改变 `p_k/q_k`，swap 权重也同时包含 data/syndrome term；cluster update 现在支持随温度变化的 `q_k` ladder，但只在所有 `p_k,q_k<0.5` 时有效。若热端超过该范围，应显式 `--disable-cluster-update`。
- 同步放大 ladder 会按 `q_hot/q_cold` 的 odds 比例同步放大 `p`，若 `p_cold` 偏高或 `q_hot` 太大，热端 `p_k` 可能超过 `0.5` 并在 submit/preflight 报错；exp35 固定 `p=0.05,q_hot=0.44` 是可行组合。
- 用 profiler 测 wall time 时优先设置 `--stage-signature-mode none`；逐环节 sector-change 诊断会反复计算 logical signature，开销可远大于真实 MCMC 更新。需要判断哪个环节改变 logical sector 时再单独用 `stage` 模式小样本跑。
- `screen` 中用 `conda run` 时日志可能被 capture/buffer；新 launcher 已使用 `conda run --no-capture-output`，旧 run 若日志为空应以 `profile_summary.json/md` 和 raw JSON/NPZ 为准。
- screen/`bash -lc` 会话内 `$(nproc)` 在 cgroup quota 下可能误报 `1`，使 `--num-workers $(nproc)` 退化为**串行**(load≈1、py 进程数≈1)；launcher 应在启动时(screen 外，nproc 可靠)探测核数并烘焙进 runner，或显式传 `--num-workers N`，并在日志确认 `workers=N`、load≈N 后再放手。nd 计算节点实测 80/80/96 核(affinity 0-303)。
- q>0 的 `q_positive_initial_chain_mode=sector` 只能用 zero-syndrome sector representatives 生成不同初态；不要对带 measurement noise 的 observed syndrome 求 section representative，因为它不一定属于 `im(H_Z)`，会导致 section/decoder 路径卡住或产生无意义初态。q=0 多起点路径才需要 syndrome representative。
