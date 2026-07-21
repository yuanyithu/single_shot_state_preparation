本文是开发本数值模拟程序的操作契约与避坑清单；物理图像与核心概念见下方「物理图像与 L·T·S 分解」一节。

使用python开发，本地使用已有的名为`12`的conda环境 本地运行和验证统一使用名为`12`的conda环境，不要默认切到别的环境

维护本文档的规则（每次完成复杂开发后执行）：

- 优先**就地改写**已有条目，而不是在底部不断追加新条目；同一个坑有了新认识就改原条目，避免各节无限增长、前后矛盾。
- 清理判据是「这条坑是否还会再次绊到 agent」：只要某个坑在后续 run 里仍可能复现、需要主动规避，就保留在本文档——无论它是通用规律还是带具体参数实例的个案（如 `exp35 p=0.05,q_hot=0.44 可行`）。只有已被代码或流程修掉、不可能再触发的历史记录，才删除或下沉到 `笔记/实验报告.md`。

## 当前主线：expander code（exp101 起，2026-07-07~）

项目从 3D toric 转向 quantum expander code（(3,4)-biregular 随机图 HGP）单发制备 q_top。
**exp101 的 `exp101.physics.v2` 已由 `validation/014_paper_alignment_20260713/` 认证；
`exp101.scan.v3` 已由 `validation/015_aggregation_safety_20260714/` 认证，可用于严格门禁后的正式
publication/FSS。014 中的 scan v2 聚合只作历史审计；旧 259 tests 与 V1–V6 仍全部是
`PRE_ALIGNMENT`。exp102 复用当前管线时必须遵守下列生产约束。**接手先读
`data/expander_code/exp101/PHYSICS_CONTRACT.md`（唯一物理权威）、`status.md` 和
`validation/README.md`。关键硬约束：

**exp102 当前为 `GLOBAL-SAMPLING DISCOVERY PREFLIGHT REPAIR / PRE-RUN`，不是已有物理结果。** 正式历史契约仍为
`exp102.physics.v1 / exp102.q0_pt.v1 / exp102.scan.v1`。2026-07-20 的固定 Q32 + multi-swap PT-v2
discovery 已因 96 条实例轨迹认证往返总数为 0 而 `EXHAUSTED`；不得追加 S128、延长轮数或复用
其 raw。随后 `exp102.q0_pa.discovery.v1` 的三节点 digest、Linux runtime、四任务 PT transport
autopsy 和 64-task PA hard screen 已全部完成：四个 autopsy 均因所需条件 attempts<200 而
`INCONCLUSIVE`；`C192-2/B96-1/B192-1/B96-2` 全部在两个 hard cells 上因 genealogy 灾难性塌缩
失败（median family ESS≈1、distinct families=1--2）。按冻结零通过分支，PA 同样 `EXHAUSTED`，
禁止 B384-2 rescue，confirmation/resolution manifests 未创建，也没有 `READY_FOR_FORMAL`。
新的 `exp102.q0_global.discovery.v1` 已实现低权重 logical catalog、hard-coset cluster/joint
heatbath、独立 defect trace、m3 full-sector TI、三节点 digest/runtime、72h schedule 与 control freeze，
首个 immutable run `exp102_q0_global_20260721_6f26fd5` 已在 Linux preflight 阶段因 archive
provenance、可选 BP-LSD cross-check、测试写 source tree 与 cold-JIT TI projection 问题永久 FAILED；
没有产生合格 runtime/digest/WMC 或 sampler raw，不得删 marker 原地复跑。修复证据见
`validation/008_q0_global_preflight_portability_20260721/`，必须用 fresh commit/deployment/run/schedule。
screen/HARD2/confirmation/resolution/TI 尚未运行，因此仍没有新物理结果。
接手必须先读 `GLOBAL_DISCOVERY_CONTRACT.md`；不得在运行前改 panels/gates/T/2T/bias 规则，也不得把
实现完成写成 `READY_FOR_FORMAL`。即使 discovery 全过，也只能另建正式 tuning/held-out 契约，不能
直接启动 production。PT/PA/global discovery raw 均不得进入正式 merge/freezer。48-code registry
已冻结，但正式新 sampler config、pilot、held-out 和 6144 个生产任务均不存在。生产 worker 必须看到 `FROZEN_HELD_OUT_PASS` 才会
启动，禁止手工绕过；
正式读取只用 `load_exp102_publication_q_top`，不得交给 exp101 loader。接手 exp102 先读该目录的
`EXPERIMENT_CONTRACT.md`、`GLOBAL_DISCOVERY_CONTRACT.md` 与 `status.md`。

- **生产 convention 固定**：`sector=x_error`、`H_check=H_Z`、稳定子 move=`H_X` 行、logical move=`logical_X`、observable=`logical_Z`、制备态=`|+>_L`；对偶 `z_error/H_X` 才对应 `|0>_L`。现有矩阵接线不交换。
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
