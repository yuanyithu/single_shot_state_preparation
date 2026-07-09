本文是开发本数值模拟程序的操作契约与避坑清单；物理图像与核心概念见下方「物理图像与 L·T·S 分解」一节。

使用python开发，本地使用已有的名为`12`的conda环境 本地运行和验证统一使用名为`12`的conda环境，不要默认切到别的环境

维护本文档的规则（每次完成复杂开发后执行）：

- 优先**就地改写**已有条目，而不是在底部不断追加新条目；同一个坑有了新认识就改原条目，避免各节无限增长、前后矛盾。
- 清理判据是「这条坑是否还会再次绊到 agent」：只要某个坑在后续 run 里仍可能复现、需要主动规避，就保留在本文档——无论它是通用规律还是带具体参数实例的个案（如 `exp35 p=0.05,q_hot=0.44 可行`）。只有已被代码或流程修掉、不可能再触发的历史记录，才删除或下沉到 `笔记/实验报告.md`。

## 当前主线：expander code（exp101 起，2026-07-07~）

项目从 3D toric 转向 quantum expander code（(3,4)-biregular 随机图 HGP）单发制备 q_top。**exp101 已交付经全面验证的可复用管线**（`data/expander_code/exp101/src`，259 tests + V1–V6 全绿；契约 `exp101/plan.md`、进度 `exp101/status.md`、结题 `exp101/report.md`）。exp102+ 直接复用 `exp101/src`。**接手 expander 工作先读 exp101/report.md + status.md。** 关键避坑（会再绊到）：

- **大 k（k=m²>10）q_top 必须用 direct/PT 采样观测量，禁用 pairwise-TI**：pairwise-TI（假扇区自由能可加）已在 K43(k=13) 证实失效（对精确 m_u 偏差达 1.55/满量程 2，validation/007）。full-TI 仅 k≤10 有效（小码交叉验证）。
- **双系综区分**：exp101 支持 `true_posterior`（双盘度 |c⊕η|+|Hc⊕s|，decoding 正解，Nishimori E[m]=E[m²] 成立）与 `repo_compat`（δ-only，等价 3D 时代模型）。**3D 时代 exp40/41 相图是 δ-only 系综结果**，非标准 decoding posterior；作阈值引用需重审。跨 run 比较必须同系综 + 同 section frame（q>0 时 m_u frame 依赖=gauge）。
- **PT 是纯 python（未 numba）**：direct 引擎 numba 快（m=6 ~1.1s/disorder），PT（冷区 sector 传输）大 m 慢（m=6 ~302s/disorder）。若成生产瓶颈：PT 内循环 numba 化 或 decoder-informed 初始化。
- **direct 单链在冷区大 k 冻结**（q_top 假值，非物理）——冷区/crossing 必须用 PT；per-u worst-u 冷端 logical 接受率 + PT round-trip 是收敛硬判据（`gates.py`，「共冻≠收敛」内建，q_top spread 符号盲，另有 m_u_spread 符号敏感判据）。
- **本地 `conda run -n 12` 必须加 `--no-capture-output`**：否则整体捕获子进程输出直到退出，长/被杀任务看似零输出、无法看进度（本地与远端同此坑）。
- run_scan 支持多进程并行（`--num-workers N`，spawn context 避 numba+fork 死锁）；跨节点 bit-identical（可移植 PRNG）；生产须显式传 N（勿依赖 in-screen `$(nproc)`）。

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
