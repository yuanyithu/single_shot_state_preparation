本文基于 `笔记/架构设计.md` 开发数值模拟程序。

使用python开发，本地使用已有的名为`12`的conda环境
本地运行和验证统一使用名为`12`的conda环境，不要默认切到别的环境

每次完成复杂开发之后可以更新维护此文档，提醒避开可能的坑；并检查文档实效性，删掉过时内容



文件结构：

- `src/main.py`：主入口，组织单尺寸/多尺寸扫描与结果保存
- `src/mcmc.py`：MCMC 状态初始化与采样相关逻辑
- `src/mcmc_parallel_tempering.py`：Parallel tempering 采样
- `src/mcmc_convergence_gate.py`：收敛 gate 诊断
- `src/mcmc_diagnostics.py`：MCMC 诊断工具
- `src/cluster_update.py`：Cluster update 算法
- `src/preprocessing.py`：预处理逻辑，包括 checks 邻接表与 logical observable masks
- `src/linear_section.py`：GF(2) 线性截面构造
- `src/exact_enumeration.py`：小规模精确枚举/校验相关逻辑
- `src/build_toric_code_examples.py`：构造 toric code 输入, `toric_code_接口衔接`是这个程序的使用说明。
- `src/production_chunked_scan.py`：生产扫描提交、merge、preflight
- `src/profile_3d_q_positive.py`：3D q>0 性能/诊断 profiler
- `src/`：其余分析、绘图、诊断相关 Python 代码统一放在这里
- `笔记/架构设计.md`：程序设计与接口依据
- `笔记/实验报告.md`：实验记录，后续按时间戳增量更新，务必使用中文，保证简洁清晰

运行规则：
- 运行完整实验之后要更新 `笔记/实验报告.md`
- 版本大改并完成必要验证之后，应只提交相关文件，使用清晰 commit message，并 push 到 GitHub；不要用 `git add .` 混入无关数据或临时产物。
- 使用服务器规则:
    - 使用命令`ssh yuany`可以登录到存储节点(nd-0)，文件传输可以在这个节点与本地实现
    - 当登录到nd-0之后进一步使用命令`ssh nd-3`或`ssh nd-1`或`ssh nd-2`可以登录到计算节点，计算节点与存储节点共享存储，计算应在这个节点开展
    - 运行python请使用名为`11`的conda环境，运行请开启`screen`后台运行
- 快速测试技巧：可以不做disorder sample，只看一个disorder固定为0的内部有没有相变
- 每次实验放在一个文件夹下，文件夹命名规则类似`exp1_极简实验内容_20260501_日内时间戳`。
- 每次实验如果分多节点分批次运行，请全部放在一个文件夹下的不同子文件夹，用来保存每个节点的运行代码，每个节点的运行数据
- 本地实验、smoke、profile 和 sanity check 只要会产生需要查看或留档的数据，都必须输出到本项目的 `data/` 目录下；不要把结果放到项目外的 `/tmp`、系统临时目录或其他不可见位置。临时 cache 可以例外，但最终 JSON/NPZ/PNG/MD/CSV 产物必须在 `project D/data/` 内。

实验参数陷阱：
- threshold 方向判读：当 `p < threshold` 时，code 越大错误应该越小，也就是 `q_top` 越大；当 `p > threshold` 时反过来，code 越大错误率越大，也就是 `q_top` 越小。
- `q=0` 生产扫描不能传 `--pt-*` 参数；parallel tempering 只支持 `q>0`，否则会在 preflight 阶段报 `parallel tempering is only supported for q>0`。
- `q=0` 多起点扫描必须确认实际使用 `q0_num_start_chains=8` 或等价 `num_start_chains=8`；若误传 `--num-start-chains 1`，会覆盖 `q0_num_start_chains`，导致多起点 spread 诊断失效。
- 新 run 若曲线明显偏离，应先检查 manifest 中的 `common_random_disorder_across_p`、`num_start_chains`、`q0_num_start_chains`、`pt_num_temperatures`、burn-in 设置、Numba 是否启用和 commit SHA。
- `r` 的数学定义已经从"线性截面"修正为"任意 section"；代码里优先走 `ldpc.BpLsdDecoder` 风格的 syndrome-to-chain 代表元映射，失败再回退到高斯消元。
- 逻辑观测量必须按修正公式 `(-1)^<z_u, c + eta + r(H_Z c) + r(H_Z eta)>` 计算；不要恢复旧的 `r^T`/factorized mask 线性化路径。
- 旧的 `q!=0` 数值结果基于错误的线性化 observable 公式，已经失效，后续分析不要沿用。

服务器/性能运行坑：
- 远端 `conda run -n 11 python - <<'PY' ...` 可能吞掉 stdin 或产生空输出；复杂脚本优先写到临时 `.py` 文件再 `conda run -n 11 python script.py`。
- 远端 conda/镜像解析不稳定；安装或假设依赖前先检查 `conda run -n 11 python -c "import ..."`，不要在生产任务中临时装包。
- 3D L=5 的默认 burn-in 会按 `num_qubits/18` 放大，`1200` 会变成约 `25000`；若只是侦察或 PT 已稳定，考虑显式设置 `--max-effective-num-burn-in-sweeps` 并记录诊断。
- `production_chunked_scan.py` 已优先调度大 L chunk；若 L=5 仍有长尾，可进一步减小 `chunk_size`。
- Numba 是可选加速依赖：有 `numba` 时 3D 主路径会走 JIT fast path，没有时自动回退；远端节点升级/换环境后要先用小 benchmark 确认确实启用。
- `profile_3d_q_positive.py` 是 opt-in 诊断 runner，不是生产 threshold 扫描；对比 config 时应确认 manifest 中 `disorder_seed_scope=lattice_size,p_value,q_value,disorder_index`，避免不同 config 使用不同 disorder 造成 A/B 噪声。
- `profile_3d_q_positive.py --suite exp35` 的 adaptive PT 目前是 profiler-only 原型：同步放大 ladder 会同时改变 `p_k/q_k`，swap 权重也同时包含 data/syndrome term；不要把它当作 production scan 接口，且 exp35 默认/应保持 `cluster_update_enabled=False`。
- 用 profiler 测 wall time 时优先设置 `--stage-signature-mode none`；逐环节 sector-change 诊断会反复计算 logical signature，开销可远大于真实 MCMC 更新。需要判断哪个环节改变 logical sector 时再单独用 `stage` 模式小样本跑。
- `screen` 中用 `conda run` 时日志可能被 capture/buffer；新 launcher 已使用 `conda run --no-capture-output`，旧 run 若日志为空应以 `profile_summary.json/md` 和 raw JSON/NPZ 为准。
