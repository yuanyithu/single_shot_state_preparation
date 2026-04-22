本文基于 `架构设计.md` 开发数值模拟程序。

使用python开发，使用已有的名为`12`的conda环境



文件结构：

- `src/main.py`：主入口，组织单尺寸/多尺寸扫描与结果保存
- `src/mcmc.py`：MCMC 状态初始化与采样相关逻辑
- `src/preprocessing.py`：预处理逻辑，包括 checks 邻接表与 logical observable masks
- `src/linear_section.py`：GF(2) 线性截面构造
- `src/exact_enumeration.py`：小规模精确枚举/校验相关逻辑
- `src/build_toric_code_examples.py`：构造 toric code 输入, `toric_code_接口衔接`是这个程序的使用说明。
- `src/`：其余分析、绘图、生产扫描相关 Python 代码统一放在这里
- `架构设计.md`：程序设计与接口依据
- `实验报告.md`：实验记录，后续按时间戳增量更新

运行规则：
- 运行完整实验之后要更新 `实验报告.md`
- 版本大改并运行试验之后自动运行 `git add .` , `git commit -m "恰当的commit说明"` , `git push`, 要正确填写commit的说明
- 使用服务器规则:
    - 使用命令`ssh yuany`可以登录到存储节点(nd-0)，文件传输可以在这个节点与本地实现
    - 当登录到nd-0之后进一步使用命令`ssh nd-3`或`ssh nd-1`或`ssh nd-2`可以登录到计算节点，计算节点与存储节点共享存储，计算应在这个节点开展
    - 运行python请使用名为`11`的conda环境，运行请开启`screen`后台运行
