本文基于 `架构设计.md` 开发数值模拟程序。

使用python开发，使用已有的名为`12`的conda环境



文件结构：

- `main.py`：主入口，组织单尺寸/多尺寸扫描与结果保存
- `mcmc.py`：MCMC 状态初始化与采样相关逻辑
- `preprocessing.py`：预处理逻辑，包括 checks 邻接表与 logical observable masks
- `linear_section.py`：GF(2) 线性截面构造
- `exact_enumeration.py`：小规模精确枚举/校验相关逻辑
- `build_toric_code_examples.py`：构造 toric code 输入, `toric_code_接口衔接`是这个程序的使用说明。
- `架构设计.md`：程序设计与接口依据
- `实验报告.md`：实验记录，后续按时间戳增量更新
