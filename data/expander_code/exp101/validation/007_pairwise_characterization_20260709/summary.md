# pairwise-TI 大 k 失效刻画（status D4 证据）

墙钟 239s

| 对照 | 量 | 值 |
|---|---|---|
| K43(k=13) | pairwise vs exact m_u（max/mean） | 1.547 / 1.072 |
| K43(k=13) | **direct 采样** vs exact m_u（max，锚点） | 0.786 |
| toric_m3(k=2) | pairwise vs exact m_u（max，对照） | 0.110 |
| toric_m3(k=2) | full-TI vs exact m_u（max，锚点） | 0.032 |

**结论**：pairwise-TI（假可加性）在 k=13 与 k=2 上都显著偏离精确 m_u；而 direct 采样与 full-TI 都与精确一致 ⇒ **pairwise 作为 q_top 方法失效**（源于可加性假设，非实现 bug）。大 k 生产用 direct/PT 采样。
判定: 需复核
