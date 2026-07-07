# 官方 (3,4) 家族注册表

- d_A=3, d_B=4, base_seed=12345
- 规则 full_rank：简单图 + 满秩（k=m²）；full_rank_d3：再加 H 列互异（d≥3）

## 验证成员

| m | seed | [[n,k,d]] | 备注 |
|---|---|---|---|
| 1 | 12345 | [[25,13,2]] | validation_only（K_{4,3}：rank=1，[[25,13,2]]，大 k 测例；不属 scaling 家族） |

## 规则 full_rank

| m | seed | offset | attempts | n | k | rank | d_cl | 量子 d(来源) | fingerprint |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 12345 | 0 | 5 | 100 | 4 | 6 | 2 | 2 (hgp_theorem_classical_sides(full-rank ⇒ d=d_classical)) | 26727d1ef8d6b48c… |
| 3 | 12345 | 0 | 3 | 225 | 9 | 9 | 2 | 2 (hgp_theorem_classical_sides(full-rank ⇒ d=d_classical)) | 4336877b5f82802e… |
| 4 | 12345 | 0 | 21 | 400 | 16 | 12 | 6 | 6 (hgp_theorem_classical_sides(full-rank ⇒ d=d_classical)) | 5c1421fad3a42ba4… |
| 5 | 12345 | 0 | 52 | 625 | 25 | 15 | 4 | 4 (hgp_theorem_classical_sides(full-rank ⇒ d=d_classical)) | 3a28b5c2c6006443… |
| 6 | 12345 | 0 | 2 | 900 | 36 | 18 | 8 | 8 (hgp_theorem_classical_sides(full-rank ⇒ d=d_classical)) | 6335d4854c361c99… |

## 规则 full_rank_d3

| m | seed | offset | attempts | n | k | rank | d_cl | 量子 d(来源) | fingerprint |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 12349 | 4 | 20 | 100 | 4 | 6 | 4 | 4 (hgp_theorem_classical_sides(full-rank ⇒ d=d_classical)) | 39f72d508aacb9ac… |
| 3 | 12347 | 2 | 24 | 225 | 9 | 9 | 4 | 4 (hgp_theorem_classical_sides(full-rank ⇒ d=d_classical)) | 5c81fabf2b44ccc5… |
| 4 | 12345 | 0 | 21 | 400 | 16 | 12 | 6 | 6 (hgp_theorem_classical_sides(full-rank ⇒ d=d_classical)) | 5c1421fad3a42ba4… |
| 5 | 12345 | 0 | 52 | 625 | 25 | 15 | 4 | 4 (hgp_theorem_classical_sides(full-rank ⇒ d=d_classical)) | 3a28b5c2c6006443… |
| 6 | 12345 | 0 | 2 | 900 | 36 | 18 | 8 | 8 (hgp_theorem_classical_sides(full-rank ⇒ d=d_classical)) | 6335d4854c361c99… |
