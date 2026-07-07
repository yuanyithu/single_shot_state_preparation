"""G1.8 证据脚本：构建官方 (3,4) 家族 m=1..6 双规则注册表并落盘。

输出：本目录 family_registry.json（本地，.gitignore 策略）与 family_registry.md（可提交）。
"""

import json
import sys
import time
from pathlib import Path

EXP101_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP101_ROOT))

from src.families import build_family_registry, registry_markdown  # noqa: E402


def main():
    started = time.perf_counter()
    registry = build_family_registry(m_list=(2, 3, 4, 5, 6), build_fingerprint=True)
    registry["wall_time_seconds"] = round(time.perf_counter() - started, 2)
    out_dir = Path(__file__).resolve().parent
    with (out_dir / "family_registry.json").open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2, ensure_ascii=False)
    (out_dir / "family_registry.md").write_text(
        registry_markdown(registry), encoding="utf-8"
    )
    print(registry_markdown(registry))
    print(f"wall_time: {registry['wall_time_seconds']}s")


if __name__ == "__main__":
    main()
