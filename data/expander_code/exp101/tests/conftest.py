import sys
from pathlib import Path

EXP101_ROOT = Path(__file__).resolve().parents[1]
if str(EXP101_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP101_ROOT))
