"""Fresh v2 wrapper around the independently replaying direct-block analyzer."""

from __future__ import annotations

from importlib import import_module


_workflow = import_module(
    "data.expander_code.exp102.validation."
    "056_q0_random_full_column_direct_block_t1_m8_v2_20260724.workflow"
)
_base = import_module(
    "data.expander_code.exp102.validation."
    "055_q0_random_full_column_direct_block_t1_m8_20260724.analyze_t1"
)

CONTRACT_VERSION = _workflow.CONTRACT_VERSION
FAMILIES = _workflow.FAMILIES
NODE_REPORT_VERSION = _workflow.NODE_REPORT_VERSION
RAW_VERSION = _workflow.RAW_VERSION
REPORT_VERSION = _workflow.REPORT_VERSION


def _configure_base():
    bindings = {
        "_workflow": _workflow,
        "CONTRACT_VERSION": CONTRACT_VERSION,
        "FAMILIES": FAMILIES,
        "NODE_REPORT_VERSION": NODE_REPORT_VERSION,
        "RAW_VERSION": RAW_VERSION,
        "REPORT_VERSION": REPORT_VERSION,
    }
    for name, value in bindings.items():
        setattr(_base, name, value)


def main():
    _configure_base()
    _base.main()


if __name__ == "__main__":
    main()
