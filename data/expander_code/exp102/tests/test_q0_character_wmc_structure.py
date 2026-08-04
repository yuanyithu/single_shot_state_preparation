"""Small deterministic checks for the character-WMC structure probe."""

import importlib.util
from pathlib import Path
import time


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "validation/035_q0_character_wmc_structure_feasibility_20260724"
    / "probe_character_wmc_structure.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("character_wmc_structure_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_greedy_orders_finish_on_a_path_with_width_one():
    module = _module()
    scopes = ((0, 1), (1, 2), (2, 3))
    deadline = time.monotonic() + 2.0
    order_degree, width_degree, _ = module._min_degree(scopes, 4, deadline, 100)
    order_fill, width_fill, _ = module._min_fill(scopes, 4, deadline, 100)
    assert order_degree == (0, 1, 2, 3)
    assert order_fill == (0, 1, 2, 3)
    assert width_degree == width_fill == 1


def test_min_fill_tie_breaks_by_variable_index():
    module = _module()
    scopes = ((0, 1), (2, 3))
    order, width, _ = module._min_fill(scopes, 4, time.monotonic() + 2.0, 100)
    assert order == (0, 1, 2, 3)
    assert width == 1


def test_edge_cap_is_reported_instead_of_silently_continuing():
    module = _module()
    scopes = ((0, 1, 2), (1, 2, 3))
    try:
        module._min_degree(scopes, 4, time.monotonic() + 2.0, 0)
    except module.StructureResourceLimit:
        pass
    else:  # pragma: no cover - keeps the expected safety behavior explicit
        raise AssertionError("min-degree ignored the adjacency edge cap")
