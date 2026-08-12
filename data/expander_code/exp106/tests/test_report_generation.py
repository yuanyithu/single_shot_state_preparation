"""The report generator must survive contact with a real aggregate.

exp105 shipped a `report.py` with three defects that only surface at report time,
and one of them -- an undefined `ms` in `write_contrasts` -- killed its **remote**
aggregate stage after the production NPZ had already been written. Nothing in its
suite ever called these functions, so nothing caught it, and the fix could not
then be applied in place without orphaning the published aggregate.

exp106 ships the corrected generator, and this file is what keeps it correct:
`run_remote_aggregate` calls `write_report` at the end of a nine-hour scan, which
is a bad place to discover a NameError.
"""

import csv
import json
from pathlib import Path

import pytest

from data.expander_code.exp106.exp106_pipeline.report import (
    p_axis_scale,
    panel_counts,
    panel_trials,
    write_report,
)


@pytest.fixture(scope="module")
def written(tmp_path_factory, complete_aggregate_factory, pilot_config):
    """Generate the whole publication set once; plotting is the slow part."""
    payload = complete_aggregate_factory()
    directory = tmp_path_factory.mktemp("final_results")
    report = write_report(directory, payload, pilot_config)
    return directory, report, payload


def test_every_promised_artifact_is_written(written):
    directory, report, _ = written
    expected = {
        "primary_curves.csv", "crossing_contrasts.csv", "distance_strata.csv",
        "ensemble_composition.csv", "code_diagnostics.csv",
        "primary_crossing.png", "distance_strata.png", "report.md",
    }
    digests = report["file_sha256"]
    assert expected <= set(digests)
    for name in expected:
        path = Path(directory) / name
        assert path.is_file() and path.stat().st_size > 0, name
        # every entry is a SHA256 of the file it names
        assert len(digests[name]) == 64
    assert (Path(directory) / "report.json").is_file()


def test_p_values_are_not_collapsed_by_formatting(written):
    """exp105's `%.2f` mapped 0.001, 0.0015 and 0.0025 to the same string.

    Any grid whose points differ below the second decimal would silently become
    duplicate rows -- and exp106's bracket branch produces exactly such a grid,
    ten uniform points inside a window a few thousandths wide.
    """
    directory, _, payload = written
    with open(Path(directory) / "primary_curves.csv", encoding="ascii") as handle:
        rows = list(csv.DictReader(handle))
    formatted = {row["p"] for row in rows}
    assert len(formatted) == len(payload["p_values"]), (
        "distinct grid points must stay distinct after formatting"
    )


def test_the_contrast_table_covers_every_adjacent_pair(written):
    """The regression for the undefined `ms`: this function is simply run."""
    directory, _, payload = written
    with open(Path(directory) / "crossing_contrasts.csv", encoding="ascii") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        rows = list(reader)
    ms = payload["m_values"].tolist()
    for index in range(len(ms) - 1):
        assert f"delta{ms[index]}{ms[index + 1]}" in header
    assert len(rows) == len(payload["p_values"])
    for row in rows:
        assert row["certification"] in ("", "certified_negative", "certified_positive")


def test_the_report_json_carries_q_and_the_bound(written):
    directory, _, payload = written
    report = json.loads((Path(directory) / "report.json").read_text(encoding="ascii"))
    assert report["schema_version"] == "exp106.report.v1"
    assert report["q_token"] == "0.01"
    bound = json.loads(report["qtop_lower_bound_json"])
    assert set(bound) == {str(m) for m in payload["m_values"].tolist()}


def test_the_scope_paragraph_states_the_real_q(written):
    """exp105's frozen copy hard-coded "at q=0", which was wrong for exp105 too."""
    directory, _, _ = written
    text = (Path(directory) / "report.md").read_text(encoding="ascii")
    assert "q = 0.01" in text
    assert "q=0" not in text
    assert "Clears no exp102 blocker" in text


def test_panel_helpers_read_the_aggregate_not_the_config(written):
    """A published aggregate has to be readable without its config."""
    _, _, payload = written
    counts = panel_counts(payload)
    trials = panel_trials(payload)
    assert set(counts) == set(payload["m_values"].tolist())
    assert set(trials) == set(counts)
    assert all(value > 0 for value in counts.values())


@pytest.mark.parametrize("values,expected", [
    ([0.001, 0.0015, 0.07], "log"),
    ([0.02, 0.0228, 0.045], "linear"),
    ([0.005, 0.07], "log"),
    ([], "linear"),
])
def test_the_x_axis_scale_follows_the_grid(values, expected):
    """Log only when the grid spans a decade.

    exp105's grid ran 0.001 to 0.07 and needed a log axis. exp106's bracket
    branch returns ten uniform points inside a narrow window, where a log axis
    would compress exactly the region the experiment exists to show.
    """
    assert p_axis_scale(values) == expected
