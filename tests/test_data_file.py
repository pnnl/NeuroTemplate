"""`data.yaml`: the record and its roles, once; and the columns no role claims."""

import pandas as pd
import pytest
import yaml

from neurotemplate.common.data_file import check_roles, load_data_file, nm_mapping, unclaimed
from neurotemplate.common.data_utils import read_timestamped_csv

RECORD = ["Zone", "Heat", "Setpoint", "Outdoor", "Tariff", "sec_elapsed"]


def _write(tmp_path, columns=None, **over):
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump({"train": "data/train.csv", "dev": ["d1.csv", "d2.csv"],
                                    "test": "t.csv", "columns": columns or {}, **over}))
    return load_data_file(path)


def test_a_split_is_a_path_or_a_list_read_from_beside_the_file(tmp_path):
    data = _write(tmp_path, {"Y": ["Zone"]})
    assert data.train == [str(tmp_path / "data/train.csv")]
    assert data.dev == [str(tmp_path / "d1.csv"), str(tmp_path / "d2.csv")]
    assert data.columns == {"Y": ["Zone"], "U": [], "D": []}


def test_with_no_roles_it_lists_the_columns_then_what_to_do(tmp_path):
    with pytest.raises(SystemExit, match="assigns no columns") as stop:
        nm_mapping(_write(tmp_path), RECORD)
    message = str(stop.value)
    lines = message.splitlines()
    # Every column, as a bare line, before anything else.
    assert [f"  {c}" for c in RECORD[:-1]] == lines[2:2 + len(RECORD) - 1]
    # The instructions are last, below the list.
    assert lines[-1] == "A column you leave out is simply not read."


def test_a_column_no_role_claims_is_allowed_and_listed(tmp_path):
    data = _write(tmp_path, {"Y": ["Zone"], "U": ["Setpoint"], "D": ["Outdoor"]})
    mapping = nm_mapping(data, RECORD)
    assert mapping["t"] == ["sec_elapsed"]
    assert unclaimed(mapping, RECORD) == ["Heat", "Tariff"]


def test_roles_are_checked_against_the_record_all_at_once(tmp_path):
    with pytest.raises(ValueError, match="(?s)not in the record: .'Nowhere'.*more than one role"):
        nm_mapping(_write(tmp_path, {"Y": ["Zone", "Nowhere"], "D": ["Zone"]}), RECORD)


def test_a_gap_in_a_column_no_one_reads_is_not_a_failure():
    from neurotemplate.common.data_utils import check_complete
    frame = pd.DataFrame({"Zone": [1.0, 2.0], "Tariff": [0.1, None]})
    check_complete([frame], ["Zone"], "train")
    with pytest.raises(ValueError, match="Tariff"):
        check_complete([frame], ["Zone", "Tariff"], "train")


def _record(tmp_path, header: str) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data/train.csv").write_text(f"{header}\n2025-01-01 00:00,1,2,3,4,5\n")


def test_the_header_check_sees_the_columns_the_loader_does(tmp_path):
    """`nt model train` checks the roles from the header before its imports; the
    columns it lists are the ones the full load would."""
    _record(tmp_path, "Timestamp,Zone,Heat,Setpoint,Outdoor,Tariff")
    with pytest.raises(SystemExit) as early:
        check_roles(_write(tmp_path).path)
    loaded = list(read_timestamped_csv(str(tmp_path / "data/train.csv")).columns)
    with pytest.raises(SystemExit) as full:
        nm_mapping(_write(tmp_path), [*loaded, "sec_elapsed"])
    assert str(early.value) == str(full.value)


def test_the_header_check_names_a_misspelled_column(tmp_path):
    _record(tmp_path, "time,Zone,Heat,Setpoint,Outdoor,Tariff")
    with pytest.raises(ValueError, match="Zoen"):
        check_roles(_write(tmp_path, {"Y": ["Zoen"]}).path)
    check_roles(_write(tmp_path, {"Y": ["Zone"], "U": ["Heat"]}).path)


@pytest.mark.parametrize("header", ["Zone,Heat,Setpoint,Outdoor,Tariff,x",  # no time
                                    "time,Zone,Zone,Setpoint,Outdoor,Tariff",  # a repeat
                                    ",time,Zone,Heat,Outdoor,Tariff"])  # a blank
def test_the_header_check_leaves_a_doubtful_header_to_the_full_load(tmp_path, header):
    _record(tmp_path, header)
    check_roles(_write(tmp_path, {"Y": ["Nowhere"]}).path)  # no verdict: the load gives it
