"""`data.yaml`: the record and its roles, once; and the columns no role claims."""

import pandas as pd
import pytest
import yaml

from neurotemplate.common.data_file import load_data_file, nm_mapping, unclaimed

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


def test_with_no_roles_it_stops_and_lists_every_column_ready_to_paste(tmp_path):
    with pytest.raises(SystemExit, match="assigns no columns") as stop:
        nm_mapping(_write(tmp_path), RECORD)
    pasted = yaml.safe_load(str(stop.value).split("\n\n")[-1])
    assert pasted == {"columns": {"Y": RECORD[:-1], "U": [], "D": []}}


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
