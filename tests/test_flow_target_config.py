import pandas as pd
import pytest

from eusoil.config import load_config
from eusoil.flow import SampleFlow
from eusoil.ingest.landcover import signed_longitude
from eusoil.target import assign_target, classes_below_minimum, excluded_code_counts


def test_flow_counts_and_treats_missing_mask_as_drop():
    df = pd.DataFrame({"x": [1, 2, 3]})
    flow = SampleFlow()
    out = flow.apply(df, pd.Series([True, None, False], dtype="boolean"), "s", "r")
    assert len(out) == 1
    assert flow.to_frame().iloc[0][["n_in", "n_removed", "n_out"]].tolist() == [3, 2, 1]


def test_flow_rejects_a_misaligned_mask():
    with pytest.raises(ValueError):
        SampleFlow().apply(
            pd.DataFrame({"x": [1, 2]}), pd.Series([True, True], index=[5, 6]), "s", "r"
        )


def test_config_groups_are_complete_and_disjoint():
    target = load_config()["target"]
    level2 = set(target["level2"])
    assert level2 == {g for gs in target["level1"].values() for g in gs}
    assert level2 == {g for gs in target["input_intensity"].values() for g in gs}
    codes = [c for cs in target["level2"].values() for c in cs]
    assert len(codes) == len(set(codes))


def test_merged_groups_follow_the_decision_of_10_november():
    level2 = load_config()["target"]["level2"]
    assert {"B21", "B22", "B23", "B16"} <= set(level2["maize_and_root_crops"])
    assert {"B82", "B71"} <= set(level2["vineyards_and_orchards"])
    assert "E30" not in {c for cs in level2.values() for c in cs}


def test_assign_target_maps_and_leaves_unknown_codes_missing():
    target = load_config()["target"]
    out = assign_target(pd.DataFrame({"lc1": ["B11", "E20", "B43"]}), target)
    assert out["crop_group"].tolist()[:2] == ["wheat", "permanent_grassland"]
    assert pd.isna(out["crop_group"].iloc[2])
    assert out["farming_system"].iloc[1] == "grassland"
    assert out["input_intensity"].iloc[1] == "low"


def test_duplicate_code_in_two_groups_is_rejected():
    target = load_config()["target"]
    target["level2"]["wheat"].append("B16")
    with pytest.raises(ValueError, match="both"):
        assign_target(pd.DataFrame({"lc1": ["B11"]}), target)


def test_classes_below_minimum():
    df = pd.DataFrame({"crop_group": ["a"] * 5 + ["b"] * 2})
    assert classes_below_minimum(df, 3).to_dict() == {"b": 2}


def test_signed_longitude_uses_the_west_flag():
    out = signed_longitude(pd.Series(["7.5", "8.0", "-3.0"]), pd.Series(["West", "East", "West"]))
    assert out.tolist() == [-7.5, 8.0, -3.0]


def test_excluded_code_counts_is_sorted_with_a_clean_index():
    df = pd.DataFrame(
        {
            "lc1": ["E30", "B43", "E30"],
            "lc1_desc": ["Re-vegetated", "Other", "Re-vegetated"],
            "crop_group": [None, None, None],
        }
    )
    out = excluded_code_counts(df)
    assert out["lc1"].tolist() == ["E30", "B43"]
    assert isinstance(out.index, pd.RangeIndex)  # written to CSV without an index column


def test_climate_block_is_terraclimate_normals_and_sensitivity_sets_match_it():
    cfg = load_config()
    block = cfg["blocks"]["C"]
    assert all(name.startswith("tc9120_bio") for name in block)
    numbers = [name.removeprefix("tc9120_bio") for name in block]
    for prefix in cfg["climate_sensitivity"].values():
        assert prefix != "tc9120_bio"
        swapped = [f"{prefix}{n}" for n in numbers]
        assert len(set(swapped)) == len(block)
