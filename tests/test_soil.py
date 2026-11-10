import numpy as np
import pandas as pd
import pytest

from eusoil.ingest.soil import caco3_from_codes, log1p_safe, parse_measurement, read_soil


def test_lod_codes_with_one_or_two_spaces_are_censored():
    parsed = parse_measurement(pd.Series(["< LOD", "<  LOD", "12.5"]), lod=10.0)
    assert parsed["below_lod"].tolist() == [True, True, False]
    assert np.isnan(parsed["value"].iloc[0])
    assert parsed["value"].iloc[2] == 12.5


def test_numbers_below_the_limit_are_censored_too():
    parsed = parse_measurement(pd.Series(["0.3", "9.99", "10.0"]), lod=10.0)
    assert parsed["below_lod"].tolist() == [True, True, False]
    assert parsed["value"].iloc[2] == 10.0


def test_negative_zero_code_is_invalid_not_a_value():
    parsed = parse_measurement(pd.Series(["<0.0", "5"]), lod=2.0)
    assert parsed["invalid"].tolist() == [True, False]
    assert np.isnan(parsed["value"].iloc[0])
    assert not parsed["below_lod"].iloc[0]


def test_missing_cells_stay_missing_and_unflagged():
    parsed = parse_measurement(pd.Series([None, "3"]), lod=2.0)
    assert np.isnan(parsed["value"].iloc[0])
    assert not parsed["below_lod"].iloc[0] and not parsed["invalid"].iloc[0]


def test_unknown_code_raises():
    with pytest.raises(ValueError, match="Unrecognised"):
        parse_measurement(pd.Series(["n.d."]), lod=1.0)


def test_caco3_codes_mean_absent():
    decoded = caco3_from_codes(pd.Series([None, "< LOD", "<  LOD", "35"]))
    assert decoded["value"].tolist() == [0.0, 0.0, 0.0, 35.0]
    assert decoded["present"].tolist() == [False, False, False, True]


def test_log1p_rejects_negative_values():
    with pytest.raises(ValueError):
        log1p_safe(pd.Series([1.0, -0.5]))


def test_read_soil_normalises_codes_and_drops_unused_columns(source_files):
    df = read_soil(source_files.soil)
    assert "BX1" in set(df["lc1"])
    assert "Elev" not in df.columns and "elev" not in df.columns
    assert df["point_id"].dtype == "int64"


def test_read_soil_rejects_duplicate_points(source_files):
    text = source_files.soil.read_text().splitlines()
    source_files.soil.write_text("\n".join(text + [text[1]]) + "\n")
    with pytest.raises(ValueError, match="not unique"):
        read_soil(source_files.soil)
