import numpy as np
import pandas as pd
import pytest

from eusoil.build import add_projected_coordinates, build_modelling_table, build_points
from eusoil.clean import clean_soil
from eusoil.flow import SampleFlow
from eusoil.ingest.soil import read_soil


def test_clean_soil_filters_and_substitutes(source_files, config):
    flow = SampleFlow()
    cleaned = clean_soil(read_soil(source_files.soil), config["cleaning"], flow)
    steps = flow.to_frame().set_index("step")
    assert steps.loc["standard depth", "n_removed"] == 1  # point 4, 0-10 cm
    assert steps.loc["excluded countries", "n_removed"] == 1  # point 5, MT
    p1 = cleaned.set_index("point_id").loc[1]
    assert p1["p_mg_kg_below_lod"] and p1["p_mg_kg"] == pytest.approx(5.0)  # 0.5 x 10
    p3 = cleaned.set_index("point_id").loc[3]
    assert p3["p_mg_kg_below_lod"] and p3["p_mg_kg"] == pytest.approx(5.0)  # 4.0 < LOD
    assert p3["caco3_present"] and p3["caco3_log1p"] == pytest.approx(np.log1p(120))


def test_build_points_applies_every_filter_and_counts_it(source_files, config):
    flow = SampleFlow()
    points, audit = build_points(source_files, config, flow)
    # Kept: 1 (B11), 2 (B16), 3 (E20). Removed: 4 depth, 5 MT, 6 woodland,
    # 7 land use U370, 8 E30 and 9 BX1 (no crop group).
    assert sorted(points["point_id"]) == [1, 2, 3]
    table = flow.to_frame().set_index("step")
    assert table.loc["agricultural land cover", "n_removed"] == 1
    assert table.loc["agricultural land use", "n_removed"] == 1
    assert table.loc["crop group", "n_removed"] == 2
    assert set(audit["excluded_codes"]["lc1"]) == {"E30", "BX1"}
    assert flow.to_frame()["n_out"].iloc[-1] == len(points)


def test_placeholder_gps_falls_back_to_theoretical(source_files, config):
    points, _ = build_points(source_files, config, SampleFlow())
    p2 = points.set_index("point_id").loc[2]
    assert p2["coord_source"] == "theoretical"
    assert p2["lat"] == pytest.approx(50.20)


def test_texture_only_where_particle_size_was_measured(source_files, config):
    points, _ = build_points(source_files, config, SampleFlow())
    assert np.isnan(points.set_index("point_id").loc[1, "clay"])  # PSDAvailable == No
    assert points.set_index("point_id").loc[2, "clay"] == 20


def test_land_cover_disagreement_stops_the_build(source_files, config):
    harmo = pd.read_csv(source_files.landcover_2018)
    harmo.loc[0, "lc1"] = "B13"
    harmo.to_csv(source_files.landcover_2018, index=False)
    with pytest.raises(ValueError, match="different LC1"):
        build_points(source_files, config, SampleFlow())


def test_missing_survey_rows_stop_the_build(source_files, config):
    harmo = pd.read_csv(source_files.landcover_2018)
    harmo.iloc[1:].to_csv(source_files.landcover_2018, index=False)
    with pytest.raises(ValueError, match="no row"):
        build_points(source_files, config, SampleFlow())


def test_class_below_minimum_stops_the_build(source_files, config):
    config["target"]["min_class_size"] = 2
    with pytest.raises(ValueError, match="below the minimum"):
        build_points(source_files, config, SampleFlow())


def test_projected_coordinates_are_in_laea_metres(config):
    frankfurt = pd.DataFrame({"lon": [8.68], "lat": [50.11]})
    out = add_projected_coordinates(frankfurt, config["crs"])
    # ETRS89-LAEA: Frankfurt lies near x = 4.21e6 m, y = 3.00e6 m.
    assert out["easting"].iloc[0] == pytest.approx(4.21e6, rel=0.01)
    assert out["northing"].iloc[0] == pytest.approx(3.00e6, rel=0.01)


def _table_inputs(config):
    points = pd.DataFrame(
        {"point_id": [1, 2], "lat": [50.0, 51.0], "lon": [8.0, 9.0], "nuts0": ["DE", "DE"]}
    )
    names = [c for cols in config["blocks"].values() for c in cols]
    values = pd.DataFrame(
        {
            "point_id": [1, 2],
            **{c: [0.0, 1.0] for c in names if c not in {"easting", "northing", "country"}},
        }
    )
    return points, values


def test_modelling_table_joins_extractions_and_checks_the_blocks(config):
    points, values = _table_inputs(config)
    table = build_modelling_table(points, [("all.parquet", values)], config)
    assert {"easting", "northing", "country", "tc9120_bio1"} <= set(table.columns)
    with pytest.raises(ValueError, match="missing"):
        build_modelling_table(points, [("x", values.drop(columns="tc9120_bio1"))], config)


def test_modelling_table_rejects_an_extraction_with_clashing_columns(config):
    points, values = _table_inputs(config)
    old = pd.DataFrame({"point_id": [1, 2], "tc9120_bio1": [5.0, 6.0]})
    with pytest.raises(ValueError, match="old.parquet repeats columns"):
        build_modelling_table(points, [("all.parquet", values), ("old.parquet", old)], config)
