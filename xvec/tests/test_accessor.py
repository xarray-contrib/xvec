import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
import xarray as xr
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_frame_equal

import xvec  # noqa
from xvec import GeometryIndex


@pytest.fixture()
def geom_array_4326():
    return shapely.from_wkt(
        ["POINT (-97.488735 0.000018)", "POINT (-97.488717 0.000036)"]
    )


@pytest.fixture()
def geom_array_z_4326():
    return shapely.from_wkt(
        ["POINT Z (-97.488654 0.00018 30)", "POINT Z (-97.488475 0.000361 50)"]
    )


@pytest.fixture()
def mixed_geom_dims_dataset(geom_array, geom_array_z):
    return (
        xr.Dataset(coords={"geom": np.concatenate([geom_array, geom_array_z])})
        .drop_indexes("geom")
        .set_xindex("geom", GeometryIndex, crs=26915)
    )


@pytest.fixture()
def geom_array_mixed_4326(geom_array_4326, geom_array_z_4326):
    return np.concatenate([geom_array_4326, geom_array_z_4326])


# Test .xvec accessor


def test_accessor(multi_geom_dataset):
    assert hasattr(multi_geom_dataset, "xvec")
    xr.testing.assert_identical(multi_geom_dataset.xvec._obj, multi_geom_dataset)


# Test .xvec geom_coords


def test_geom_coords(multi_geom_no_index_dataset):
    assert multi_geom_no_index_dataset.xvec._geom_coords_all == [
        "geom",
        "geom_z",
        "geom_no_ix",
    ]

    actual = multi_geom_no_index_dataset.xvec.geom_coords
    expected = multi_geom_no_index_dataset.coords
    actual.keys() == expected.keys()

    # check assignment
    with pytest.raises(AttributeError):
        multi_geom_no_index_dataset.xvec.geom_coords = (
            multi_geom_no_index_dataset.coords
        )


def test_geom_coords_indexed(multi_geom_no_index_dataset):
    assert multi_geom_no_index_dataset.xvec._geom_indexes == ["geom", "geom_z"]

    actual = multi_geom_no_index_dataset.xvec.geom_coords_indexed
    expected = multi_geom_no_index_dataset.coords
    actual.keys() == expected.keys()

    # check assignment
    with pytest.raises(AttributeError):
        multi_geom_no_index_dataset.xvec.geom_coords = (
            multi_geom_no_index_dataset.coords
        )


# Test .xvec.is_geom_variable
@pytest.mark.parametrize(
    "label,has_index,expected",
    [
        ("geom", True, True),
        ("geom", False, True),
        ("geom2", True, False),
        ("geom2", False, True),
        ("foo", True, False),
        ("foo", False, False),
    ],
)
def test_is_geom_variable(multi_geom_one_ix_foo, label, has_index, expected):
    assert (
        multi_geom_one_ix_foo.xvec.is_geom_variable(label, has_index=has_index)
        == expected
    )


# Test .xvec.to_crs


def test_to_crs(geom_dataset, geom_array_4326):
    transformed = geom_dataset.xvec.to_crs(geom=4326)
    assert shapely.equals_exact(geom_array_4326, transformed.geom, tolerance=1e-5).all()
    assert transformed.xindexes["geom"].crs.equals(4326)


def test_to_crs_two_geoms(multi_geom_dataset, geom_array_4326, geom_array_z_4326):
    transformed = multi_geom_dataset.xvec.to_crs(geom=4326, geom_z=4326)
    assert shapely.equals_exact(geom_array_4326, transformed.geom, tolerance=1e-5).all()
    assert shapely.equals_exact(
        geom_array_z_4326, transformed.geom_z, tolerance=1e-5
    ).all()
    assert transformed.xindexes["geom"].crs.equals(4326)
    assert transformed.xindexes["geom_z"].crs.equals(4326)


def test_to_crs_dict(multi_geom_dataset, geom_array_4326, geom_array_z_4326):
    transformed = multi_geom_dataset.xvec.to_crs({"geom": 4326, "geom_z": 4326})
    assert shapely.equals_exact(geom_array_4326, transformed.geom, tolerance=1e-5).all()
    assert shapely.equals_exact(
        geom_array_z_4326, transformed.geom_z, tolerance=1e-5
    ).all()
    assert transformed.xindexes["geom"].crs.equals(4326)
    assert transformed.xindexes["geom_z"].crs.equals(4326)


def test_to_crs_arg_err(multi_geom_dataset):
    with pytest.raises(ValueError, match="Cannot specify both keyword"):
        multi_geom_dataset.xvec.to_crs(geom=4326, variable_crs={"geom_z": 4326})


def test_to_crs_naive(geom_dataset_no_crs):
    with pytest.raises(ValueError, match="Cannot transform naive geometries."):
        geom_dataset_no_crs.xvec.to_crs(geom=4326)


def test_to_crs_same(geom_dataset):
    transformed = geom_dataset.xvec.to_crs(geom=26915)
    xr.testing.assert_identical(transformed, geom_dataset)


def test_to_crs_mixed_geom_dims(mixed_geom_dims_dataset, geom_array_mixed_4326):
    transformed = mixed_geom_dims_dataset.xvec.to_crs(geom=4326)
    assert shapely.equals_exact(
        geom_array_mixed_4326, transformed.geom, tolerance=1e-5
    ).all()
    assert transformed.xindexes["geom"].crs.equals(4326)


def test_to_crs_no_GeometryIndex(dataset_w_geom):
    with pytest.raises(
        ValueError, match="The index 'geom' is not an xvec.GeometryIndex."
    ):
        dataset_w_geom.xvec.to_crs(geom=4326)


# Test .xvec.set_crs


def test_set_crs(geom_dataset_no_crs):
    with_crs = geom_dataset_no_crs.xvec.set_crs(geom=4326)
    assert with_crs.xindexes["geom"].crs.equals(4326)


def test_set_crs_dict(geom_dataset_no_crs):
    with_crs = geom_dataset_no_crs.xvec.set_crs({"geom": 4326})
    assert with_crs.xindexes["geom"].crs.equals(4326)


def test_set_crs_arg_err(multi_geom_dataset):
    with pytest.raises(ValueError, match="Cannot specify both keyword"):
        multi_geom_dataset.xvec.set_crs(geom=4326, variable_crs={"geom_z": 4326})


def test_set_crs_mismatch(geom_dataset):
    with pytest.raises(ValueError, match="The index 'geom' already has a CRS"):
        geom_dataset.xvec.set_crs(geom=4326)


def test_set_crs_override(geom_dataset, geom_array):
    with_crs = geom_dataset.xvec.set_crs(allow_override=True, geom=4326)
    # new CRS
    assert with_crs.xindexes["geom"].crs.equals(4326)
    # same geometries
    assert shapely.equals_exact(
        geom_array,
        with_crs.geom,
    ).all()


def test_set_crs_no_GeometryIndex(dataset_w_geom):
    with pytest.raises(
        ValueError, match="The index 'geom' is not an xvec.GeometryIndex."
    ):
        dataset_w_geom.xvec.set_crs(geom=4326)


def test_set_crs_multiple(multi_geom_dataset):
    with_crs = multi_geom_dataset.xvec.to_crs(geom=4326, geom_z=4326)
    assert with_crs.xindexes["geom"].crs.equals(4326)
    assert with_crs.xindexes["geom_z"].crs.equals(4326)


# Test .xvec.set_geom_indexes


def test_set_geom_indexes(dataset_w_geom):
    result = dataset_w_geom.xvec.set_geom_indexes("geom")
    assert "geom" in result.xvec.geom_coords_indexed
    assert result.xindexes["geom"].crs is None


def test_set_geom_indexes_crs(dataset_w_geom):
    result = dataset_w_geom.xvec.set_geom_indexes("geom", crs=4326)
    assert "geom" in result.xvec.geom_coords_indexed
    assert result.xindexes["geom"].crs.equals(4326)


def test_set_geom_indexes_multi(multi_dataset):
    result = multi_dataset.xvec.set_geom_indexes(["geom", "geom_z"], crs=4326)
    assert "geom" in result.xvec.geom_coords_indexed
    assert "geom_z" in result.xvec.geom_coords_indexed
    assert result.xindexes["geom"].crs.equals(4326)
    assert result.xindexes["geom_z"].crs.equals(4326)


def test_set_geom_indexes_override(first_geom_dataset):
    result = first_geom_dataset.xvec.set_geom_indexes(
        "geom", crs=4326, allow_override=True
    )
    assert "geom" in result.xvec.geom_coords_indexed
    assert result.xindexes["geom"].crs.equals(4326)


def test_set_geom_indexes_mismatch(first_geom_dataset):
    with pytest.raises(ValueError, match="The index 'geom' already has a CRS"):
        first_geom_dataset.xvec.set_geom_indexes("geom", crs=4326, allow_override=False)


# Test .xvec.query


def test_query(multi_geom_dataset):
    expected = multi_geom_dataset.isel(geom=[0])
    actual = multi_geom_dataset.xvec.query("geom", shapely.box(0, 0, 2.4, 2.2))
    xr.testing.assert_identical(expected, actual)


@pytest.mark.parametrize(
    "predicate,expected",
    [
        (None, [0]),
        ("intersects", [0]),
        ("within", []),
        ("contains", [0]),
        ("overlaps", []),
        ("crosses", []),
        ("touches", []),
        ("covers", [0]),
        ("covered_by", []),
        ("contains_properly", [0]),
    ],
)
def test_query_predicate(multi_geom_dataset, predicate, expected):
    expected = multi_geom_dataset.isel(geom=expected)
    actual = multi_geom_dataset.xvec.query(
        "geom", shapely.box(0, 0, 2.4, 2.2), predicate=predicate
    )
    xr.testing.assert_identical(expected, actual)


@pytest.mark.parametrize(
    "distance,expected",
    [
        (10, [0, 1]),
        (0.1, [0]),
    ],
)
def test_query_dwithin(multi_geom_dataset, distance, expected):
    expected = multi_geom_dataset.isel(geom=expected)
    actual = multi_geom_dataset.xvec.query(
        "geom", shapely.box(0, 0, 2.4, 2.2), predicate="dwithin", distance=distance
    )
    xr.testing.assert_identical(expected, actual)


@pytest.mark.parametrize(
    "unique,expected",
    [
        (False, [0, 0, 0]),
        (True, [0]),
    ],
)
def test_query_array(multi_geom_dataset, unique, expected):
    expected = multi_geom_dataset.isel(geom=expected)
    actual = multi_geom_dataset.xvec.query(
        "geom", [shapely.box(0, 0, 2.4, 2.2)] * 3, unique=unique
    )
    xr.testing.assert_identical(expected, actual)


def test_to_geopandas_array(traffic_counts_array, geom_array):
    with pytest.raises(ValueError, match="Cannot convert arrays"):
        traffic_counts_array.xvec.to_geopandas()

    with pytest.raises(ValueError, match="Multiple coordinates based on"):
        traffic_counts_array.sel(
            mode="car", hour=1, day="2023-01-01"
        ).xvec.to_geopandas()

    expected = pd.DataFrame(
        {
            "destination": geom_array,
            0: [
                1.0,
                1.0,
            ],
            1: [
                1.0,
                1.0,
            ],
        },
    )
    expected.columns.name = "hour"
    expected = expected.set_geometry("destination", crs=26915)

    # transposition needed
    actual = traffic_counts_array.sel(
        mode="car", day="2023-01-01", origin=shapely.Point(1, 2)
    ).xvec.to_geopandas()

    assert_geodataframe_equal(expected, actual)

    expected = pd.DataFrame(
        {
            "destination": geom_array,
            "car": [
                1.0,
                1.0,
            ],
            "bike": [
                1.0,
                1.0,
            ],
            "walk": [
                1.0,
                1.0,
            ],
        },
    )
    expected.columns.name = "mode"
    expected = expected.set_geometry("destination", crs=26915)
    actual = traffic_counts_array.sel(
        origin=shapely.Point(1, 2), hour=0, day="2023-01-01"
    ).xvec.to_geopandas()

    assert_geodataframe_equal(expected, actual)

    # upcast Series
    expected = pd.DataFrame(
        {
            "destination": geom_array,
            0: [
                1.0,
                1.0,
            ],
        },
    ).set_geometry("destination", crs=26915)

    actual = traffic_counts_array.sel(
        origin=shapely.Point(1, 2), hour=0, day="2023-01-01", mode="car"
    ).xvec.to_geopandas()

    assert_geodataframe_equal(expected, actual)

    with pytest.warns(UserWarning, match="No geometry"):
        traffic_counts_array.sel(
            destination=geom_array[0],
            origin=geom_array[0],
            mode="car",
            day="2023-01-01",
        ).xvec.to_geopandas()


def test_to_geopandas_dataset(traffic_dataset, geom_array):
    expected = pd.DataFrame(
        {
            "destination": geom_array,
            "count": [1.0, 1.0],
            "time": [1.0, 1.0],
            "mode": ["car", "car"],
            "origin": [geom_array[0], geom_array[0]],
            "hour": [0, 0],
            "day": pd.to_datetime(["2023-01-01", "2023-01-01"]),
        }
    ).set_geometry("destination", crs=26915)
    expected["origin"] = gpd.GeoSeries(expected["origin"], crs=26915)

    actual = traffic_dataset.sel(
        origin=shapely.Point(1, 2), hour=0, day="2023-01-01", mode="car"
    ).xvec.to_geopandas()

    assert_geodataframe_equal(expected, actual)

    expected = traffic_dataset.sel(
        hour=0,
        day="2023-01-01",
        destination=shapely.Point(3, 4),
        origin=shapely.Point(1, 2),
    ).to_pandas()
    expected["origin"] = gpd.array.GeometryArray(expected["origin"].values, crs=26915)
    expected["destination"] = gpd.array.GeometryArray(
        expected["destination"].values, crs=26915
    )

    with pytest.warns(UserWarning, match="No active geometry column to be set"):
        actual = traffic_dataset.sel(
            hour=0,
            day="2023-01-01",
            destination=shapely.Point(3, 4),
            origin=shapely.Point(1, 2),
        ).xvec.to_geopandas()

    assert_frame_equal(expected, actual)
    assert actual.origin.array.crs == 26915
    assert actual.destination.array.crs == 26915
