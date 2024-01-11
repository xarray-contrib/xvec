import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
import xarray as xr
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_frame_equal

import xvec  # noqa: F401
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


def test_geom_coords(multi_geom_no_index_dataset, traffic_counts_array, geom_array):
    assert multi_geom_no_index_dataset.xvec._geom_coords_all == [
        "geom",
        "geom_z",
        "geom_no_ix",
    ]

    actual = multi_geom_no_index_dataset.xvec.geom_coords
    expected = multi_geom_no_index_dataset.coords
    assert actual.keys() == expected.keys()

    # check assignment
    with pytest.raises(AttributeError):
        multi_geom_no_index_dataset.xvec.geom_coords = (
            multi_geom_no_index_dataset.coords
        )

    actual = traffic_counts_array.xvec.geom_coords
    expected = xr.DataArray(
        coords={"origin": geom_array, "destination": geom_array},
        dims=("origin", "destination"),
    ).coords
    assert actual.keys() == expected.keys()


def test_geom_coords_indexed(multi_geom_dataset, traffic_counts_array, geom_array):
    assert multi_geom_dataset.xvec._geom_indexes == ["geom", "geom_z"]

    actual = multi_geom_dataset.xvec.geom_coords_indexed
    expected = multi_geom_dataset.coords
    assert actual.keys() == expected.keys()

    # check assignment
    with pytest.raises(AttributeError):
        multi_geom_dataset.xvec.geom_coords = multi_geom_dataset.coords

    actual = traffic_counts_array.xvec.geom_coords
    expected = xr.DataArray(
        coords={"origin": geom_array, "destination": geom_array},
        dims=("origin", "destination"),
    ).coords
    assert actual.keys() == expected.keys()


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
def test_is_geom_variable(
    multi_geom_one_ix_foo, label, has_index, expected, geom_array
):
    assert (
        multi_geom_one_ix_foo.xvec.is_geom_variable(label, has_index=has_index)
        == expected
    )

    # test array longer than 10 items
    arr = xr.DataArray(
        coords={
            "geom": np.repeat(geom_array, 10),
        },
        dims=["geom"],
    ).xvec.set_geom_indexes(["geom"], crs=26915)

    assert arr.xvec.is_geom_variable("geom")


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
        traffic_counts_array.sel(mode="car", day="2023-01-01").xvec.to_geopandas()

    expected = pd.DataFrame(
        {
            "destination": geom_array,
            0: [
                1.0,
                1.0,
            ],
        },
    )
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
        origin=shapely.Point(1, 2), day="2023-01-01"
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
        origin=shapely.Point(1, 2), day="2023-01-01", mode="car"
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
            "day": pd.to_datetime(["2023-01-01", "2023-01-01"]),
        }
    ).set_geometry("destination", crs=26915)
    expected["origin"] = gpd.GeoSeries(expected["origin"], crs=26915)

    actual = traffic_dataset.sel(
        origin=shapely.Point(1, 2), day="2023-01-01", mode="car"
    ).xvec.to_geopandas()

    assert_geodataframe_equal(expected, actual)

    expected = traffic_dataset.sel(
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
            day="2023-01-01",
            destination=shapely.Point(3, 4),
            origin=shapely.Point(1, 2),
        ).xvec.to_geopandas()

    assert_frame_equal(expected, actual)
    assert actual.origin.array.crs == 26915
    assert actual.destination.array.crs == 26915


def test_to_geodataframe_array(
    traffic_counts_array, traffic_counts_array_named, geom_array
):
    expected = pd.DataFrame(
        {
            "origin": geom_array.take([0, 0, 1, 1]),
            "destination": geom_array.take([0, 1, 0, 1]),
            "mode": ["car"] * 4,
            "day": pd.to_datetime(["2023-01-01"] * 4),
            "traffic_counts": [1.0] * 4,
        }
    )
    expected["origin"] = gpd.array.GeometryArray(expected["origin"].values, crs=26915)
    expected["destination"] = gpd.array.GeometryArray(
        expected["destination"].values, crs=26915
    )
    with pytest.warns(UserWarning, match="No active geometry"):
        actual = traffic_counts_array_named.sel(
            mode="car", day="2023-01-01"
        ).xvec.to_geodataframe()

    assert_frame_equal(expected, actual)

    expected = pd.DataFrame(
        {
            "origin": geom_array,
            "mode": ["car"] * 2,
            "day": pd.to_datetime(["2023-01-01"] * 2),
            "destination": geom_array.take([0, 0]),
            "traffic_counts": [1.0] * 2,
        }
    ).set_geometry("origin", crs=26915)
    expected["destination"] = gpd.GeoSeries(expected["destination"], crs=26915)

    actual = traffic_counts_array_named.sel(
        mode="car", day="2023-01-01", destination=geom_array[0]
    ).xvec.to_geodataframe()

    assert_geodataframe_equal(expected, actual)

    # unnamed
    actual = traffic_counts_array.sel(
        mode="car", day="2023-01-01", destination=geom_array[0]
    ).xvec.to_geodataframe(name="traffic_counts")

    assert_geodataframe_equal(expected, actual)

    with pytest.warns(UserWarning, match="No active geometry"):
        reordered = traffic_counts_array_named.xvec.to_geodataframe(
            dim_order=["day", "mode", "origin", "destination"]
        )
    assert reordered.index.names == ["day", "mode"]
    assert (reordered.columns == ["origin", "destination", "traffic_counts"]).all()


def test_to_geodataframe_wide(geom_array, traffic_counts_array_named):
    # DataArray
    arr = xr.DataArray(
        np.ones((3, 2)),
        coords={
            "mode": ["car", "bike", "walk"],
            "origin": geom_array,
        },
        name="traffic_counts",
    ).xvec.set_geom_indexes(
        [
            "origin",
        ],
        crs=26915,
    )

    expected = pd.DataFrame(
        columns=pd.MultiIndex.from_product(
            [["traffic_counts"], ["car", "bike", "walk"]], names=["mode", ""]
        ),
        index=range(2),
    ).fillna(1.0)
    expected["origin"] = geom_array
    expected = expected.set_geometry("origin", crs=26915)

    actual = arr.xvec.to_geodataframe(long=False)
    assert_geodataframe_equal(expected, actual, check_like=True)

    # Dataset
    count = np.ones((3, 2))
    time = np.ones((3, 2))

    ds = xr.Dataset(
        {
            "count": (
                [
                    "mode",
                    "origin",
                ],
                count,
            ),
            "time": (["mode", "origin"], time),
        },
        coords={
            "mode": ["car", "bike", "walk"],
            "origin": geom_array,
        },
    ).xvec.set_geom_indexes(["origin"], crs=26915)

    expected = pd.DataFrame(
        columns=pd.MultiIndex.from_product(
            [["count", "time"], ["car", "bike", "walk"]], names=["mode", ""]
        ),
        index=range(2),
    ).fillna(1.0)
    expected["origin"] = geom_array
    expected = expected.set_geometry("origin", crs=26915)

    actual = ds.xvec.to_geodataframe(long=False)
    assert_geodataframe_equal(expected, actual, check_like=True, check_crs=False)

    with pytest.raises(ValueError, match="Creating a wide form"):
        traffic_counts_array_named.xvec.to_geodataframe(long=False)


def test_to_geodataframe_dataset(traffic_dataset):
    with pytest.warns(UserWarning, match="No active geometry"):
        actual = traffic_dataset.xvec.to_geodataframe()
    assert actual.origin.values.crs == 26915
    assert actual.destination.values.crs == 26915

    actual = traffic_dataset.xvec.to_geodataframe(
        dim_order=["origin", "mode", "day", "destination"],
        geometry="destination",
    )
    assert actual.geometry.name == "destination"
    assert actual.crs == 26915
    assert actual.origin.crs == 26915

    actual = traffic_dataset.drop_vars("origin").xvec.to_geodataframe()
    assert actual.geometry.name == "destination"
    assert actual.crs == 26915


def test_extract_points_array():
    da = xr.DataArray(
        np.ones((10, 10, 5)),
        coords={
            "x": range(10),
            "y": range(20, 30),
            "time": pd.date_range("2023-01-01", periods=5),
        },
    )

    points = shapely.points([0, 3, 6], [12, 14, 11])
    expected = xr.DataArray(
        np.ones((3, 5)),
        coords={
            "geometry": points,
            "time": pd.date_range("2023-01-01", periods=5),
        },
    ).xvec.set_geom_indexes("geometry")
    actual = da.xvec.extract_points(points, "x", "y")

    xr.testing.assert_identical(actual, expected)

    # manual CRS
    actual = da.xvec.extract_points(points, "x", "y", crs=4326)
    xr.testing.assert_identical(actual, expected.xvec.set_crs(geometry=4326))

    # CRS inferred from GeoSeries
    gs = gpd.GeoSeries(points, crs=4326)
    actual = da.xvec.extract_points(gs, "x", "y")
    xr.testing.assert_identical(actual, expected.xvec.set_crs(geometry=4326))

    # CRS inferred from DataArray
    pts_arr = xr.DataArray(points, attrs={"crs": 4326})
    actual = da.xvec.extract_points(pts_arr, "x", "y")
    xr.testing.assert_identical(actual, expected.xvec.set_crs(geometry=4326))

    # custom name
    actual = da.xvec.extract_points(points, "x", "y", name="location")
    xr.testing.assert_identical(actual, expected.rename(geometry="location"))

    # retain index
    actual = da.xvec.extract_points(gs, "x", "y", index=True)
    xr.testing.assert_identical(
        actual,
        expected.assign_coords({"index": ("geometry", range(3))}).xvec.set_crs(
            geometry=4326
        ),
    )

    # retain named index
    gs.index.name = "my_index"
    actual = da.xvec.extract_points(
        gs,
        "x",
        "y",
    )
    xr.testing.assert_identical(
        actual,
        expected.assign_coords({"my_index": ("geometry", range(3))}).xvec.set_crs(
            geometry=4326
        ),
    )

    # retain non-default index
    gs2 = gpd.GeoSeries(points, crs=4326, index=range(3, 6))
    actual = da.xvec.extract_points(
        gs2,
        "x",
        "y",
    )
    xr.testing.assert_identical(
        actual,
        expected.assign_coords({"index": ("geometry", range(3, 6))}).xvec.set_crs(
            geometry=4326
        ),
    )

    # retain additional coords of points DataArray
    actual = da.xvec.extract_points(
        actual.geometry,
        "x",
        "y",
    )
    xr.testing.assert_identical(
        actual,
        expected.assign_coords({"index": ("geometry", range(3, 6))}).xvec.set_crs(
            geometry=4326
        ),
    )
