import numpy as np
import pytest
import shapely
import xarray as xr
from pyproj import CRS

from xvec import GeoVectorIndex


@pytest.fixture(scope="session")
def geom_array():
    return np.array([shapely.Point(1, 2), shapely.Point(3, 4)])


@pytest.fixture(scope="session")
def geom_dataset_no_index(geom_array):
    # a dataset with a geometry coordinate but no index
    ds = xr.Dataset(coords={"geom": geom_array})
    return ds.drop_indexes("geom")


@pytest.fixture(scope="session")
def geom_dataset(geom_dataset_no_index):
    # a dataset with a geometry coordinate baked by a GeoVectorIndex
    crs = CRS.from_user_input(26915)
    return geom_dataset_no_index.set_xindex("geom", GeoVectorIndex, crs=crs)


@pytest.fixture(scope="session")
def first_geom_dataset(geom_dataset, geom_array):
    return (
        xr.Dataset(coords={"geom": [geom_array[0]]})
        .drop_indexes("geom")
        .set_xindex("geom", GeoVectorIndex, crs=geom_dataset.xindexes["geom"].crs)
    )


def test_set_index(geom_dataset_no_index):
    crs = CRS.from_user_input(26915)
    ds = geom_dataset_no_index.set_xindex("geom", GeoVectorIndex, crs=crs)

    # test properties
    assert isinstance(ds.xindexes["geom"], GeoVectorIndex)
    assert ds.xindexes["geom"].crs == crs
    np.testing.assert_array_equal(ds.xindexes["geom"].sindex.geometries, ds.geom.values)

    # test `GeoVectorIndex.create_variables`
    assert ds.geom.variable._data.array is ds.xindexes["geom"]._index.index

    with pytest.raises(ValueError, match="a CRS must be provided"):
        geom_dataset_no_index.set_xindex("geom", GeoVectorIndex)

    no_geom_ds = xr.Dataset(coords={"no_geom": ("x", [0, 1, 2])})
    with pytest.raises(ValueError, match="array must contain shapely.Geometry objects"):
        no_geom_ds.set_xindex("no_geom", GeoVectorIndex, crs=crs)


def test_concat(geom_dataset, geom_array):
    expected = (
        xr.Dataset(coords={"geom": np.concatenate([geom_array, geom_array])})
        .drop_indexes("geom")
        .set_xindex("geom", GeoVectorIndex, crs=geom_dataset.xindexes["geom"].crs)
    )
    actual = xr.concat([geom_dataset, geom_dataset], "geom")
    xr.testing.assert_identical(actual, expected)


def test_to_pandas_index(geom_dataset):
    index = geom_dataset.xindexes["geom"]
    assert index.to_pandas_index() is index._index.index


def test_isel(geom_dataset, first_geom_dataset):
    actual = geom_dataset.isel(geom=[0])
    xr.testing.assert_identical(actual, first_geom_dataset)


def test_sel_strict(geom_dataset, geom_array, first_geom_dataset):
    actual = geom_dataset.sel(geom=[geom_array[0]])
    xr.testing.assert_identical(actual, first_geom_dataset)


@pytest.mark.parametrize(
    "label",
    [
        shapely.Point(1, 1),
        [shapely.Point(1, 1)],
        xr.Variable("geom", [shapely.Point(1, 1)]),
        xr.DataArray([shapely.Point(1, 1)], dims="geom"),
    ],
)
def test_sel_nearest(geom_dataset, geom_array, first_geom_dataset, label):
    actual = geom_dataset.sel(geom=label, method="nearest")
    xr.testing.assert_identical(actual, first_geom_dataset)


def test_sel_query(geom_dataset, first_geom_dataset):
    actual = geom_dataset.sel(geom=shapely.box(0, 0, 2, 2), method="intersects")
    xr.testing.assert_identical(actual, first_geom_dataset)


def test_equals(geom_dataset, geom_dataset_no_index, first_geom_dataset):
    # different index types
    other = xr.Dataset(coords={"geom": [0, 1]})
    assert not geom_dataset.xindexes["geom"].equals(other.xindexes["geom"])

    # different CRS
    crs = CRS.from_user_input(4267)
    other = geom_dataset_no_index.set_xindex("geom", GeoVectorIndex, crs=crs)
    assert not geom_dataset.xindexes["geom"].equals(other.xindexes["geom"])

    # different geometries
    assert not geom_dataset.xindexes["geom"].equals(first_geom_dataset.xindexes["geom"])


def test_align(geom_dataset, first_geom_dataset):
    # test GeoVectorIndex's `join` and `reindex_like`
    aligned = xr.align(geom_dataset, first_geom_dataset, join="inner")
    assert all(ds.identical(first_geom_dataset) for ds in aligned)


def test_roll(geom_dataset, geom_array):
    expected = (
        xr.Dataset(coords={"geom": np.roll(geom_array, 1)})
        .drop_indexes("geom")
        .set_xindex("geom", GeoVectorIndex, crs=geom_dataset.xindexes["geom"].crs)
    )
    actual = geom_dataset.roll(geom=1, roll_coords=True)
    xr.testing.assert_identical(actual, expected)


def test_rename(geom_dataset):
    ds = geom_dataset.rename_vars(geom="renamed")
    assert ds.xindexes["renamed"]._index.index.name == "renamed"
    assert ds.xindexes["renamed"]._index.dim == "geom"


def test_repr_inline(geom_dataset):
    actual = geom_dataset.xindexes["geom"]._repr_inline_(70)
    expected = "GeoVectorIndex(crs=EPSG:26915)"
    assert actual == expected
