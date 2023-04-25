import numpy as np
import pytest
import shapely
import xarray as xr
from pyproj import CRS

from xvec import GeometryIndex


def test_set_index(geom_dataset_no_index):
    crs = CRS.from_user_input(26915)
    ds = geom_dataset_no_index.set_xindex("geom", GeometryIndex, crs=crs)

    # test properties
    assert isinstance(ds.xindexes["geom"], GeometryIndex)
    assert ds.xindexes["geom"].crs == crs
    np.testing.assert_array_equal(ds.xindexes["geom"].sindex.geometries, ds.geom.values)

    # test `GeometryIndex.create_variables`
    assert ds.geom.variable._data.array is ds.xindexes["geom"]._index.index

    no_crs_ds = geom_dataset_no_index.set_xindex("geom", GeometryIndex)
    assert no_crs_ds.xindexes["geom"].crs is None

    no_geom_ds = xr.Dataset(coords={"no_geom": ("x", [0, 1, 2])})
    with pytest.raises(ValueError, match="array must contain shapely.Geometry objects"):
        no_geom_ds.set_xindex("no_geom", GeometryIndex, crs=crs)


def test_concat(geom_dataset, geom_array, geom_dataset_no_index, geom_dataset_no_crs):
    expected = xr.Dataset(
        coords={"geom": np.concatenate([geom_array, geom_array])}
    ).xvec.set_geom_indexes("geom", crs=geom_dataset.xindexes["geom"].crs)
    actual = xr.concat([geom_dataset, geom_dataset], "geom")
    xr.testing.assert_identical(actual, expected)

    # different CRS
    crs = CRS.from_user_input(4267)
    geom_dataset_alt = geom_dataset_no_index.set_xindex("geom", GeometryIndex, crs=crs)

    with pytest.raises(ValueError, match="cannot determine common CRS"):
        xr.concat([geom_dataset, geom_dataset_alt], "geom")

    # no CRS
    expected = (
        xr.Dataset(coords={"geom": np.concatenate([geom_array, geom_array])})
        .drop_indexes("geom")
        .set_xindex("geom", GeometryIndex)
    )
    actual = xr.concat([geom_dataset_no_crs, geom_dataset_no_crs], "geom")
    xr.testing.assert_identical(actual, expected)

    # mixed CRS / no CRS
    with pytest.warns(
        UserWarning, match="CRS not set for some of the concatenation inputs"
    ):
        xr.concat([geom_dataset, geom_dataset_no_crs], "geom")


def test_to_pandas_index(geom_dataset):
    index = geom_dataset.xindexes["geom"]
    assert index.to_pandas_index() is index._index.index


def test_isel(geom_dataset, first_geom_dataset):
    actual = geom_dataset.isel(geom=[0])
    xr.testing.assert_identical(actual, first_geom_dataset)

    # scalar selection
    actual = geom_dataset.isel(geom=0)
    assert len(actual.geom.dims) == 0
    assert "geom" not in actual.xindexes


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
def test_sel_nearest(geom_dataset, first_geom_dataset, label):
    actual = geom_dataset.sel(geom=label, method="nearest")
    xr.testing.assert_identical(actual, first_geom_dataset)


def test_sel_nearest_error(geom_dataset):
    with pytest.raises(ValueError, match="labels must be shapely.Geometry objects"):
        geom_dataset.sel(geom=[0], method="nearest")


def test_sel_query(geom_dataset, first_geom_dataset):
    actual = geom_dataset.sel(geom=shapely.box(0, 0, 2, 2), method="intersects")
    xr.testing.assert_identical(actual, first_geom_dataset)


def test_equals(
    geom_dataset, geom_dataset_no_index, first_geom_dataset, geom_dataset_no_crs
):
    assert geom_dataset.xindexes["geom"].equals(geom_dataset.xindexes["geom"])

    # different index types
    other = xr.Dataset(coords={"geom": [0, 1]})
    assert not geom_dataset.xindexes["geom"].equals(other.xindexes["geom"])

    # different CRS
    crs = CRS.from_user_input(4267)
    other = geom_dataset_no_index.set_xindex("geom", GeometryIndex, crs=crs)
    assert not geom_dataset.xindexes["geom"].equals(other.xindexes["geom"])

    # no CRS
    assert geom_dataset_no_crs.xindexes["geom"].equals(
        geom_dataset_no_crs.xindexes["geom"]
    )

    # different geometries
    assert not geom_dataset.xindexes["geom"].equals(first_geom_dataset.xindexes["geom"])


def test_align(
    geom_dataset, first_geom_dataset, geom_dataset_no_index, geom_dataset_no_crs
):
    # test both GeometryIndex's `join` and `reindex_like`
    aligned = xr.align(geom_dataset, first_geom_dataset, join="inner")
    assert all(ds.identical(first_geom_dataset) for ds in aligned)

    # test conflicting CRS
    crs = CRS.from_user_input(4267)
    geom_dataset_alt = geom_dataset_no_index.set_xindex("geom", GeometryIndex, crs=crs)

    with pytest.raises(ValueError, match="CRS mismatch"):
        xr.align(geom_dataset_alt, first_geom_dataset, join="inner")

    with pytest.raises(ValueError, match="CRS mismatch"):
        first_geom_dataset.reindex_like(geom_dataset_alt)

    # no CRS
    first_geom_dataset_no_crs = geom_dataset_no_crs.isel(geom=[0])
    aligned = xr.align(geom_dataset_no_crs, first_geom_dataset_no_crs, join="inner")
    assert all(ds.identical(first_geom_dataset_no_crs) for ds in aligned)


def test_roll(geom_dataset, geom_array):
    expected = (
        xr.Dataset(coords={"geom": np.roll(geom_array, 1)})
        .drop_indexes("geom")
        .set_xindex("geom", GeometryIndex, crs=geom_dataset.xindexes["geom"].crs)
    )
    expected["geom"].attrs["crs"] = geom_dataset.xindexes["geom"].crs
    actual = geom_dataset.roll(geom=1, roll_coords=True)
    xr.testing.assert_identical(actual, expected)


def test_rename(geom_dataset):
    ds = geom_dataset.rename_vars(geom="renamed")
    assert ds.xindexes["renamed"]._index.index.name == "renamed"
    assert ds.xindexes["renamed"]._index.dim == "geom"


def test_repr_inline(geom_dataset, geom_dataset_no_crs):
    actual = geom_dataset.xindexes["geom"]._repr_inline_(70)
    expected = "GeometryIndex (crs=EPSG:26915)"
    assert actual == expected

    actual = geom_dataset_no_crs.xindexes["geom"]._repr_inline_(70)
    expected = "GeometryIndex (crs=None)"
    assert actual == expected
