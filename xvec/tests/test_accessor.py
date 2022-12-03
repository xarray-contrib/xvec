import numpy as np
import pytest
import shapely
import xarray as xr

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


# Test .xvec geom_coords_names


def test_geom_coords_names(multi_geom_dataset):
    assert multi_geom_dataset.xvec._geom_coords_names == ["geom", "geom_z"]
    assert multi_geom_dataset.xvec.geom_coords_names == ["geom", "geom_z"]

    with pytest.raises(AttributeError, match="can't set attribute 'geom_coords_names'"):
        multi_geom_dataset.xvec.geom_coords_names = ["a", "b"]


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


def test_set_crs_multiple(multi_geom_dataset):
    with_crs = multi_geom_dataset.xvec.to_crs(geom=4326, geom_z=4326)
    assert with_crs.xindexes["geom"].crs.equals(4326)
    assert with_crs.xindexes["geom_z"].crs.equals(4326)
