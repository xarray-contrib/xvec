import numpy as np
import pytest
import shapely
import xarray as xr
from pyproj import CRS

from xvec import GeometryIndex


@pytest.fixture(scope="session")
def geom_array():
    return np.array([shapely.Point(1, 2), shapely.Point(3, 4)])


@pytest.fixture(scope="session")
def geom_array_z():
    return np.array([shapely.Point(10, 20, 30), shapely.Point(30, 40, 50)])


@pytest.fixture(scope="session")
def dataset_w_geom(geom_array):
    # a dataset with a geometry coordinate and default index
    return xr.Dataset(coords={"geom": geom_array})


@pytest.fixture(scope="session")
def geom_dataset_no_index(geom_array):
    # a dataset with a geometry coordinate but no index
    ds = xr.Dataset(coords={"geom": geom_array})
    return ds.drop_indexes("geom")


@pytest.fixture(scope="session")
def geom_dataset(geom_dataset_no_index):
    # a dataset with a geometry coordinate baked by a GeometryIndex
    crs = CRS.from_user_input(26915)
    geom_dataset_no_index["geom"].attrs["crs"] = crs
    return geom_dataset_no_index.set_xindex("geom", GeometryIndex, crs=crs)


@pytest.fixture(scope="session")
def geom_dataset_no_crs(geom_dataset_no_index):
    # a dataset with a geometry coordinate baked by a GeometryIndex (no CRS)
    return geom_dataset_no_index.set_xindex("geom", GeometryIndex)


@pytest.fixture(scope="session")
def first_geom_dataset(geom_dataset, geom_array):
    return (
        xr.Dataset(coords={"geom": [geom_array[0]]})
        .drop_indexes("geom")
        .set_xindex("geom", GeometryIndex, crs=geom_dataset.xindexes["geom"].crs)
    )


@pytest.fixture(scope="session")
def multi_dataset(geom_array, geom_array_z):
    return xr.Dataset(
        coords={
            "geom": geom_array,
            "geom_z": geom_array_z,
        }
    )


@pytest.fixture(scope="session")
def multi_geom_dataset(geom_array, geom_array_z):
    return (
        xr.Dataset(
            coords={
                "geom": geom_array,
                "geom_z": geom_array_z,
            }
        )
        .drop_indexes(["geom", "geom_z"])
        .set_xindex("geom", GeometryIndex, crs=26915)
        .set_xindex("geom_z", GeometryIndex, crs=26915)
    )


@pytest.fixture(scope="session")
def multi_geom_no_index_dataset(geom_array, geom_array_z):
    return (
        xr.Dataset(
            coords={
                "geom": geom_array,
                "geom_z": geom_array_z,
                "geom_no_ix": geom_array,
            }
        )
        .drop_indexes(["geom", "geom_z"])
        .set_xindex("geom", GeometryIndex, crs=26915)
        .set_xindex("geom_z", GeometryIndex, crs=26915)
    )


@pytest.fixture(scope="session")
def multi_geom_one_ix_foo(geom_array):
    return (
        xr.Dataset(
            coords={
                "geom": geom_array,
                "geom2": geom_array,
                "foo": np.array([1, 2]),
            }
        )
        .drop_indexes(["geom"])
        .set_xindex("geom", GeometryIndex, crs=26915)
    )
