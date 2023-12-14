import geodatasets
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
import xarray as xr

import xvec  # noqa: F401


@pytest.mark.parametrize("method", ["rasterize", "iterate"])
def test_structure(method):
    da = xr.DataArray(
        np.ones((10, 10, 5)),
        coords={
            "x": range(10),
            "y": range(20, 30),
            "time": pd.date_range("2023-01-01", periods=5),
        },
    )

    polygon1 = shapely.geometry.Polygon([(1, 22), (4, 22), (4, 26), (1, 26)])
    polygon2 = shapely.geometry.Polygon([(6, 22), (9, 22), (9, 29), (6, 26)])
    polygons = gpd.GeoSeries([polygon1, polygon2], crs="EPSG:4326")

    expected = xr.DataArray(
        np.array([[12.0] * 5, [18.0] * 5]),
        coords={
            "geometry": polygons,
            "time": pd.date_range("2023-01-01", periods=5),
        },
    ).xvec.set_geom_indexes("geometry", crs="EPSG:4326")

    actual = da.xvec.zonal_stats(polygons, "x", "y", stats="sum", method=method)
    xr.testing.assert_identical(actual, expected)

    actual_ix = da.xvec.zonal_stats(
        polygons, "x", "y", stats="sum", method=method, index=True
    )
    xr.testing.assert_identical(
        actual_ix, expected.assign_coords({"index": ("geometry", polygons.index)})
    )

    # dataset
    ds = da.to_dataset(name="test")

    expected_ds = expected.to_dataset(name="test").set_coords("geometry")
    actual_ds = ds.xvec.zonal_stats(polygons, "x", "y", stats="sum", method=method)
    xr.testing.assert_identical(actual_ds, expected_ds)

    actual_ix_ds = ds.xvec.zonal_stats(
        polygons, "x", "y", stats="sum", method=method, index=True
    )
    xr.testing.assert_identical(
        actual_ix_ds, expected_ds.assign_coords({"index": ("geometry", polygons.index)})
    )

    # named index
    polygons.index.name = "my_index"
    actual_ix_named = da.xvec.zonal_stats(
        polygons, "x", "y", stats="sum", method=method
    )
    xr.testing.assert_identical(
        actual_ix_named,
        expected.assign_coords({"my_index": ("geometry", polygons.index)}),
    )
    actual_ix_names_ds = ds.xvec.zonal_stats(
        polygons, "x", "y", stats="sum", method=method
    )
    xr.testing.assert_identical(
        actual_ix_names_ds,
        expected_ds.assign_coords({"my_index": ("geometry", polygons.index)}),
    )


def test_match():
    ds = xr.tutorial.open_dataset("eraint_uvz")
    world = gpd.read_file(geodatasets.get_path("naturalearth land"))
    rasterize = ds.xvec.zonal_stats(
        world.geometry, "longitude", "latitude", method="rasterize"
    )
    iterate = ds.xvec.zonal_stats(
        world.geometry, "longitude", "latitude", method="iterate"
    )

    xr.testing.assert_allclose(rasterize, iterate)


@pytest.mark.parametrize("method", ["rasterize", "iterate"])
def test_dataset(method):
    ds = xr.tutorial.open_dataset("eraint_uvz")
    world = gpd.read_file(geodatasets.get_path("naturalearth land"))
    result = ds.xvec.zonal_stats(world.geometry, "longitude", "latitude", method=method)

    xr.testing.assert_allclose(
        xr.Dataset(
            {
                "z": np.array(61367.76185577),
                "u": np.array(4.19631497),
                "v": np.array(-0.49170332),
            }
        ),
        result.mean(),
    )


@pytest.mark.parametrize("method", ["rasterize", "iterate"])
def test_dataarray(method):
    ds = xr.tutorial.open_dataset("eraint_uvz")
    world = gpd.read_file(geodatasets.get_path("naturalearth land"))
    result = ds.z.xvec.zonal_stats(
        world.geometry, "longitude", "latitude", method=method
    )

    assert result.shape == (127, 2, 3)
    assert result.dims == ("geometry", "month", "level")
    assert result.mean() == pytest.approx(61367.76185577)


@pytest.mark.parametrize("method", ["rasterize", "iterate"])
def test_stat(method):
    ds = xr.tutorial.open_dataset("eraint_uvz")
    world = gpd.read_file(geodatasets.get_path("naturalearth land"))

    mean_ = ds.z.xvec.zonal_stats(
        world.geometry, "longitude", "latitude", method=method
    )
    median_ = ds.z.xvec.zonal_stats(
        world.geometry, "longitude", "latitude", method=method, stats="median"
    )
    quantile_ = ds.z.xvec.zonal_stats(
        world.geometry, "longitude", "latitude", method=method, stats="quantile", q=0.2
    )

    assert mean_.mean() == pytest.approx(61367.76185577)
    assert median_.mean() == pytest.approx(61370.18563539)
    assert quantile_.mean() == pytest.approx(61279.93619836)


@pytest.mark.parametrize("method", ["rasterize", "iterate"])
def test_all_touched(method):
    ds = xr.tutorial.open_dataset("eraint_uvz")
    world = gpd.read_file(geodatasets.get_path("naturalearth land"))

    default = ds.z.xvec.zonal_stats(
        world.geometry[:10],
        "longitude",
        "latitude",
        all_touched=False,
        stats="sum",
        method=method,
    )
    touched = ds.z.xvec.zonal_stats(
        world.geometry[:10],
        "longitude",
        "latitude",
        all_touched=True,
        stats="sum",
        method=method,
    )

    assert (default < touched).all()


def test_n_jobs():
    ds = xr.tutorial.open_dataset("eraint_uvz")
    world = gpd.read_file(geodatasets.get_path("naturalearth land"))

    one = ds.xvec.zonal_stats(
        world.geometry[:10], "longitude", "latitude", method="iterate", n_jobs=1
    )
    default = ds.xvec.zonal_stats(
        world.geometry[:10], "longitude", "latitude", method="iterate", n_jobs=1
    )

    xr.testing.assert_identical(one, default)


def test_method_error():
    ds = xr.tutorial.open_dataset("eraint_uvz")
    world = gpd.read_file(geodatasets.get_path("naturalearth land"))
    with pytest.raises(ValueError, match="method 'quick' is not supported"):
        ds.xvec.zonal_stats(world.geometry, "longitude", "latitude", method="quick")


@pytest.mark.parametrize("method", ["rasterize", "iterate"])
def test_crs(method):
    da = xr.DataArray(
        np.ones((10, 10, 5)),
        coords={
            "x": range(10),
            "y": range(20, 30),
            "time": pd.date_range("2023-01-01", periods=5),
        },
    )

    polygon1 = shapely.geometry.Polygon([(1, 22), (4, 22), (4, 26), (1, 26)])
    polygon2 = shapely.geometry.Polygon([(6, 22), (9, 22), (9, 29), (6, 26)])
    polygons = np.array([polygon1, polygon2])

    expected = xr.DataArray(
        np.array([[12.0] * 5, [18.0] * 5]),
        coords={
            "geometry": polygons,
            "time": pd.date_range("2023-01-01", periods=5),
        },
    ).xvec.set_geom_indexes("geometry", crs=None)

    actual = da.xvec.zonal_stats(polygons, "x", "y", stats="sum", method=method)
    xr.testing.assert_identical(actual, expected)
