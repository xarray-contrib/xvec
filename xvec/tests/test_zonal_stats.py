import geodatasets
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr

import xvec  # noqa: F401


def test_aggregate_raster_cubes():
    da = xr.DataArray(
        np.zeros((10, 10, 5)),
        coords={
            "x": range(10),
            "y": range(20, 30),
            "time": pd.date_range("2023-01-01", periods=5),
        },
    )
    da = da.to_dataset(name="test")

    # Create the polygons
    polygon1 = shapely.geometry.Polygon([(1, 22), (4, 22), (4, 26), (1, 26)])
    polygon2 = shapely.geometry.Polygon([(6, 22), (9, 22), (9, 26), (6, 26)])
    polygons = gpd.GeoSeries([polygon1, polygon2], crs="EPSG:4326")

    # Expected results
    expected = xr.DataArray(
        np.zeros((2, 5)),
        coords={
            "geometry": polygons,
            "time": pd.date_range("2023-01-01", periods=5),
        },
    ).xvec.set_geom_indexes("geometry", crs="EPSG:4326")

    expected = expected.to_dataset(name="test")
    expected = expected.set_coords("geometry")

    # Actual results
    actual = da.xvec.zonal_stats(polygons, "x", "y", stat="sum")

    # Testing
    xr.testing.assert_identical(actual, expected)

    ds = xr.tutorial.open_dataset("eraint_uvz")
    world = gpd.read_file(geodatasets.get_path("naturalearth land"))
    ds.xvec.zonal_stats(world.geometry, "longitude", "latitude")
