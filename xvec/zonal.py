from __future__ import annotations

import gc
from collections.abc import Hashable, Sequence

import shapely
import xarray as xr


def _zonal_stats_rasterio(
    acc,
    polygons: Sequence[shapely.Geometry],
    x_coords: Hashable,
    y_coords: Hashable,
    stat: str = "mean",
    name: str = "geometry",
    all_touched: bool = False,
    n_jobs: int = -1,
    **kwargs,
):
    """Extract the values from a dataset indexed by a set of geometries

    The CRS of the raster and that of polygons need to be equal.
    Xvec does not verify their equality.

    Parameters
    ----------
    polygons : Sequence[shapely.Geometry]
        An arrray-like (1-D) of shapely geometries, like a numpy array or
        :class:`geopandas.GeoSeries`.
    x_coords : Hashable
        name of the coordinates containing ``x`` coordinates (i.e. the first value
        in the coordinate pair encoding the vertex of the polygon)
    y_coords : Hashable
        name of the coordinates containing ``y`` coordinates (i.e. the second value
        in the coordinate pair encoding the vertex of the polygon)
    stat : Hashable
        Spatial aggregation statistic method, by default "mean". It supports the
        following statistcs: ['mean', 'median', 'min', 'max', 'sum']
    name : Hashable, optional
        Name of the dimension that will hold the ``polygons``, by default "geometry"
    all_touched : bool, optional
        If True, all pixels touched by geometries will be considered. If False, only
        pixels whose center is within the polygon or that are selected by
        Bresenham’s line algorithm will be considered.
    n_jobs : int, optional
        Number of parallel threads to use.
        It is recommended to set this to the number of physical cores in the CPU.
    **kwargs : optional
        Keyword arguments to be passed to the aggregation function
        (as ``Dataset.mean(**kwargs)``).

    Returns
    -------
    Dataset | DataArray
        A subset of the original object with N-1 dimensions indexed by
        the the GeometryIndex.

    """
    try:
        import rioxarray  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "The rioxarray package is required for `zonal_stats()`. "
            "You can install it using 'conda install -c conda-forge rioxarray' or "
            "'pip install rioxarray'."
        ) from err

    try:
        from joblib import Parallel, delayed
    except ImportError as err:
        raise ImportError(
            "The joblib package is required for `xvec._spatial_agg()`. "
            "You can install it using 'conda install -c conda-forge joblib' or "
            "'pip install joblib'."
        ) from err

    transform = acc._obj.rio.transform()

    zonal = Parallel(n_jobs=n_jobs)(
        delayed(_agg_geom)(
            acc,
            geom,
            transform,
            x_coords,
            y_coords,
            stat=stat,
            all_touched=all_touched,
            **kwargs,
        )
        for geom in polygons
    )
    if hasattr(polygons, "crs"):
        crs = polygons.crs
    else:
        crs = None
    vec_cube = xr.concat(
        zonal, dim=xr.DataArray(polygons, name=name, dims=name)
    ).xvec.set_geom_indexes(name, crs=crs)
    gc.collect()

    return vec_cube


def _agg_geom(
    acc,
    geom,
    trans,
    x_coords: str = None,
    y_coords: str = None,
    stat: str = "mean",
    all_touched=False,
    **kwargs,
):
    """Aggregate the values from a dataset over a polygon geometry.

    The CRS of the raster and that of points need to be in wgs84.
    Xvec does not verify their equality.

    Parameters
    ----------
    geom : Polygon[shapely.Geometry]
        An arrray-like (1-D) of shapely geometry, like a numpy array or GeoPandas
        GeoSeries.
    trans : affine.Affine
        Affine transformer.
        Representing the geometric transformation applied to the data.
    x_coords : Hashable
        Name of the axis containing ``x`` coordinates.
    y_coords : Hashable
        Name of the axis containing ``y`` coordinates.
    stat : Hashable
        Spatial aggregation statistic method, by default "mean". It supports the
        following statistcs: ['mean', 'median', 'min', 'max', 'sum']
    all_touched : bool, optional
        If True, all pixels touched by geometries will be considered. If False, only
        pixels whose center is within the polygon or that are selected by
        Bresenham’s line algorithm will be considered.

    Returns
    -------
    Array
        Aggregated values over the geometry.

    """
    import rasterio

    mask = rasterio.features.geometry_mask(
        [geom],
        out_shape=(
            acc._obj[y_coords].shape[0],
            acc._obj[x_coords].shape[0],
        ),
        transform=trans,
        invert=True,
        all_touched=all_touched,
    )
    result = getattr(
        acc._obj.where(xr.DataArray(mask, dims=(y_coords, x_coords))), stat
    )(dim=(y_coords, x_coords), **kwargs)

    del mask
    gc.collect()

    return result