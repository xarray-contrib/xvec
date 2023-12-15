from __future__ import annotations

import gc
from collections.abc import Hashable, Sequence
from typing import Callable

import numpy as np
import shapely
import xarray as xr


def _zonal_stats_rasterize(
    acc,
    polygons: Sequence[shapely.Geometry],
    x_coords: Hashable,
    y_coords: Hashable,
    stats: str | Callable = "mean",
    name: str = "geometry",
    all_touched: bool = False,
    **kwargs,
):
    try:
        import rasterio
        import rioxarray  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "The rioxarray package is required for `zonal_stats()`. "
            "You can install it using 'conda install -c conda-forge rioxarray' or "
            "'pip install rioxarray'."
        ) from err

    if hasattr(polygons, "crs"):
        crs = polygons.crs
    else:
        crs = None

    transform = acc._obj.rio.transform()

    labels = rasterio.features.rasterize(
        zip(polygons, range(len(polygons))),
        out_shape=(
            acc._obj[y_coords].shape[0],
            acc._obj[x_coords].shape[0],
        ),
        transform=transform,
        fill=np.nan,
        all_touched=all_touched,
    )
    groups = acc._obj.groupby(xr.DataArray(labels, dims=(y_coords, x_coords)))
    if isinstance(stats, str):
        agg = getattr(groups, stats)(**kwargs)
    else:
        agg = groups.reduce(stats, keep_attrs=True, **kwargs)
    vec_cube = (
        agg.reindex(group=range(len(polygons)))
        .assign_coords(group=polygons)
        .rename(group=name)
    ).xvec.set_geom_indexes(name, crs=crs)

    del groups
    gc.collect()

    return vec_cube


def _zonal_stats_iterative(
    acc,
    polygons: Sequence[shapely.Geometry],
    x_coords: Hashable,
    y_coords: Hashable,
    stats: str | Callable = "mean",
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
    stats : Hashable
        Spatial aggregation statistic method, by default "mean". Any of the
        aggregations available as DataArray or DataArrayGroupBy like
        :meth:`~xarray.DataArray.mean`, :meth:`~xarray.DataArray.min`,
        :meth:`~xarray.DataArray.max`, or :meth:`~xarray.DataArray.quantile`,
        methods are available. Alternatively, you can pass a ``Callable`` supported
        by :meth:`~xarray.DataArray.reduce`.
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
            stats=stats,
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
    stats: str | Callable = "mean",
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
    stats : Hashable
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
    masked = acc._obj.where(xr.DataArray(mask, dims=(y_coords, x_coords)))
    if isinstance(stats, str):
        result = getattr(masked, stats)(
            dim=(y_coords, x_coords), keep_attrs=True, **kwargs
        )
    else:
        result = masked.reduce(
            stats, dim=(y_coords, x_coords), keep_attrs=True, **kwargs
        )

    del mask
    gc.collect()

    return result
