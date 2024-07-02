from __future__ import annotations

import gc
from collections.abc import Hashable, Iterable, Sequence
from typing import Any, Callable

import numpy as np
import pandas as pd
import shapely
import xarray as xr


def _agg_rasterize(groups, stats, **kwargs):
    if isinstance(stats, str):
        return getattr(groups, stats)(**kwargs)
    return groups.reduce(stats, keep_attrs=True, **kwargs)


def _agg_iterate(masked, stats, x_coords, y_coords, **kwargs):
    if isinstance(stats, str):
        return getattr(masked, stats)(
            dim=(y_coords, x_coords), keep_attrs=True, **kwargs
        )
    return masked.reduce(stats, dim=(y_coords, x_coords), keep_attrs=True, **kwargs)


def _zonal_stats_rasterize(
    acc,
    geometry: Sequence[shapely.Geometry],
    x_coords: Hashable,
    y_coords: Hashable,
    stats: str | Callable | Sequence[str | Callable | tuple] = "mean",
    name: str = "geometry",
    all_touched: bool = False,
    **kwargs,
) -> xr.DataArray | xr.Dataset:
    try:
        import rioxarray  # noqa: F401
        from rasterio import features
    except ImportError as err:
        raise ImportError(
            "The rioxarray package is required for `zonal_stats()`. "
            "You can install it using 'conda install -c conda-forge rioxarray' or "
            "'pip install rioxarray'."
        ) from err

    if hasattr(geometry, "crs"):
        crs = geometry.crs  # type: ignore
    else:
        crs = None

    transform = acc._obj.rio.transform()

    labels = features.rasterize(
        zip(geometry, range(len(geometry))),
        out_shape=(
            acc._obj[y_coords].shape[0],
            acc._obj[x_coords].shape[0],
        ),
        transform=transform,
        fill=np.nan,  # type: ignore
        all_touched=all_touched,
    )
    groups = acc._obj.groupby(xr.DataArray(labels, dims=(y_coords, x_coords)))

    if pd.api.types.is_list_like(stats):
        agg = {}
        for stat in stats:  # type: ignore
            if isinstance(stat, str):
                agg[stat] = _agg_rasterize(groups, stat, **kwargs)
            elif callable(stat):
                agg[stat.__name__] = _agg_rasterize(groups, stat, **kwargs)
            elif isinstance(stat, tuple):
                kws = stat[2] if len(stat) == 3 else {}
                agg[stat[0]] = _agg_rasterize(groups, stat[1], **kws)
            else:
                raise ValueError(f"{stat} is not a valid aggregation.")

        agg_array = xr.concat(
            agg.values(),
            dim=xr.DataArray(
                list(agg.keys()), name="zonal_statistics", dims="zonal_statistics"
            ),
        )
    elif isinstance(stats, str) or callable(stats):
        agg_array = _agg_rasterize(groups, stats, **kwargs)
    else:
        raise ValueError(f"{stats} is not a valid aggregation.")

    vec_cube = (
        agg_array.reindex(group=range(len(geometry)))
        .assign_coords(group=geometry)
        .rename(group=name)
    ).xvec.set_geom_indexes(name, crs=crs)

    del groups
    gc.collect()

    return vec_cube


def _zonal_stats_iterative(
    acc,
    geometry: Sequence[shapely.Geometry],
    x_coords: Hashable,
    y_coords: Hashable,
    stats: str | Callable | Sequence[str | Callable | tuple] = "mean",
    name: str = "geometry",
    all_touched: bool = False,
    n_jobs: int = -1,
    **kwargs: dict[str, Any],
) -> xr.DataArray | xr.Dataset:
    """Extract the values from a dataset indexed by a set of geometries

    The CRS of the raster and that of geometry need to be equal.
    Xvec does not verify their equality.

    Parameters
    ----------
    geometry : Sequence[shapely.Geometry]
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
    name : str, optional
        Name of the dimension that will hold the ``geometry``, by default "geometry"
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
        from joblib import Parallel, delayed  # type: ignore
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
        for geom in geometry
    )
    if hasattr(geometry, "crs"):
        crs = geometry.crs  # type: ignore
    else:
        crs = None
    vec_cube = xr.concat(
        zonal,  # type: ignore
        dim=xr.DataArray(geometry, name=name, dims=name),
    ).xvec.set_geom_indexes(name, crs=crs)
    gc.collect()

    return vec_cube


def _agg_geom(
    acc,
    geom,
    trans,
    x_coords: str | None = None,
    y_coords: str | None = None,
    stats: str | Callable | Iterable[str | Callable | tuple] = "mean",
    all_touched: bool = False,
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
    from rasterio import features

    mask = features.geometry_mask(
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
    if pd.api.types.is_list_like(stats):
        agg = {}
        for stat in stats:  # type: ignore
            if isinstance(stat, str):
                agg[stat] = _agg_iterate(masked, stat, x_coords, y_coords, **kwargs)
            elif callable(stat):
                agg[stat.__name__] = _agg_iterate(
                    masked, stat, x_coords, y_coords, **kwargs
                )
            elif isinstance(stat, tuple):
                kws = stat[2] if len(stat) == 3 else {}
                agg[stat[0]] = _agg_iterate(masked, stat[1], x_coords, y_coords, **kws)
            else:
                raise ValueError(f"{stat} is not a valid aggregation.")

        result = xr.concat(
            agg.values(),
            dim=xr.DataArray(
                list(agg.keys()), name="zonal_statistics", dims="zonal_statistics"
            ),
        )
    elif isinstance(stats, str) or callable(stats):
        result = _agg_iterate(masked, stats, x_coords, y_coords, **kwargs)
    else:
        raise ValueError(f"{stats} is not a valid aggregation.")

    del mask
    gc.collect()

    return result


def _zonal_stats_exactextract(
    acc,
    geometry: Sequence[shapely.Geometry],
    x_coords: Hashable,
    y_coords: Hashable,
    stats: str | Callable | Sequence[str | Callable | tuple] = "mean",
    name: str = "geometry",
) -> xr.DataArray | xr.Dataset:
    """Extract the values from a dataset indexed by a set of geometries

    The CRS of the raster and that of geometry need to be equal.
    Xvec does not verify their equality.

    Parameters
    ----------
    geometry : Sequence[shapely.Geometry]
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
        strings that can be used to construct :py:class:`Operation` objects
        supported by :func:`exactextract.exact_extract` (e.g., ``"mean"``,
        ``"quantile(q=0.20)"``)
    name : str, optional
        Name of the dimension that will hold the ``geometry``, by default "geometry"

    Returns
    -------
    Dataset | DataArray
        A subset of the original object with N-1 dimensions indexed by
        the the GeometryIndex.

    """
    if hasattr(geometry, "crs"):
        crs = geometry.crs  # type: ignore
    else:
        raise AttributeError(
            "Geometry input does not have a Coordinate Reference System (CRS)."
        )

    # the input should be xarray.DataArray
    original_is_ds = False
    if not isinstance(acc._obj, xr.core.dataarray.DataArray):
        original_ds = acc._obj
        acc._obj = acc._obj.to_dataarray()
        original_is_ds = True

    # Unstack the results
    agg = {}
    if pd.api.types.is_list_like(stats):
        for stat in stats:  # type: ignore
            if not isinstance(stat, str):
                raise ValueError(f"{stat} is not a valid aggregation.")

        results, original_shape, coords_info, locs = _agg_exactextract(
            acc, geometry, crs, x_coords, y_coords, stats, name, original_is_ds
        )
        i = 0
        for stat in stats:  # type: ignore
            df = results.iloc[:, i : i + locs]
            # Unstack the result
            arr = df.values.reshape(original_shape)
            if original_is_ds is True:
                data_vars = {}
                for idx, data_var in enumerate(acc._obj["variable"].values):
                    data_vars[data_var] = (coords_info.keys(), arr[:, idx])
                result = xr.Dataset(
                    data_vars=data_vars,
                    coords=coords_info,
                ).xvec.set_geom_indexes(name, crs=crs)
            else:
                result = xr.DataArray(
                    arr, coords=coords_info, dims=coords_info.keys()
                ).xvec.set_geom_indexes(name, crs=crs)

            agg[stat] = result
            i += locs
        vec_cube = xr.concat(
            agg.values(),
            dim=xr.DataArray(
                list(agg.keys()), name="zonal_statistics", dims="zonal_statistics"
            ),
        )
    elif isinstance(stats, str):
        results, original_shape, coords_info, _ = _agg_exactextract(
            acc, geometry, crs, x_coords, y_coords, stats, name, original_is_ds
        )
        # Unstack the result
        arr = results.values.reshape(original_shape)
        if original_is_ds is True:
            data_vars = {}
            for idx, data_var in enumerate(acc._obj["variable"].values):
                data_vars[data_var] = (coords_info.keys(), arr[:, idx])
                vec_cube = xr.Dataset(
                    data_vars=data_vars,
                    coords=coords_info,
                ).xvec.set_geom_indexes(name, crs=crs)
        else:
            vec_cube = xr.DataArray(
                arr, coords=coords_info, dims=coords_info.keys()
            ).xvec.set_geom_indexes(name, crs=crs)
    else:
        raise ValueError(f"{stats} is not a valid aggregation.")

    vec_cube.attrs = acc._obj.attrs

    if original_is_ds:
        acc._obj = original_ds

    return vec_cube


def _agg_exactextract(
    acc,
    geometry,
    crs,
    x_coords: Hashable,
    y_coords: Hashable,
    stats: str | Callable | Iterable[str | Callable | tuple] = "mean",
    name: str = "geometry",
    original_is_ds: bool = False,
):
    """Extract the values from a dataset indexed by a set of geometries

    The CRS of the raster and that of geometry need to be equal.
    Xvec does not verify their equality.

    Parameters
    ----------
    geometry : Sequence[shapely.Geometry]
        An arrray-like (1-D) of shapely geometries, like a numpy array or
        :class:`geopandas.GeoSeries`.
    crs : Coordinate Reference System of the geometry objects.
    x_coords : Hashable
        name of the coordinates containing ``x`` coordinates (i.e. the first value
        in the coordinate pair encoding the vertex of the polygon)
    y_coords : Hashable
        name of the coordinates containing ``y`` coordinates (i.e. the second value
        in the coordinate pair encoding the vertex of the polygon)
    stats : Hashable
        Spatial aggregation statistic method, by default "mean". Any of the
        strings that can be used to construct :py:class:`Operation` objects
        supported by :func:`exactextract.exact_extract` (e.g., ``"mean"``,
        ``"quantile(q=0.20)"``)
    name : str, optional
        Name of the dimension that will hold the ``geometry``, by default "geometry"
    original_is_ds : bool
        If True, all pixels touched by geometries will be considered. If False, only
        pixels whose center is within the polygon or that are selected by
        Bresenham’s line algorithm will be considered.

    Returns
    -------
    pandas.DataFrame
        Aggregated values over the geometry.

    """
    try:
        import exactextract
    except ImportError as err:
        raise ImportError(
            "The exactextract package is required for `zonal_stats()` with "
            "method='exactextract'."
        ) from err
    try:
        import geopandas as gpd
    except ImportError as err:
        raise ImportError(
            "The geopandas package is required for `zonal_stats()` with "
            "method='exactextract'. "
            "You can install it using 'conda install -c conda-forge geopandas' or "
            "'pip install geopandas'."
        ) from err

    # Stack the other dimensions into one dimension called "location"
    arr_dims = tuple(dim for dim in acc._obj.dims if dim not in [x_coords, y_coords])
    data = acc._obj.stack(location=arr_dims)
    locs = data.location.size

    # Check the order of dimensions
    data = data.transpose("location", y_coords, x_coords)

    # Aggregation result
    gdf = gpd.GeoDataFrame(geometry=geometry, crs=crs)
    results = exactextract.exact_extract(rast=data, vec=gdf, ops=stats, output="pandas")
    # Get all the dimensions execpt x_coords, y_coords, they will be used to stack the
    # dataarray later
    if original_is_ds is True:
        # Get the original dataset information to use for unstacking the resulte later
        coords_info = {name: geometry}
        original_shape = [len(geometry)]
        for dim in arr_dims:
            original_shape.append(acc._obj[dim].size)
            if dim != "variable":
                coords_info[dim] = acc._obj[dim].values
    else:
        # Get the original dataarray information to use for unstacking the resulte later
        coords_info = {name: geometry}
        original_shape = [len(geometry)]
        for dim in arr_dims:
            original_shape.append(acc._obj[dim].size)
            coords_info[dim] = acc._obj[dim].values
    return results, original_shape, coords_info, locs
