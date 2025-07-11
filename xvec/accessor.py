from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
import shapely
import xarray as xr
import xproj  # noqa: F401
from pyproj import CRS

from .index import GeometryIndex
from .plotting import _plot
from .utils import transform_geom
from .zonal import (
    _get_method,
    _variable_zonal,
    _variable_zonal_exactextract,
    _zonal_stats_exactextract,
    _zonal_stats_iterative,
    _zonal_stats_rasterize,
)

if TYPE_CHECKING:
    from geopandas import GeoDataFrame


@xr.register_dataarray_accessor("xvec")
@xr.register_dataset_accessor("xvec")
class XvecAccessor:
    """Access geometry-based methods for DataArrays and Datasets with Shapely geometry.

    Currently works on coordinates with :class:`xvec.GeometryIndex`.
    """

    def __init__(self, xarray_obj: xr.Dataset | xr.DataArray) -> None:
        """xvec init, nothing to be done here."""
        self._obj = xarray_obj
        self._geom_coords_all = [
            name
            for name in self._obj.coords
            if self.is_geom_variable(name, has_index=False)
        ]
        self._geom_indexes = [
            name
            for name in self._obj.coords
            if self.is_geom_variable(name, has_index=True)
        ]

    def is_geom_variable(
        self, name: Hashable, has_index: bool = True
    ) -> bool | np.bool_:
        """Check if coordinate variable is composed of :class:`shapely.Geometry`.

        Can return all such variables or only those using :class:`~xvec.GeometryIndex`.

        Parameters
        ----------
        name : coordinate variable name
        has_index : bool, optional
            control if all variables are returned (``False``) or only those that have
            :class:`xvec.GeometryIndex` assigned (``True``). By default ``False``

        Returns
        -------
        bool

        Examples
        --------

        >>> ds = (
        ...     xr.Dataset(
        ...         coords={
        ...             "geom": np.array([shapely.Point(1, 2), shapely.Point(3, 4)]),
        ...             "geom2": np.array([shapely.Point(1, 2), shapely.Point(3, 4)]),
        ...             "foo": np.array([1, 2]),
        ...         }
        ...     )
        ...     .xvec.set_geom_indexes("geom", crs=26915)
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:  (geom: 2, geom2: 2, foo: 2)
        Coordinates:
          * geom     (geom) object POINT (1 2) POINT (3 4)
          * geom2    (geom2) object POINT (1 2) POINT (3 4)
          * foo      (foo) int64 1 2
        Data variables:
            *empty*
        Indexes:
            geom     GeometryIndex (crs=EPSG:26915)
        >>> ds.xvec.is_geom_variable("geom")
        True
        >>> ds.xvec.is_geom_variable("foo")
        False
        >>> ds.xvec.is_geom_variable("geom2")
        False
        >>> ds.xvec.is_geom_variable("geom2", has_index=False)
        True

        See also
        --------
        geom_coords
        geom_coords_indexed
        """
        if isinstance(self._obj.xindexes.get(name), GeometryIndex):
            return True
        if not has_index:
            if self._obj[name].dtype is np.dtype("O"):
                if self._obj[name].ndim > 0 and len(self._obj[name].data) > 10:
                    # try first on a small subset
                    subset = self._obj[name].data[0:10]
                    if np.all(shapely.is_valid_input(subset)):
                        return np.all(shapely.is_valid_input(self._obj[name].data))
                else:
                    return np.all(shapely.is_valid_input(self._obj[name].data))
        return False

    @property
    def geom_coords(self) -> xr.Coordinates:
        """Returns a dictionary of xarray.DataArray objects corresponding to
        coordinate variables composed of :class:`shapely.Geometry` objects.

        Examples
        --------
        >>> ds = (
        ...     xr.Dataset(
        ...         coords={
        ...             "geom": np.array([shapely.Point(1, 2), shapely.Point(3, 4)]),
        ...             "geom_z": np.array(
        ...                 [shapely.Point(10, 20, 30), shapely.Point(30, 40, 50)]
        ...             ),
        ...             "geom_no_index": np.array(
        ...                 [shapely.Point(1, 2), shapely.Point(3, 4)]
        ...             ),
        ...         }
        ...     )
        ...     .xvec.set_geom_indexes(["geom", "geom_z"], crs=26915)
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:        (geom: 2, geom_z: 2, geom_no_index: 2)
        Coordinates:
          * geom           (geom) object POINT (1 2) POINT (3 4)
          * geom_z         (geom_z) object POINT Z (10 20 30) POINT Z (30 40 50)
          * geom_no_index  (geom_no_index) object POINT (1 2) POINT (3 4)
        Data variables:
            *empty*
        Indexes:
            geom           GeometryIndex (crs=EPSG:26915)
            geom_z         GeometryIndex (crs=EPSG:26915)

        >>> ds.xvec.geom_coords
        Coordinates:
          * geom           (geom) object POINT (1 2) POINT (3 4)
          * geom_z         (geom_z) object POINT Z (10 20 30) POINT Z (30 40 50)
          * geom_no_index  (geom_no_index) object POINT (1 2) POINT (3 4)

        See also
        --------
        geom_coords_indexed
        is_geom_variable
        """
        return xr.Coordinates(
            coords={
                c: coo
                for c, coo in self._obj.coords.items()
                if c in self._geom_coords_all
            },
            indexes={
                c: self._obj.xindexes[c]
                for c in self._obj.coords
                if c in self._geom_coords_all and c in self._obj.xindexes
            },
        )

    @property
    def geom_coords_indexed(self) -> xr.Coordinates:
        """Returns a dictionary of xarray.DataArray objects corresponding to
        coordinate variables using :class:`~xvec.GeometryIndex`.

        Examples
        --------
        >>> ds = (
        ...     xr.Dataset(
        ...         coords={
        ...             "geom": np.array([shapely.Point(1, 2), shapely.Point(3, 4)]),
        ...             "geom_z": np.array(
        ...                 [shapely.Point(10, 20, 30), shapely.Point(30, 40, 50)]
        ...             ),
        ...             "geom_no_index": np.array(
        ...                 [shapely.Point(1, 2), shapely.Point(3, 4)]
        ...             ),
        ...         }
        ...     )
        ...     .xvec.set_geom_indexes(["geom", "geom_z"], crs=26915)
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:        (geom: 2, geom_z: 2, geom_no_index: 2)
        Coordinates:
          * geom           (geom) object POINT (1 2) POINT (3 4)
          * geom_z         (geom_z) object POINT Z (10 20 30) POINT Z (30 40 50)
          * geom_no_index  (geom_no_index) object POINT (1 2) POINT (3 4)
        Data variables:
            *empty*
        Indexes:
            geom           GeometryIndex (crs=EPSG:26915)
            geom_z         GeometryIndex (crs=EPSG:26915)

        >>> ds.xvec.geom_coords_indexed
        Coordinates:
          * geom     (geom) object POINT (1 2) POINT (3 4)
          * geom_z   (geom_z) object POINT Z (10 20 30) POINT Z (30 40 50)

        See also
        --------
        geom_coords
        is_geom_variable

        """
        return xr.Coordinates(
            coords={
                c: coo for c, coo in self._obj.coords.items() if c in self._geom_indexes
            },
            indexes={
                c: self._obj.xindexes[c]
                for c in self._obj.coords
                if c in self._geom_indexes and c in self._obj.xindexes
            },
        )

    def to_crs(
        self,
        variable_crs: Mapping[Any, Any] | None = None,
        **variable_crs_kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """
        Transform :class:`shapely.Geometry` objects of a variable to a new coordinate
        reference system.

        Returns a new object with all the original data in addition to the transformed
        variable. The CRS the current array must be set using
        :class:`~xvec.GeometryIndex`.

        This method will transform all points in all objects. It has no notion or
        projecting entire geometries. All segments joining points are assumed to be
        lines in the current projection, not geodesics. Objects crossing the dateline
        (or other projection boundary) will have undesirable behavior.

        Parameters
        ----------
        variable_crs : dict-like or None, optional
            A dict where the keys are the names of the coordinates and values target
            CRS in any format accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>` such
            as an authority string (e.g. ``"EPSG:4326"``), EPSG code (e.g. ``4326``) or
            a WKT string.
        **variable_crs_kwargs : optional
            The keyword arguments form of ``variable_crs``.
            One of ``variable_crs`` or ``variable_crs_kwargs`` must be provided.

        Returns
        -------
        assigned : same type as caller
            A new object with the variables transformed to target CRSs.

        See also
        --------
        set_crs

        Examples
        --------
        Transform coordinates backed by :class:`~xvec.GeometryIndex` from `EPSG:4326`
        to `ESPG:3857`.

        >>> da = (
        ...     xr.DataArray(
        ...         np.random.rand(2),
        ...         coords={"geom": [shapely.Point(1, 2), shapely.Point(3, 4)]},
        ...         dims="geom",
        ...     )
        ...     .xvec.set_geom_indexes("geom", crs=4326)
        ... )
        >>> da
        <xarray.DataArray (geom: 2)>
        array([0.47575118, 0.09271935])
        Coordinates:
          * geom     (geom) object POINT (1 2) POINT (3 4)
        Indexes:
            geom     GeometryIndex (crs=EPSG:4326)
        >>> da.xvec.to_crs(geom=3857)
        <xarray.DataArray (geom: 2)>
        array([0.47575118, 0.09271935])
        Coordinates:
          * geom     (geom) object POINT (111319.49079327357 222684.20850554405) POIN...
        Indexes:
            geom     GeometryIndex (crs=EPSG:3857)

        The same can be done using dictionary arguments.

        >>> da.xvec.to_crs({"geom": 3857})
        <xarray.DataArray (geom: 2)>
        array([0.47575118, 0.09271935])
        Coordinates:
        * geom     (geom) object POINT (111319.49079327357 222684.20850554405) POIN...
        Indexes:
            geom     GeometryIndex (crs=EPSG:3857)

        The same applies to a :class:`xarray.Dataset`.

        >>> ds = (
        ...     xr.Dataset(coords={"geom": [shapely.Point(1, 2), shapely.Point(3, 4)]})
        ...     .xvec.set_geom_indexes("geom", crs=4326)
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:  (geom: 2)
        Coordinates:
        * geom     (geom) object POINT (1 2) POINT (3 4)
        Data variables:
            *empty*
        Indexes:
            geom     GeometryIndex (crs=EPSG:4326)
        >>> ds.xvec.to_crs(geom=3857)
        <xarray.Dataset>
        Dimensions:  (geom: 2)
        Coordinates:
          * geom     (geom) object POINT (111319.49079327357 222684.20850554405) POIN...
        Data variables:
            *empty*
        Indexes:
            geom     GeometryIndex (crs=EPSG:3857)

        Notes
        -----
        Currently supports only :class:`xarray.Variable` objects that are set as
        coordinates with :class:`~xvec.GeometryIndex` assigned. The implementation
        currently wraps :meth:`Dataset.assign_coords <xarray.Dataset.assign_coords>`
        or :meth:`DataArray.assign_coords <xarray.DataArray.assign_coords>`.
        """
        variable_crs_solved = _resolve_input(
            variable_crs, variable_crs_kwargs, "to_crs"
        )

        _obj = self._obj.copy(deep=False)

        transformed = {}

        for key, crs in variable_crs_solved.items():
            if not isinstance(self._obj.xindexes[key], GeometryIndex):
                raise ValueError(
                    f"The index '{key}' is not an xvec.GeometryIndex. "
                    "Set the xvec.GeometryIndex using '.xvec.set_geom_indexes' before "
                    "handling projection information."
                )

            data = _obj[key].data
            data_crs = self._obj.xindexes[key].crs  # type: ignore

            # transformation code taken from geopandas (BSD 3-clause license)
            if data_crs is None:
                raise ValueError(
                    "Cannot transform naive geometries. "
                    f"Please set a CRS on the '{key}' coordinates first."
                )

            crs = CRS.from_user_input(crs)

            if data_crs.is_exact_same(crs):
                pass

            result = transform_geom(data, data_crs, crs)

            transformed[key] = (result, crs)

        for key, (result, _crs) in transformed.items():
            _obj = _obj.assign_coords({key: result})

        _obj = _obj.drop_indexes(variable_crs_solved.keys())

        for key, crs in variable_crs_solved.items():
            if crs:
                _obj[key].attrs["crs"] = CRS.from_user_input(crs)
            _obj = _obj.set_xindex([key], GeometryIndex, crs=crs)

        return _obj

    def set_crs(
        self,
        variable_crs: Mapping[Any, Any] | None = None,
        allow_override: bool = False,
        **variable_crs_kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """Set the Coordinate Reference System (CRS) of coordinates backed by
        :class:`~xvec.GeometryIndex`.

        Parameters
        ----------
        variable_crs : dict-like or None, optional
            A dict where the keys are the names of the coordinates and values target
            CRS in any format accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>` such
            as an authority string (e.g. ``"EPSG:4326"``), EPSG code (e.g. ``4326``) or
            a WKT string.
        allow_override : bool, default False
            If the :class:`~xvec.GeometryIndex` already has a CRS,
            allow to replace the existing CRS, even when both are not equal.
        **variable_crs_kwargs : optional
            The keyword arguments form of ``variable_crs``.
            One of ``variable_crs`` or ``variable_crs_kwargs`` must be provided.

        Returns
        -------
        assigned : same type as caller
            A new object with the assigned target CRS.

        See also
        --------
        to_crs

        Examples
        --------
        The method is the most useful, when there is no CRS assigned (illustrated on
        a :class:`xarray.Dataset` but the same is applicable on
        a :class:`xarray.DataArray`).

        >>> ds = (
        ...     xr.Dataset(coords={"geom": [shapely.Point(1, 2), shapely.Point(3, 4)]})
        ...     .xvec.set_geom_indexes("geom")
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:  (geom: 2)
        Coordinates:
          * geom     (geom) object POINT (1 2) POINT (3 4)
        Data variables:
            *empty*
        Indexes:
            geom     GeometryIndex (crs=None)
        >>> ds.xvec.set_crs(geom=4326)
        <xarray.Dataset>
        Dimensions:  (geom: 2)
        Coordinates:
          * geom     (geom) object POINT (1 2) POINT (3 4)
        Data variables:
            *empty*
        Indexes:
            geom     GeometryIndex (crs=EPSG:4326)

        It can also be used to overwrite the existing CRS. Note, that in most cases
        you probably want to use the :meth:`to_crs` instead is such a case.

        >>> ds = (
        ...     xr.Dataset(coords={"geom": [shapely.Point(1, 2), shapely.Point(3, 4)]})
        ...     .xvec.set_geom_indexes("geom", crs=4326)
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:  (geom: 2)
        Coordinates:
        * geom     (geom) object POINT (1 2) POINT (3 4)
        Data variables:
            *empty*
        Indexes:
            geom     GeometryIndex (crs=EPSG:4326)
        >>> ds.xvec.set_crs(geom=3857, allow_override=True)
        <xarray.Dataset>
        Dimensions:  (geom: 2)
        Coordinates:
        * geom     (geom) object POINT (1 2) POINT (3 4)
        Data variables:
            *empty*
        Indexes:
            geom     GeometryIndex (crs=EPSG:3857)

        See that only the CRS has changed, not the geometries.


        Notes
        -----
        The underlying geometries are not transformed to this CRS. To
        transform the geometries to a new CRS, use the :meth:`to_crs`
        method.
        """
        variable_crs_solved = _resolve_input(
            variable_crs, variable_crs_kwargs, "set_crs"
        )

        _obj = self._obj.copy(deep=False)

        for key, crs in variable_crs_solved.items():
            if not isinstance(self._obj.xindexes[key], GeometryIndex):
                raise ValueError(
                    f"The index '{key}' is not an xvec.GeometryIndex. "
                    "Set the xvec.GeometryIndex using '.xvec.set_geom_indexes' before "
                    "handling projection information."
                )

            data_crs = self._obj.xindexes[key].crs  # type: ignore

            if not allow_override and data_crs is not None and not data_crs == crs:
                raise ValueError(
                    f"The index '{key}' already has a CRS which is not equal to the "
                    "passed CRS. Specify 'allow_override=True' to allow replacing the "
                    "existing CRS without doing any transformation. If you actually "
                    "want to transform the geometries, use '.xvec.to_crs' instead."
                )

        _obj = _obj.drop_indexes(variable_crs_solved.keys())

        for key, crs in variable_crs_solved.items():
            if crs:
                _obj[key].attrs["crs"] = CRS.from_user_input(crs)
            _obj = _obj.set_xindex([key], GeometryIndex, crs=crs)

        return _obj

    def query(
        self,
        coord_name: str | None,
        geometry: shapely.Geometry | Sequence[shapely.Geometry],
        predicate: str | None = None,
        distance: float | Sequence[float] | None = None,
        unique: bool = False,
    ) -> xr.DataArray | xr.Dataset:
        """Return a subset of a DataArray/Dataset filtered using a spatial query on
        :class:`~xvec.GeometryIndex`.

        Return the subset where the bounding box of each input geometry intersects the
        bounding box of a geometry in an :class:`~xvec.GeometryIndex`. If a predicate is
        provided, the tree geometries are first queried based on the bounding box of the
        input geometry and then are further filtered to those that meet the predicate
        when comparing the input geometry to the tree geometry: predicate(geometry,
        index_geometry)

        Bounding boxes are limited to two dimensions and are axis-aligned (equivalent to
        the bounds property of a geometry); any Z values present in input geometries are
        ignored when querying the tree.


        Parameters
        ----------
        coord_name : str | None
            name of the coordinate axis backed by :class:`~xvec.GeometryIndex`. If None,
            the query is done against the DataArray.data assuming it contains shapely
            geometries.
        geometry : shapely.Geometry | Sequence[shapely.Geometry]
            Input geometries to query the :class:`~xvec.GeometryIndex` and filter
            results using the optional predicate.
        predicate : str, optional
            The predicate to use for testing geometries from the
            :class:`~xvec.GeometryIndex` that are within the input geometry’s bounding
            box, by default None. Any predicate supported by
            :meth:`shapely.STRtree.query` is a valid input.
        distance : float | Sequence[float], optional
            Distances around each input geometry within which to query the tree for the
            ``dwithin`` predicate. If array_like, shape must be broadcastable to shape
            of geometry. Required if ``predicate=’dwithin’``, by default None
        unique : bool, optional
            Keep only unique geometries from the :class:`~xvec.GeometryIndex` in a
            result. If False, index geometries that match the query for multiple input
            geometries are duplicated for each match. If False, such geometries are
            returned only once. By default False

        Returns
        -------
        filtered : same type as caller
            A new object filtered according to the query

        Examples
        --------
        >>> da = (
        ...     xr.DataArray(
        ...         np.random.rand(2),
        ...         coords={"geom": [shapely.Point(1, 2), shapely.Point(3, 4)]},
        ...         dims="geom",
        ...     )
        ...     .xvec.set_geom_indexes("geom", crs=4326)
        ... )
        >>> da
        <xarray.DataArray (geom: 2)>
        array([0.76385513, 0.2312171 ])
        Coordinates:
          * geom     (geom) object POINT (1 2) POINT (3 4)
        Indexes:
            geom     GeometryIndex (crs=EPSG:4326)

        >>> da.xvec.query("geom", shapely.box(0, 0, 2.4, 2.2))
        <xarray.DataArray (geom: 1)>
        array([0.76385513])
        Coordinates:
          * geom     (geom) object POINT (1 2)
        Indexes:
            geom     GeometryIndex (crs=EPSG:4326)

        >>> da.xvec.query(
        ...     "geom", [shapely.box(0, 0, 2.4, 2.2), shapely.Point(2, 2).buffer(1)]
        ... )
        <xarray.DataArray (geom: 2)>
        array([0.76385513, 0.76385513])
        Coordinates:
          * geom     (geom) object POINT (1 2) POINT (1 2)
        Indexes:
            geom     GeometryIndex (crs=EPSG:4326)

        >>> da.xvec.query(
        ...     "geom",
        ...     [shapely.box(0, 0, 2.4, 2.2), shapely.Point(2, 2).buffer(1)],
        ...     unique=True,
        ... )
        <xarray.DataArray (geom: 1)>
        array([0.76385513])
        Coordinates:
          * geom     (geom) object POINT (1 2)
        Indexes:
            geom     GeometryIndex (crs=EPSG:4326)

        """
        if isinstance(geometry, shapely.Geometry):
            ilocs = self._obj.xindexes[coord_name].sindex.query(  # type: ignore
                geometry, predicate=predicate, distance=distance
            )

        else:
            _, ilocs = self._obj.xindexes[coord_name].sindex.query(  # type: ignore
                geometry, predicate=predicate, distance=distance
            )
            if unique:
                ilocs = np.unique(ilocs)

        return self._obj.isel({coord_name: ilocs})

    def mask(
        self,
        geometry: shapely.Geometry | Sequence[shapely.Geometry],
        predicate: str | None = None,
        distance: float | Sequence[float] | None = None,
    ) -> xr.DataArray:
        """
        Return boolean array representing the outcome of spatial predicate query

        Take the DataArray containing variable geometry and return a mask matching
        the spaital predicate query.

        Parameters
        ----------
        geometry : shapely.Geometry or Sequence[shapely.Geometry]
            The geometry or sequence of geometries to use for masking.
        predicate : str, optional
            The spatial predicate to use for the query (e.g., 'intersects', 'contains'). Default is None.
        distance : float or Sequence[float], optional
            The distance or sequence of distances to use for the query. Default is None.

        Returns
        -------
        xr.DataArray
            A DataArray with the same shape as the original, where the elements matching the predicate are set to True.
        """
        # need to replace nan with None
        cube_data = self._obj.where(~self._obj.isnull(), None).data.ravel()
        tree = shapely.STRtree(cube_data)
        indices = tree.query(geometry, predicate=predicate, distance=distance)
        if indices.ndim == 1:
            dense = np.zeros(len(cube_data), dtype=bool)
            dense[indices] = True
        else:
            dense = np.zeros((len(cube_data), len(geometry)), dtype=bool)  # type: ignore
            tree, other = indices[::-1]
            dense[tree, other] = True
            dense = dense.any(axis=1)  # type: ignore
        return xr.DataArray(dense.reshape(self._obj.shape), coords=self._obj.coords)

    def set_geom_indexes(
        self,
        coord_names: str | Sequence[str],
        crs: Any = None,
        allow_override: bool = False,
        **kwargs: dict[str, Any],
    ) -> xr.DataArray | xr.Dataset:
        """Set a new  :class:`~xvec.GeometryIndex` for one or more existing
        coordinate(s). One :class:`~xvec.GeometryIndex` is set per coordinate. Only
        1-dimensional coordinates are supported.

        Parameters
        ----------
        coord_names : str or list
            Name(s) of the coordinate(s) used to build the index.
        crs : Any, optional
            CRS in any format accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>` such
            as an authority string (e.g. ``"EPSG:4326"``), EPSG code (e.g. ``4326``) or
            a WKT string.
        allow_override : bool, default False
            If the coordinate(s) already have a :class:`~xvec.GeometryIndex`,
            allow to replace the existing CRS, even when both are not equal.

        Returns
        -------
        assigned : same type as caller
            A new object with the same data and new index(es)

        Examples
        --------
        >>> da = (
        ...     xr.DataArray(
        ...         np.random.rand(2),
        ...         coords={"geom": [shapely.Point(1, 2), shapely.Point(3, 4)]},
        ...         dims="geom",
        ...     )
        ... )
        >>> da
        <xarray.DataArray (geom: 2)>
        array([0.06610343, 0.03144603])
        Coordinates:
          * geom     (geom) object POINT (1 2) POINT (3 4)

        >>> da.xvec.set_geom_indexes("geom", crs=4326)
        <xarray.DataArray (geom: 2)>
        array([0.06610343, 0.03144603])
        Coordinates:
          * geom     (geom) object POINT (1 2) POINT (3 4)
        Indexes:
            geom     GeometryIndex (crs=EPSG:4326)
        """
        _obj = self._obj.copy(deep=False)

        if isinstance(coord_names, str):
            coord_names = [coord_names]

        for coord in coord_names:
            if isinstance(self._obj.xindexes[coord], GeometryIndex):
                data_crs = self._obj.xindexes[coord].crs  # type: ignore

                if not allow_override and data_crs is not None and not data_crs == crs:
                    raise ValueError(
                        f"The index '{coord}' already has a CRS which is not equal to "
                        "the passed CRS. Specify 'allow_override=True' to allow "
                        "replacing the existing CRS without doing any transformation. "
                        "If you actually want to transform the geometries, use "
                        "'.xvec.to_crs' instead."
                    )
        _obj = _obj.drop_indexes(coord_names)

        for coord in coord_names:
            if crs:
                _obj[coord].attrs["crs"] = CRS.from_user_input(crs)
            _obj = _obj.set_xindex(coord, GeometryIndex, crs=crs, **kwargs)

        return _obj

    def to_geopandas(self) -> GeoDataFrame | pd.DataFrame:
        """Convert this array into a GeoPandas :class:`~geopandas.GeoDataFrame`

        Returns a :class:`~geopandas.GeoDataFrame` with coordinates based on a
        :class:`~xvec.GeometryIndex` set as an active geometry. The array needs to
        contain only a single geometry-based coordinate axis and can be 1D or 2D. Other
        cases are not supported.

        Unlike :meth:`~xarray.DataArray.to_pandas`, this always returns a
        :class:`~geopandas.GeoDataFrame` with a single geometry column and one
        (1D array) or multiple (2D array) variable columns.

        Only works for arrays with 2 or fewer dimensions.

        Returns
        -------
        GeoDataFrame | DataFrame
            :class:`~geopandas.GeoDataFrame` with coordinates based on
            :class:`~xvec.GeometryIndex` set as an active geometry
            or pandas DataFrame with GeometryArrays

        See also
        --------
        to_geodataframe
        """
        try:
            import geopandas as gpd
        except ImportError as err:
            raise ImportError(
                "The geopandas package is required for `xvec.to_geodataframe()`. "
                "You can install it using 'conda install -c conda-forge geopandas' or "
                "'pip install geopandas'."
            ) from err

        if isinstance(self._obj, xr.DataArray) and self._obj.ndim > 2:
            raise ValueError(
                f"Cannot convert arrays with {self._obj.ndim} dimensions into pandas "
                "objects. Requires 2 or fewer dimensions."
            )

        if len(self._geom_indexes) > 1:
            raise ValueError(
                "Multiple coordinates based on xvec.GeometryIndex are not supported "
                "as GeoPandas.GeoDataFrame cannot be indexed by geometry. Try using "
                "`.xvec.to_geodataframe()` instead."
            )

        # DataArray
        if isinstance(self._obj, xr.DataArray):
            if len(self._geom_indexes):
                if self._obj.ndim == 1:
                    gdf = self._obj.to_pandas()
                else:
                    gdf = self._obj.to_pandas()
                    if gdf.columns.name == self._geom_indexes[0]:
                        gdf = gdf.T
                return gdf.reset_index().set_geometry(  # type: ignore
                    self._geom_indexes[0],
                    crs=self._obj.xindexes[self._geom_indexes[0]].crs,  # type: ignore
                )
            warnings.warn(
                "No geometry to return, falling back to DataArray.to_pandas().",
                UserWarning,
                stacklevel=2,
            )
            return self._obj.to_pandas()

        # Dataset
        gdf = self._obj.to_pandas()

        # ensure CRS of all columns is preserved
        for c in gdf.columns:
            if c in self._geom_coords_all:
                gdf[c] = gpd.GeoSeries(gdf[c], crs=self._obj[c].attrs.get("crs", None))

        # if geometry is an index, reset and assign as active
        index_name = gdf.index.name
        if index_name in self._geom_coords_all:
            return gdf.reset_index().set_geometry(
                index_name, crs=self._obj[index_name].attrs.get("crs", None)
            )  # type: ignore

        warnings.warn(
            "No active geometry column to be set. The resulting object "
            "will be a pandas.DataFrame with geopandas.GeometryArray(s) containing "
            "geometry and CRS information. Use `.set_geometry()` to set an active "
            "geometry and upcast to the geopandas.GeoDataFrame manually.",
            UserWarning,
            stacklevel=2,
        )

        return gdf

    def to_geodataframe(
        self,
        *,
        name: Hashable | None = None,
        dim_order: Sequence[Hashable] | None = None,
        geometry: Hashable | None = None,
        long: bool = True,
    ) -> GeoDataFrame | pd.DataFrame:
        """Convert this array and its coordinates into a tidy geopandas.GeoDataFrame.

        The GeoDataFrame is indexed by the Cartesian product of index coordinates
        (in the form of a :class:`pandas.MultiIndex`) excluding geometry coordinates.
        Other coordinates are included as columns in the GeoDataFrame.
        For 1D and 2D DataArrays, see also :meth:`to_geopandas` which
        doesn't rely on a MultiIndex to build the GeoDataFrame.

        Geometry coordinates are forcibly removed from the index and stored as columns.

        If there is only a single geometry coordinate axis, it is set as active
        geometry of the GeoDataFrame. If there are multiple or none in coordinates,
        ``geometry`` must be passed to set an active geometry column.

        Parameters
        ----------
        name : Hashable or None, optional
            Name to give to this array (required if unnamed).
            Applies only if the object is DataArray, otherwise ignored.
        dim_order : Sequence of Hashable or None, optional
            Hierarchical dimension order for the resulting dataframe.
            Array content is transposed to this order and then written out as flat
            vectors in contiguous order, so the last dimension in this list
            will be contiguous in the resulting DataFrame. This has a major
            influence on which operations are efficient on the resulting
            dataframe.
            If provided, must include all dimensions of this DataArray. By default,
            dimensions are sorted according to the DataArray dimensions order.
        geometry : Hashable, optional
            A key of a geometry coordinates to be used as an active geometry of the
            resulting GeoDataFrame.
        long : Bool, optional (default True)
            A form of the table. If True, creates a long form as
            ``DataArray.to_dataframe``. If False, creates a wide form with MultiIndex
            columns (as if you would call ``.unstack()``). A wide form supports objects
            with only a single dimension of geometries.

        Returns
        -------
        GeoDataFrame | DataFrame
            GeoPandas GeoDataFrame with coordinates based on GeometryIndex
            set as an active geometry or pandas DataFrame with GeometryArrays

        See also
        --------
        to_geopandas
        """

        try:
            import geopandas as gpd
        except ImportError as err:
            raise ImportError(
                "The geopandas package is required for `xvec.to_geodataframe()`. "
                "You can install it using 'conda install -c conda-forge geopandas' or "
                "'pip install geopandas'."
            ) from err

        if isinstance(self._obj, xr.Dataset):
            df = self._obj.to_dataframe(dim_order=dim_order)
        else:
            df = self._obj.to_dataframe(name=name, dim_order=dim_order)

        if not long:
            if len(self._geom_coords_all) != 1:
                raise ValueError(
                    "Creating a wide form GeoDataFrame with `long=False` requires "
                    "exactly one dimension with a GeometryIndex."
                )
            df = df.unstack(
                [
                    level
                    for level in df.index.names
                    if level not in self._geom_coords_all
                ]  # type: ignore
            )

        if isinstance(df.index, pd.MultiIndex):
            to_reset = [g for g in self._geom_coords_all if g in df.index.names]
            if to_reset:
                df = df.reset_index(to_reset)
            if len(to_reset) == 1 and geometry is None:
                geometry = to_reset[0]
        else:
            if df.index.name in self._geom_coords_all:
                if geometry is None:
                    geometry = df.index.name
                df = df.reset_index()
        # ensure CRS of all columns is preserved
        for c in df.columns:
            if c in self._geom_coords_all:
                # there is a geopandas bug that does not allow upcasting a series
                # with a geometry dtype directly
                df[c] = gpd.GeoSeries(
                    df[c].values,
                    crs=self._obj[c].attrs.get("crs", None),
                    index=df[c].index,
                )

        if geometry is not None:
            if geometry not in self._geom_coords_all:  # variable geometry
                return df.set_geometry(geometry, crs=self._obj.proj.crs)

            # coordinate geometry
            return df.set_geometry(
                geometry, crs=self._obj[geometry].attrs.get("crs", None)
            )  # type: ignore

        # check if the geometry comes from the values, rather than an index
        name = (
            name if name else (self._obj.name if hasattr(self._obj, "name") else None)
        )
        if name is not None and shapely.is_valid_input(df[name]).all():
            return df.set_geometry(name, crs=self._obj.proj.crs)

        warnings.warn(
            "No active geometry column to be set. The resulting object "
            "will be a pandas.DataFrame with geopandas.GeometryArray(s) containing "
            "geometry and CRS information. Use `.set_geometry()` to set an active "
            "geometry and upcast to the geopandas.GeoDataFrame manually.",
            UserWarning,
            stacklevel=2,
        )
        return df

    def zonal_stats(
        self,
        geometry: Sequence[shapely.Geometry] | xr.DataArray,
        x_coords: Hashable,
        y_coords: Hashable,
        stats: str | Callable | Sequence[str | Callable | tuple] = "mean",
        name: str = "geometry",
        index: bool | None = None,
        method: str | None = None,
        all_touched: bool = False,
        n_jobs: int = -1,
        nodata: Any = None,
        strategy: Literal[
            "feature-sequential", "raster-sequential"
        ] = "raster-sequential",
        **kwargs: dict[str, Any],
    ) -> xr.DataArray | xr.Dataset:
        """Extract the values from a dataset indexed by a set of geometries

        Given an object indexed by x and y coordinates (or latitude and longitude), such
        as an typical geospatial raster dataset, aggregate multidimensional data for a
        set of Polygons or LineStrings represented as shapely geometry.

        The CRS of the raster and that of geometry need to be equal.
        Xvec does not verify their equality.

        Requires ``rioxarray``.

        Parameters
        ----------
        geometry : Sequence[shapely.Geometry] | xr.DataArray
            An arrray-like (1-D) of shapely geometries, like a numpy array or
            :class:`geopandas.GeoSeries` or xr.DataArray holding variable geometry.
            Polygon and LineString geometry types are supported.
        x_coords : Hashable
            name of the coordinates containing ``x`` coordinates (i.e. the first value
            in the coordinate pair encoding the vertex of the polygon)
        y_coords : Hashable
            name of the coordinates containing ``y`` coordinates (i.e. the second value
            in the coordinate pair encoding the vertex of the polygon)
        stats : string | Callable | Sequence[str | Callable | tuple]
            Spatial aggregation statistic method, by default ``"mean"``.

            Any of the aggregations available as :class:`xarray.DataArray` or
            :class:`~xarray.core.groupby.DataArrayGroupBy` methods like
            :meth:`~xarray.DataArray.mean`, :meth:`~xarray.DataArray.min`,
            :meth:`~xarray.DataArray.max`, or :meth:`~xarray.DataArray.quantile` are
            available. Alternatively, you can pass a ``Callable`` supported by
            :meth:`~xarray.DataArray.reduce` or a list with ``strings``, ``callables``
            or ``tuples`` in a ``(name, func, {kwargs})`` format, where ``func`` can be
            a string or a callable.

            If the method is ``"exactextract"`` then the stats should be string or list
            of strings that can be used to construct :class:`exactextract.Operation`
            objects supported by :func:`exactextract.exact_extract` (e.g., ``"mean"``,
            ``"quantile(q=0.20)"``).
        name : str, optional
            Name of the dimension that will hold the ``geometry``, by default "geometry"
        index : bool, optional
            If ``geometry`` is a :class:`~geopandas.GeoSeries`, ``index=True`` will
            attach its index as another coordinate to the geometry dimension in the
            resulting object. If ``index=None``, the index will be stored if the
            `geometry.index` is a named or non-default index. If ``index=False``, it
            will never be stored. This is useful as an attribute link between the
            resulting array and the GeoPandas object from which the geometry is sourced.
        method : str, optional
            The method of data extraction.

            ``"rasterize"``
                uses :func:`rasterio.features.rasterize` and is faster, but can lead to
                loss of information in case of small polygons or lines. Not supported
                for zonal stats using variable geometry (n-D array of geometry).

            ``"iterate"``
                iterates over geometries and uses
                :func:`rasterio.features.geometry_mask`. Requires ``joblib`` on top of
                ``rioxarray``.

            ``"exactextract"``
                calculates precise stats by determining the fraction of each raster cell
                that is covered by the polygon and uses
                :func:`exactextract.exact_extract`.

            The default is selected based on the availability of engines in the order
            of priority 1. ``"exactextract"``, 2. ``"rasterize"`` 3. ``"iterate"``.
        all_touched : bool, optional
            If True, all pixels touched by geometries will be considered. If False, only
            pixels whose center is within the polygon or that are selected by
            Bresenham’s line algorithm will be considered. Applies only if ``method="iterate"``
            or ``method="rasterize"``.
        n_jobs : int, optional
            Number of parallel threads to use. It is recommended to set this to the
            number of physical cores of the CPU. ``-1`` uses all available cores.
            Applies only if ``method="iterate"``.
        nodata : Any
            Value representing missing data. If not specified, the value is included in
            the aggregation.
        strategy : str, optional
            The strategy for ``exactextract`` method to use for the extraction, by
            default "raster-sequential". Use either "feature-sequential" and
            "raster-sequential". See :func:`exactextract.exact_extract` for details.
        **kwargs : optional
            Keyword arguments to be passed to the aggregation function
            (e.g., ``Dataset.quantile(**kwargs)``).

        Returns
        -------
        Dataset or DataArray
            A subset of the original object with N-1 dimensions indexed by
            the :class:`GeometryIndex` of ``geometry``.

        Examples
        --------
        >>> import geodatasets
        >>> import geopandas as gpd

        A typical raster Dataset indexed by longitude and latitude:

        >>> ds = xr.tutorial.open_dataset("eraint_uvz")
        >>> ds
        <xarray.Dataset>
        Dimensions:    (longitude: 480, latitude: 241, level: 3, month: 2)
        Coordinates:
        * longitude  (longitude) float32 -180.0 -179.2 -178.5 ... 177.8 178.5 179.2
        * latitude   (latitude) float32 90.0 89.25 88.5 87.75 ... -88.5 -89.25 -90.0
        * level      (level) int32 200 500 850
        * month      (month) int32 1 7
        Data variables:
            z          (month, level, latitude, longitude) float64 ...
            u          (month, level, latitude, longitude) float64 ...
            v          (month, level, latitude, longitude) float64 ...
        Attributes:
            Conventions:  CF-1.0
            Info:         Monthly ERA-Interim data. Downloaded

        A set of polygons representing land mass:

        >>> world = gpd.read_file(geodatasets.get_path("naturalearth land"))
        >>> world
            featurecla  ...                                           geometry
        0         Land  ...  POLYGON ((-59.57209 -80.04018, -59.86585 -80.5...
        1         Land  ...  POLYGON ((-159.20818 -79.49706, -161.12760 -79...
        2         Land  ...  POLYGON ((-45.15476 -78.04707, -43.92083 -78.4...
        3         Land  ...  POLYGON ((-121.21151 -73.50099, -119.91885 -73...
        4         Land  ...  POLYGON ((-125.55957 -73.48135, -124.03188 -73...
        ..         ...  ...                                                ...
        122       Land  ...  POLYGON ((51.13619 80.54728, 49.79368 80.41543...
        123       Land  ...  POLYGON ((99.93976 78.88094, 97.75794 78.75620...
        124       Land  ...  POLYGON ((-87.02000 79.66000, -85.81435 79.336...
        125       Land  ...  POLYGON ((-68.50000 83.10632, -65.82735 83.028...
        126       Land  ...  POLYGON ((-27.10046 83.51966, -20.84539 82.726...

        [127 rows x 4 columns]

        Dataset with N-1 dimensions indexed by the geometry aggregated using mean:

        >>> ds.xvec.zonal_stats(world.geometry, "longitude", "latitude")
        <xarray.Dataset>
        Dimensions:   (level: 3, month: 2, geometry: 127)
        Coordinates:
        * level     (level) int32 200 500 850
        * month     (month) int32 1 7
        * geometry  (geometry) object POLYGON ((-59.57209469261153 -80.040178725096...
        Data variables:
            z         (geometry, month, level) float64 1.1e+05 5.025e+04 ... 1.394e+04
            u         (geometry, month, level) float64 2.401 1.482 ... 2.393 0.8898
            v         (geometry, month, level) float64 0.4296 0.07286 ... 1.116 0.6399
        Indexes:
            geometry  GeometryIndex (crs=EPSG:4326)
        Attributes:
            Conventions:  CF-1.0
            Info:         Monthly ERA-Interim data. Downloaded and edited by fabien.m...

        Notes
        -----

        See the `User Guide <../zonal_stats.html>`_ for detailed explanation of the
        functionality.

        See also
        --------
        extract_points : extraction of values for the raster object for points
        """

        if isinstance(geometry, xr.DataArray) and len(geometry.dims) > 1:
            if method is None:
                method = _get_method(variable=True)

            if method == "iterate":
                return _variable_zonal(
                    self,
                    variable_geometry=geometry,
                    x_coords=x_coords,
                    y_coords=y_coords,
                    stats=stats,
                    all_touched=all_touched,
                    n_jobs=n_jobs,
                    nodata=nodata,
                )
            if method == "exactextract":
                return _variable_zonal_exactextract(
                    self,
                    geometry=geometry,
                    x_coords=x_coords,
                    y_coords=y_coords,
                    stats=stats,
                    nodata=nodata,
                    strategy=strategy,
                )
            raise ValueError(
                f"Method '{method}' is not supported for zonal statistics based on "
                "variable geometry. Use one of `exactextract` or `iterate`."
            )

        if method is None:
            method = _get_method(variable=False)

        if method == "rasterize":
            result = _zonal_stats_rasterize(
                self,
                geometry=geometry,
                x_coords=x_coords,
                y_coords=y_coords,
                stats=stats,
                name=name,
                all_touched=all_touched,
                nodata=nodata,
                **kwargs,
            )
        elif method == "iterate":
            result = _zonal_stats_iterative(
                self,
                geometry=geometry,
                x_coords=x_coords,
                y_coords=y_coords,
                stats=stats,
                name=name,
                all_touched=all_touched,
                n_jobs=n_jobs,
                nodata=nodata,
                **kwargs,
            )
        elif method == "exactextract":
            result = _zonal_stats_exactextract(
                self,
                geometry=geometry,
                x_coords=x_coords,
                y_coords=y_coords,
                stats=stats,
                name=name,
                nodata=nodata,
                strategy=strategy,
                **kwargs,
            )
        else:
            raise ValueError(
                f"method '{method}' is not supported. Allowed options are 'rasterize' "
                "and 'iterate'."
            )

        # save the index as a data variable
        if isinstance(geometry, pd.Series):
            if index is None:
                if geometry.index.name is not None or not geometry.index.equals(
                    pd.RangeIndex(0, len(geometry))
                ):
                    index = True
            if index:
                index_name = geometry.index.name if geometry.index.name else "index"
                result = result.assign_coords({index_name: (name, geometry.index)})

        # standardize the shape - each method comes with a different one
        return result.transpose(name, ...)

    def extract_points(
        self,
        points: Sequence[shapely.Geometry],
        x_coords: Hashable,
        y_coords: Hashable,
        tolerance: float | None = None,
        name: str = "geometry",
        crs: Any | None = None,
        index: bool | None = None,
    ) -> xr.DataArray | xr.Dataset:
        """Extract points from a DataArray or a Dataset indexed by spatial coordinates

        Given an object indexed by x and y coordinates (or latitude and longitude), such
        as an typical geospatial raster dataset, extract multidimensional data for a
        set of points represented as shapely geometry.

        The CRS of the raster and that of points need to match. Xvec does not verify
        their equality.

        Parameters
        ----------
        points : Sequence[shapely.Geometry]
            An arrray-like (1-D) of shapely geometries, like a numpy array or GeoPandas
            GeoSeries.
        x_coords : Hashable
            name of the coordinates containing ``x`` coordinates (i.e. the first value
            in the coordinate pair encoding the point)
        y_coords : Hashable
            name of the coordinates containing ``y`` coordinates (i.e. the second value
            in the coordinate pair encoding the point)
        tolerance : float | None, optional
            Maximum distance between original and new labels for inexact matches.
            The values of the index at the matching locations must satisfy the equation
            ``abs(index[indexer] - target) <= tolerance``, by default ``None``.
        name : Hashable, optional
            Name of the dimension that will hold the ``points``, by default "geometry"
        crs : Any, optional
            Cordinate reference system of shapely geometries. If ``points`` have a
            ``.crs`` attribute (e.g. ``geopandas.GeoSeries`` or a ``DataArray`` with
            ``"crs"`` in ``.attrs``), ``crs`` will be automatically inferred. For more
            generic objects (numpy  array, list), CRS shall be specified manually.
        index : bool, optional
            If `points` is a GeoSeries, ``index=True`` will attach its index as another
            coordinate to the geometry dimension in the resulting object. If
            ``index=None``, the index will be stored if the `points.index` is a named
            or non-default index. If ``index=False``, it will never be stored. This is
            useful as an attribute link between the resulting array and the GeoPandas
            object from which the points are sourced.

        Returns
        -------
        DataArray or Dataset
            A subset of the original object with N-1 dimensions indexed by
            the array of points.

        Examples
        --------
        A typical raster Dataset indexed by longitude and latitude:

        >>> ds = xr.tutorial.open_dataset("eraint_uvz")
        >>> ds
        <xarray.Dataset>
        Dimensions:    (longitude: 480, latitude: 241, level: 3, month: 2)
        Coordinates:
        * longitude  (longitude) float32 -180.0 -179.2 -178.5 ... 177.8 178.5 179.2
        * latitude   (latitude) float32 90.0 89.25 88.5 87.75 ... -88.5 -89.25 -90.0
        * level      (level) int32 200 500 850
        * month      (month) int32 1 7
        Data variables:
            z          (month, level, latitude, longitude) float64 ...
            u          (month, level, latitude, longitude) float64 ...
            v          (month, level, latitude, longitude) float64 ...
        Attributes:
            Conventions:  CF-1.0
            Info:         Monthly ERA-Interim data. Downloaded

        Set of points representing locations you want to extract:

        >>> points = shapely.points(
        ...     np.random.uniform(ds.longitude.min(), ds.longitude.max(), 10),
        ...     np.random.uniform(ds.latitude.min(), ds.latitude.max(), 10),
        ... )

        Dataset with N-1 dimensions indexed by the geometry:

        >>> ds.xvec.extract_points(points, "longitude", "latitude", crs=4326)
        <xarray.Dataset>
        Dimensions:   (level: 3, month: 2, geometry: 10)
        Coordinates:
        * level     (level) int32 200 500 850
        * month     (month) int32 1 7
        * geometry  (geometry) object POINT (100.98750049682788 25.66910238029458) ...
        Data variables:
            z         (month, level, geometry) float64 ...
            u         (month, level, geometry) float64 ...
            v         (month, level, geometry) float64 ...
        Indexes:
            geometry  GeometryIndex (crs=EPSG:4326)
        Attributes:
            Conventions:  CF-1.0
            Info:         Monthly ERA-Interim data. Downloaded and edited by fabien.m...

        Notes
        -----

        See the `User Guide <../extract_pts.html>`_ for detailed explanation of the
        functionality.

        See also
        --------
        zonal_stats : zonal statistics for polygons and linestrings
        """
        if crs is None and hasattr(points, "crs"):
            crs = points.crs

        coords = shapely.get_coordinates(points)
        x_ = xr.DataArray(coords[:, 0], dims=name)
        y_ = xr.DataArray(coords[:, 1], dims=name)
        subset = self._obj.sel(
            {x_coords: x_, y_coords: y_}, method="nearest", tolerance=tolerance
        )

        subset[name] = (name, np.asarray(points))
        result = subset.drop_vars([x_coords, y_coords]).xvec.set_geom_indexes(
            name, crs=crs
        )

        # save the index as a data variable
        if isinstance(points, pd.Series):
            if index is None:
                if points.index.name is not None or not points.index.equals(
                    pd.RangeIndex(0, len(points))
                ):
                    index = True
            if index:
                index_name = points.index.name if points.index.name else "index"
                result = result.assign_coords({index_name: (name, points.index)})

        # preserve additional DataArray coords
        elif isinstance(points, xr.DataArray):
            if len(points.coords) > 1:
                result = result.assign_coords(
                    {
                        coo: (name, points[coo].data)
                        for coo in points.coords
                        if coo != name
                    }
                )
        return result

    def encode_cf(self) -> xr.Dataset:
        """
        Encode all geometry variables and associated CRS with CF conventions.

        Use this method prior to writing an Xarray dataset to any array format
        (e.g. netCDF or Zarr).

        The following invariant is satisfied:
            ``assert ds.xvec.encode_cf().xvec.decode_cf().identical(ds) is True``

        CRS information on the ``GeometryIndex`` is encoded using CF's ``grid_mapping`` convention.

        This function uses ``cf_xarray.geometry.encode_geometries`` under the hood and will only
        work on Datasets.

        Returns
        -------
        Dataset
        """
        import cf_xarray as cfxr

        if not isinstance(self._obj, xr.Dataset):
            raise ValueError(
                "CF encoding is only valid on Datasets. Convert to a dataset using `.to_dataset()` first."
            )

        ds = self._obj.copy()
        coords = self.geom_coords_indexed

        # TODO: this could use geoxarray, but is quite simple in any case
        # Adapted from rioxarray
        # 1. First find all unique CRS objects
        # preserve ordering for roundtripping
        unique_crs = []
        for _, xi in sorted(coords.xindexes.items()):
            if xi.crs not in unique_crs:
                unique_crs.append(xi.crs)
        if len(unique_crs) == 1:
            grid_mappings = {unique_crs.pop(): "spatial_ref"}
        else:
            grid_mappings = {
                crs_: f"spatial_ref_{i}" for i, crs_ in enumerate(unique_crs)
            }

        # 2. Convert CRS to grid_mapping variables and assign them
        for crs, grid_mapping in grid_mappings.items():
            grid_mapping_attrs = crs.to_cf()
            # TODO: not all CRS can be represented by CF grid_mappings
            # For now, we allow this.
            # if "grid_mapping_name" not in grid_mapping_attrs:
            #     raise ValueError
            wkt_str = crs.to_wkt()
            grid_mapping_attrs["spatial_ref"] = wkt_str
            grid_mapping_attrs["crs_wkt"] = wkt_str
            ds.coords[grid_mapping] = xr.Variable(
                dims=(), data=0, attrs=grid_mapping_attrs
            )

        # 3. Associate other variables with appropriate grid_mapping variable
        #    We asumme that this relation follows from dimension names being shared between
        #    the GeometryIndex and the variable being checked.
        for name, coord in coords.items():
            dims = set(coord.dims)
            index = coords.xindexes[name]
            varnames = (k for k, v in ds._variables.items() if dims & set(v.dims))
            for name in varnames:
                if TYPE_CHECKING:
                    assert isinstance(index, GeometryIndex)
                ds._variables[name].attrs["grid_mapping"] = grid_mappings[index.crs]

        encoded = cfxr.geometry.encode_geometries(ds)
        return encoded

    def decode_cf(self) -> xr.Dataset:
        """
        Decode geometries stored as CF-compliant arrays to shapely geometries.

        The following invariant is satisfied:
            ``assert ds.xvec.encode_cf().xvec.decode_cf().identical(ds) is True``


        A ``GeometryIndex`` is created automatically and CRS information, if available
        following CF's ``grid_mapping`` convention, will be associated with the ``GeometryIndex``.

        This function uses ``cf_xarray.geometry.decode_geometries`` under the hood, and will only
        work on Datasets.

        Returns
        -------
        Dataset
        """
        import cf_xarray as cfxr

        if not isinstance(self._obj, xr.Dataset):
            raise ValueError(
                "CF decoding is only supported on Datasets. Convert to a Dataset using `.to_dataset()` first."
            )

        decoded = cfxr.geometry.decode_geometries(self._obj.copy())
        crs = {
            name: CRS.from_user_input(var.attrs["crs_wkt"])
            for name, var in decoded._variables.items()
            if "crs_wkt" in var.attrs or "grid_mapping_name" in var.attrs
        }
        dims = decoded.xvec.geom_coords.dims
        for dim in dims:
            decoded = (
                decoded.set_xindex(dim) if dim not in decoded._indexes else decoded
            )
            decoded = decoded.xvec.set_geom_indexes(
                dim, crs=crs.get(decoded[dim].attrs.get("grid_mapping", None))
            )
        for name in crs:
            # remove spatial_ref so the coordinate system is only stored on the index
            del decoded[name]
        for var in decoded._variables.values():
            if set(dims) & set(var.dims):
                var.attrs.pop("grid_mapping", None)
        return decoded

    def encode_wkb(self) -> xr.DataArray | xr.Dataset:
        """
        Encode geometries to Well-Known Binary (WKB) format.

        This method converts all geometry coordinates and data variables in the
        DataArray or Dataset to WKB format. CRS information is stored in the
        attributes of the encoded geometries as PROJJSON. CRS information on variable
        geometry is assumed to be stored using the ``xproj`` package. Additionally, the
        ``wkb_encoded_geometry`` attribute equal to ``True`` is added to avoid ambiguity
        during the decoding step.

        Returns
        -------
        xr.DataArray or xr.Dataset
            A new object with geometries encoded in WKB format.
        """
        # process coordinate geometries
        obj = self._obj.assign_coords(
            {coord: shapely.to_wkb(self._obj[coord]) for coord in self._geom_coords_all}
        )
        if isinstance(obj, xr.DataArray):
            if np.all(shapely.is_valid_input(obj.data)):
                obj = shapely.to_wkb(obj)
                if obj.proj.crs:
                    obj.attrs["crs"] = obj.proj.crs.to_json()
                obj.attrs["wkb_encoded_geometry"] = True

        else:
            for data in obj.data_vars:
                if np.all(shapely.is_valid_input(obj[data].data)):
                    obj[data] = shapely.to_wkb(obj[data])
                    if obj[data].proj.crs:
                        obj[data].attrs["crs"] = obj[data].proj.crs.to_json()
                    obj[data].attrs["wkb_encoded_geometry"] = True

        for coord in self._geom_coords_all:
            if hasattr(obj[coord], "crs"):
                obj[coord].attrs["crs"] = obj[coord].crs.to_json()
            obj[coord].attrs["wkb_encoded_geometry"] = True

        return obj

    def decode_wkb(self) -> xr.DataArray | xr.Dataset:
        """
        Decode geometries from Well-Known Binary (WKB) format.

        This method converts all geometry coordinates and data variables in the
        DataArray or Dataset from WKB format back to shapely geometries. CRS information
        is restored from the attributes of the encoded geometries. CRS information on variable
        geometry will be stored using the ``xproj`` package. The
        ``wkb_encoded_geometry=True`` atrribute needs to be present at every variable
        that is supposed to be decoded.

        Returns
        -------
        xr.DataArray or xr.Dataset
            A new object with geometries decoded from WKB format to shapely geometries.
        """
        obj = self._obj.copy()

        if isinstance(obj, xr.DataArray):
            if obj.attrs.get("wkb_encoded_geometry", False):
                obj.data = shapely.from_wkb(obj)
                if "crs" in obj.attrs:
                    obj = obj.proj.assign_crs(
                        spatial_ref=json.loads(obj.attrs.pop("crs")),
                        allow_override=True,
                    )
                del obj.attrs["wkb_encoded_geometry"]

        else:
            for data in obj.data_vars:
                if obj[data].attrs.get("wkb_encoded_geometry", False):
                    obj[data].data = shapely.from_wkb(obj[data])
                    if "crs" in obj[data].attrs:
                        obj = obj.proj.assign_crs(
                            spatial_ref=json.loads(obj[data].attrs.pop("crs")),
                            allow_override=True,
                        )
                    del obj[data].attrs["wkb_encoded_geometry"]

        for coord in obj.coords:
            if obj[coord].attrs.get("wkb_encoded_geometry", False):
                crs = obj[coord].attrs.pop("crs", None)
                obj = obj.assign_coords({coord: shapely.from_wkb(obj[coord]).data})
                obj = obj.xvec.set_geom_indexes(coord, crs=crs)

        return obj

    def summarize_geometry(
        self,
        dim: str
        | xr.DataArray
        | xr.IndexVariable
        | Sequence[Hashable]
        | Mapping[Any, xr.groupers.Grouper],
        geom_array: Hashable | None = None,
        aggfunc: str | Callable = "envelope",
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """
        Summarize the geometry of an variable geometry along a specified dimension.

        Given a DataArray of variable geometry, summary geometry captures the overall
        spatial extent of a series of individual geometry objects along a single
        dimension. A typical use case would be a summary geometry of a moving object
        based on its ID.

        Parameters
        ----------
        dim : Hashable
            The dimension along which to summarize the geometry.
        geom_array : Hashable, optional
            The name of the geometry array to use for xr.Dataset objects. Must be
            specified for a xr.Dataset, ignored for a xr.DataArray.
        aggfunc : str or Callable, default "envelope"
            The aggregation function to use for summarizing the geometry. Can be one of the following strings:

            - ``"envelope"``: Computes the envelope (bounding box) of the geometries.
            - ``"centroid"``: Computes the centroid of the geometries.
            - ``"oriented_envelope"``: Computes the oriented envelope of the geometries.
            - ``"convex_hull"``: Computes the convex hull of the geometries.
            - ``"concave_hull"``: Computes the concave hull of the geometries. Additional parameters can be passed via ``kwargs``.
            - ``"collection"``: Collects the geometries into a geometry collection.
            - ``"union"``: Computes the union of the geometries.

            Or a callable that takes an ``xr.DataArray`` and returns an scalar ``xr.DataArray`` with shapely geometry.
        **kwargs : Any
            Additional keyword arguments to pass to the aggregation function if it is callable.

        Returns
        -------
        xr.DataArray or xr.Dataset
            The original xarray DataArray or Dataset with a new coordinate geometry
            ``summary_geometry`` with a ``GeometryIndex`` linked to a given dimension.
        """
        if isinstance(self._obj, xr.Dataset):
            if geom_array is None:
                raise ValueError("geom_array must be specified for xr.Dataset objects.")
            obj = self._obj[geom_array]
        else:
            obj = self._obj

        def _collect(x: xr.DataArray) -> xr.DataArray:
            return xr.DataArray(shapely.geometrycollections(np.ravel(x)))

        def _union(x: xr.DataArray) -> xr.DataArray:
            return xr.DataArray(shapely.union_all(np.ravel(x)))

        match aggfunc:
            case "envelope":
                summary = shapely.envelope(obj.groupby(dim).map(_collect)).data
            case "centroid":
                summary = shapely.centroid(obj.groupby(dim).map(_collect)).data
            case "oriented_envelope":
                summary = shapely.oriented_envelope(obj.groupby(dim).map(_collect)).data
            case "convex_hull":
                summary = shapely.convex_hull(obj.groupby(dim).map(_collect)).data
            case "concave_hull":
                summary = shapely.concave_hull(
                    obj.groupby(dim).map(_collect),
                    **kwargs,
                ).data
            case "collection":
                summary = obj.groupby(dim).map(_collect).data
            case "union":
                summary = obj.groupby(dim).map(_union).data
            case _ if callable(aggfunc):
                summary = obj.groupby(dim).map(aggfunc, **kwargs).data

        return (
            self._obj.assign_coords(summary_geometry=(dim, summary))
            .set_xindex("summary_geometry")
            .xvec.set_geom_indexes("summary_geometry", crs=self._obj.proj.crs)
        )

    def plot(
        self,
        *,
        row: Hashable | None = None,
        col: Hashable | None = None,
        col_wrap: int | None = None,
        hue: Hashable | None = None,
        subplot_kws: dict[str, Any] | None = None,
        figsize: Iterable[float] | None = None,
        geometry: Hashable | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str | Any | None = None,
        center: float | bool | None = None,
        robust: bool = False,
        extend: str | None = None,
        levels: int | Iterable[float] | None = None,
        norm: Any | None = None,
        **kwargs: dict[str, Any],
    ) -> tuple[Any, Any]:
        """
        Plot geometry with optional faceting and color mapping.

        Uses GeoPandas to plot the geometry and data from the object.

        Parameters
        ----------
        arr : xarray.DataArray or xarray.Dataset
            The data to be plotted.
        row : Hashable or None, optional
            If passed, make row faceted plots on this dimension name.
        col : Hashable or None, optional
            If passed, make column faceted plots on this dimension name.
        col_wrap : int, optional
            Number of columns to wrap facets into. Use together with ``col``.
        hue : Hashable or None, optional
            If passed, make faceted plots with value on from this dimension.
        subplot_kws : dict, optional
            Dictionary of keyword arguments for Matplotlib subplots
            (see :py:meth:`~matplotlib:matplotlib.figure.Figure.add_subplot`).
        figsize : tuple, optional
            A tuple (width, height) of the figure in inches.
        geometry : str, optional
            Geometry array to use for plotting. Could be both coordinate geometry and
            variable geometry. If None, the method tries to infer the geometry when
            plotting a DataArray. Must be specified for a Dataset.
        vmin : float or None, optional
            Lower value to anchor the colormap, otherwise it is inferred from the
            data and other keyword arguments. When a diverging dataset is inferred,
            setting `vmin` or `vmax` will fix the other by symmetry around
            ``center``. Setting both values prevents use of a diverging colormap.
            If discrete levels are provided as an explicit list, both of these
            values are ignored.
        vmax : float or None, optional
            Upper value to anchor the colormap, otherwise it is inferred from the
            data and other keyword arguments. When a diverging dataset is inferred,
            setting `vmin` or `vmax` will fix the other by symmetry around
            ``center``. Setting both values prevents use of a diverging colormap.
            If discrete levels are provided as an explicit list, both of these
            values are ignored.
        cmap : matplotlib colormap name or colormap, optional
            The mapping from data values to color space. Either a
            Matplotlib colormap name or object. If not provided, this will
            be either ``'viridis'`` (if the function infers a sequential
            dataset) or ``'RdBu_r'`` (if the function infers a diverging
            dataset).
            See :doc:`Choosing Colormaps in Matplotlib <matplotlib:users/explain/colors/colormaps>`
            for more information.
        center : float or False, optional
            The value at which to center the colormap. Passing this value implies
            use of a diverging colormap. Setting it to ``False`` prevents use of a
            diverging colormap.
        robust : bool, optional
            If ``True`` and ``vmin`` or ``vmax`` are absent, the colormap range is
            computed with 2nd and 98th percentiles instead of the extreme values.
        extend : {'neither', 'both', 'min', 'max'}, optional
            How to draw arrows extending the colorbar beyond its limits. If not
            provided, ``extend`` is inferred from ``vmin``, ``vmax`` and the data limits.
        levels : int or array-like, optional
            Split the colormap (``cmap``) into discrete color intervals. If an integer
            is provided, "nice" levels are chosen based on the data range: this can
            imply that the final number of levels is not exactly the expected one.
            Setting ``vmin`` and/or ``vmax`` with ``levels=N`` is equivalent to
            setting ``levels=np.linspace(vmin, vmax, N)``.
        norm : matplotlib.colors.Normalize, optional
            If ``norm`` has ``vmin`` or ``vmax`` specified, the corresponding
            kwarg must be ``None``.
        **kwargs : dict
            Additional keyword arguments passed to geopandas plotting method.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        axs : numpy.ndarray of matplotlib.axes.Axes
            Array of axes objects for the plot.
        """
        return _plot(
            self._obj,
            row=row,
            col=col,
            col_wrap=col_wrap,
            hue=hue,
            subplot_kws=subplot_kws,
            figsize=figsize,
            geometry=geometry,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            center=center,
            robust=robust,
            extend=extend,
            levels=levels,
            norm=norm,
            **kwargs,
        )


def _resolve_input(
    positional: Mapping[Any, Any] | None,
    keyword: Mapping[str, Any],
    func_name: str,
) -> Mapping[Hashable, Any]:
    """Resolve combination of positional and keyword arguments.

    Based on xarray's ``either_dict_or_kwargs``.
    """
    if positional and keyword:
        raise ValueError(
            "Cannot specify both keyword and positional arguments to "
            f"'.xvec.{func_name}'."
        )
    if positional is None or positional == {}:
        return cast(Mapping[Hashable, Any], keyword)
    return positional
