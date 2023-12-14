from __future__ import annotations

import warnings
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import shapely
import xarray as xr
from pyproj import CRS, Transformer

from .index import GeometryIndex
from .zonal import _zonal_stats_iterative, _zonal_stats_rasterize


@xr.register_dataarray_accessor("xvec")
@xr.register_dataset_accessor("xvec")
class XvecAccessor:
    """Access geometry-based methods for DataArrays and Datasets with Shapely geometry.

    Currently works on coordinates with :class:`xvec.GeometryIndex`.
    """

    def __init__(self, xarray_obj: xr.Dataset | xr.DataArray):
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

    def is_geom_variable(self, name: Hashable, has_index: bool = True):
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
    def geom_coords(self) -> Mapping[Hashable, xr.DataArray]:
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
        # TODO: use xarray.Coordinates constructor instead once available in xarray
        return xr.DataArray(
            coords={c: self._obj[c] for c in self._geom_coords_all},
            dims=self._geom_coords_all,
        ).coords

    @property
    def geom_coords_indexed(self) -> Mapping[Hashable, xr.DataArray]:
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
        # TODO: use xarray.Coordinates constructor instead once available in xarray
        return self._obj.drop_vars(
            [c for c in self._obj.coords if c not in self._geom_indexes]
        ).coords

    def to_crs(
        self,
        variable_crs: Mapping[Any, Any] | None = None,
        **variable_crs_kwargs: Any,
    ):
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
        if variable_crs and variable_crs_kwargs:
            raise ValueError(
                "Cannot specify both keyword and positional arguments to "
                "'.xvec.to_crs'."
            )

        _obj = self._obj.copy(deep=False)

        if variable_crs_kwargs:
            variable_crs = variable_crs_kwargs

        transformed = {}

        for key, crs in variable_crs.items():
            if not isinstance(self._obj.xindexes[key], GeometryIndex):
                raise ValueError(
                    f"The index '{key}' is not an xvec.GeometryIndex. "
                    "Set the xvec.GeometryIndex using '.xvec.set_geom_indexes' before "
                    "handling projection information."
                )

            data = _obj[key]
            data_crs = self._obj.xindexes[key].crs

            # transformation code taken from geopandas (BSD 3-clause license)
            if data_crs is None:
                raise ValueError(
                    "Cannot transform naive geometries. "
                    f"Please set a CRS on the '{key}' coordinates first."
                )

            crs = CRS.from_user_input(crs)

            if data_crs.is_exact_same(crs):
                pass

            transformer = Transformer.from_crs(data_crs, crs, always_xy=True)

            has_z = shapely.has_z(data)

            result = np.empty_like(data)

            coordinates = shapely.get_coordinates(data[~has_z], include_z=False)
            new_coords = transformer.transform(coordinates[:, 0], coordinates[:, 1])
            result[~has_z] = shapely.set_coordinates(
                data[~has_z].copy(), np.array(new_coords).T
            )

            coords_z = shapely.get_coordinates(data[has_z], include_z=True)
            new_coords_z = transformer.transform(
                coords_z[:, 0], coords_z[:, 1], coords_z[:, 2]
            )
            result[has_z] = shapely.set_coordinates(
                data[has_z].copy(), np.array(new_coords_z).T
            )

            transformed[key] = (result, crs)

        for key, (result, _crs) in transformed.items():
            _obj = _obj.assign_coords({key: result})

        _obj = _obj.drop_indexes(variable_crs.keys())

        for key, crs in variable_crs.items():
            if crs:
                _obj[key].attrs["crs"] = CRS.from_user_input(crs)
            _obj = _obj.set_xindex(key, GeometryIndex, crs=crs)

        return _obj

    def set_crs(
        self,
        variable_crs: Mapping[Any, Any] | None = None,
        allow_override=False,
        **variable_crs_kwargs: Any,
    ):
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

        if variable_crs and variable_crs_kwargs:
            raise ValueError(
                "Cannot specify both keyword and positional arguments to "
                ".xvec.set_crs."
            )

        _obj = self._obj.copy(deep=False)

        if variable_crs_kwargs:
            variable_crs = variable_crs_kwargs

        for key, crs in variable_crs.items():
            if not isinstance(self._obj.xindexes[key], GeometryIndex):
                raise ValueError(
                    f"The index '{key}' is not an xvec.GeometryIndex. "
                    "Set the xvec.GeometryIndex using '.xvec.set_geom_indexes' before "
                    "handling projection information."
                )

            data_crs = self._obj.xindexes[key].crs

            if not allow_override and data_crs is not None and not data_crs == crs:
                raise ValueError(
                    f"The index '{key}' already has a CRS which is not equal to the "
                    "passed CRS. Specify 'allow_override=True' to allow replacing the "
                    "existing CRS without doing any transformation. If you actually "
                    "want to transform the geometries, use '.xvec.to_crs' instead."
                )

        _obj = _obj.drop_indexes(variable_crs.keys())

        for key, crs in variable_crs.items():
            if crs:
                _obj[key].attrs["crs"] = CRS.from_user_input(crs)
            _obj = _obj.set_xindex(key, GeometryIndex, crs=crs)

        return _obj

    def query(
        self,
        coord_name: str,
        geometry: shapely.Geometry | Sequence[shapely.Geometry],
        predicate: str = None,
        distance: float | Sequence[float] = None,
        unique=False,
    ):
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
        coord_name : str
            name of the coordinate axis backed by :class:`~xvec.GeometryIndex`
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
            ilocs = self._obj.xindexes[coord_name].sindex.query(
                geometry, predicate=predicate, distance=distance
            )

        else:
            _, ilocs = self._obj.xindexes[coord_name].sindex.query(
                geometry, predicate=predicate, distance=distance
            )
            if unique:
                ilocs = np.unique(ilocs)

        return self._obj.isel({coord_name: ilocs})

    def set_geom_indexes(
        self,
        coord_names: str | Sequence[Hashable],
        crs: Any = None,
        allow_override: bool = False,
        **kwargs,
    ):
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
                data_crs = self._obj.xindexes[coord].crs

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

    def to_geopandas(self):
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
                elif self._obj.ndim == 2:
                    gdf = self._obj.to_pandas()
                    if gdf.columns.name == self._geom_indexes[0]:
                        gdf = gdf.T
                return gdf.reset_index().set_geometry(
                    self._geom_indexes[0],
                    crs=self._obj.xindexes[self._geom_indexes[0]].crs,
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
            )

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
    ):
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
                ]
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
                df[c] = gpd.GeoSeries(df[c], crs=self._obj[c].attrs.get("crs", None))

        if geometry is not None:
            return df.set_geometry(
                geometry, crs=self._obj[geometry].attrs.get("crs", None)
            )

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
        polygons: Sequence[shapely.Geometry],
        x_coords: Hashable,
        y_coords: Hashable,
        stats: str = "mean",
        name: Hashable = "geometry",
        index: bool = None,
        method: str = "rasterize",
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
        stats : string
            Spatial aggregation statistic method, by default "mean". It supports the
            following statistcs: ['mean', 'median', 'min', 'max', 'sum']
        name : Hashable, optional
            Name of the dimension that will hold the ``polygons``, by default "geometry"
        index : bool, optional
            If `polygons` is a GeoSeries, ``index=True`` will attach its index as another
            coordinate to the geometry dimension in the resulting object. If
            ``index=None``, the index will be stored if the `polygons.index` is a named
            or non-default index. If ``index=False``, it will never be stored. This is
            useful as an attribute link between the resulting array and the GeoPandas
            object from which the polygons are sourced.
        method : str, optional
            The method of data extraction. The default is ``"rasterize"``, which uses
            :func:`rasterio.features.rasterize` and is faster, but can lead to loss
            of information in case of small polygons. Other option is ``"iterate"``, which
            iterates over polygons and uses :func:`rasterio.features.geometry_mask`.
        all_touched : bool, optional
            If True, all pixels touched by geometries will be considered. If False, only
            pixels whose center is within the polygon or that are selected by
            Bresenham’s line algorithm will be considered.
        n_jobs : int, optional
            Number of parallel threads to use. It is recommended to set this to the
            number of physical cores of the CPU. ``-1`` uses all available cores. Applies
            only if ``method="iterate"``.
        **kwargs : optional
            Keyword arguments to be passed to the aggregation function
            (e.g., ``Dataset.mean(**kwargs)``).

        Returns
        -------
        Dataset
            A subset of the original object with N-1 dimensions indexed by
            the the GeometryIndex.

        """
        # TODO: allow multiple stats at the same time (concat along a new axis),
        # TODO: possibly as a list of tuples to include names?
        # TODO: allow callable in stat (via .reduce())
        if method == "rasterize":
            result = _zonal_stats_rasterize(
                self,
                polygons=polygons,
                x_coords=x_coords,
                y_coords=y_coords,
                stats=stats,
                name=name,
                all_touched=all_touched,
                **kwargs,
            )
        elif method == "iterate":
            result = _zonal_stats_iterative(
                self,
                polygons=polygons,
                x_coords=x_coords,
                y_coords=y_coords,
                stats=stats,
                name=name,
                all_touched=all_touched,
                n_jobs=n_jobs,
                **kwargs,
            )
        else:
            raise ValueError(
                f"method '{method}' is not supported. Allowed options are 'rasterize' "
                "and 'iterate'."
            )

        # save the index as a data variable
        if isinstance(polygons, pd.Series):
            if index is None:
                if polygons.index.name is not None or not polygons.index.equals(
                    pd.RangeIndex(0, len(polygons))
                ):
                    index = True
            if index:
                index_name = polygons.index.name if polygons.index.name else "index"
                result = result.assign_coords({index_name: (name, polygons.index)})

        # standardize the shape - each method comes with a different one
        return result.transpose(
            name, *tuple(d for d in self._obj.dims if d not in [x_coords, y_coords])
        )

    def extract_points(
        self,
        points: Sequence[shapely.Geometry],
        x_coords: Hashable,
        y_coords: Hashable,
        tolerance: float | None = None,
        name: str = "geometry",
        crs: Any = None,
        index: bool = None,
    ):
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
