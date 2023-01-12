from typing import Any, Hashable, Mapping, Sequence, Union

import numpy as np
import shapely
import xarray as xr
from pyproj import CRS, Transformer

from .index import GeometryIndex


@xr.register_dataarray_accessor("xvec")
@xr.register_dataset_accessor("xvec")
class XvecAccessor:
    """Access geometry-based methods for DataArrays and Datasets with Shapely geometry.

    Currently works on coordinates with :class:`xvec.GeometryIndex`.
    """

    def __init__(self, xarray_obj: Union[xr.Dataset, xr.DataArray]):
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
                # try first on a small subset
                subset = self._obj[name].data[0:10]
                if np.all(shapely.is_valid_input(subset)):
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
        return self._obj[self._geom_coords_all].coords

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
        >>> ds.xvec.geom_coords_indexed
        Coordinates:
          * geom     (geom) object POINT (1 2) POINT (3 4)
          * geom_z   (geom_z) object POINT Z (10 20 30) POINT Z (30 40 50)

        See also
        --------
        geom_coords
        is_geom_variable

        """
        return self._obj[self._geom_indexes].coords

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

        for key, (result, crs) in transformed.items():
            _obj = _obj.assign_coords({key: result})

        _obj = _obj.drop_indexes(variable_crs.keys())

        for key, crs in variable_crs.items():
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
            _obj = _obj.set_xindex(coord, GeometryIndex, crs=crs, **kwargs)

        return _obj
