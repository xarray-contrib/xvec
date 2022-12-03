from typing import Any, List, Mapping, Union

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
        self._geom_coords_names = [
            name
            for name, index in self._obj.xindexes.items()
            if isinstance(index, GeometryIndex)
        ]

    @property
    def geom_coords_names(self) -> List:
        """Returns a list of coordinates using :class:`~xvec.GeometryIndex`.

        Returns
        -------
        List
            list of strings representing names of coordinates

        Examples
        --------
        >>> ds = (
        ...     xr.Dataset(
        ...         coords={
        ...             "geom": np.array([shapely.Point(1, 2), shapely.Point(3, 4)]),
        ...             "geom_z": np.array(
        ...                 [shapely.Point(10, 20, 30), shapely.Point(30, 40, 50)]
        ...             ),
        ...         }
        ...     )
        ...     .drop_indexes(["geom", "geom_z"])
        ...     .set_xindex("geom", xvec.GeometryIndex, crs=26915)
        ...     .set_xindex("geom_z", xvec.GeometryIndex, crs=26915)
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:  (geom: 2, geom_z: 2)
        Coordinates:
        * geom     (geom) object POINT (1 2) POINT (3 4)
        * geom_z   (geom_z) object POINT Z (10 20 30) POINT Z (30 40 50)
        Data variables:
            *empty*
        Indexes:
            geom     GeometryIndex (crs=EPSG:26915)
            geom_z   GeometryIndex (crs=EPSG:26915)
        >>> ds.xvec.geom_coords_names
        ['geom', 'geom_z']
        """
        return self._geom_coords_names

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
        ...     .drop_indexes("geom")
        ...     .set_xindex("geom", xvec.GeometryIndex, crs=4326)
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
        ...     .drop_indexes("geom")
        ...     .set_xindex("geom", xvec.GeometryIndex, crs=4326)
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
            If the the :class:`~xvec.GeometryIndex` already has a CRS,
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
        ...     .drop_indexes("geom")
        ...     .set_xindex("geom", xvec.GeometryIndex)
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
        ...     .drop_indexes("geom")
        ...     .set_xindex("geom", xvec.GeometryIndex, crs=4326)
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
