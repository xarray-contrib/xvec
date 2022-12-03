from typing import Any, List, Mapping, Union

import numpy as np
import shapely
import xarray as xr
from pyproj import CRS, Transformer

from .index import GeometryIndex


@xr.register_dataarray_accessor("xvec")
@xr.register_dataset_accessor("xvec")
class XvecAccessor:
    def __init__(self, xarray_obj: Union[xr.Dataset, xr.DataArray]):
        self._obj = xarray_obj
        self._geom_coord_names = [
            name
            for name, index in self._obj.xindexes.items()
            if isinstance(index, GeometryIndex)
        ]

    @property
    def geom_coords_names(self) -> List:
        return self._geom_coord_names

    def to_crs(
        self,
        variable_crs: Mapping[Any, Any] | None = None,
        **variable_crs_kwargs: Any,
    ):
        if variable_crs and variable_crs_kwargs:
            raise ValueError(
                "cannot specify both keyword and positional arguments to "
                ".xvec.to_crs"
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

        if variable_crs and variable_crs_kwargs:
            raise ValueError(
                "cannot specify both keyword and positional arguments to "
                ".xvec.set_crs"
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

        _obj = _obj.drop_indexes(variable_crs_kwargs.keys())

        for key, crs in variable_crs_kwargs.items():
            _obj = _obj.set_xindex(key, GeometryIndex, crs=crs)

        return _obj
