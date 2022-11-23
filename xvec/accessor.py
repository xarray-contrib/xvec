from typing import Any, Hashable, Union

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

    def coords_to_crs(self, coords: Hashable, crs: Any):

        data = self._obj[coords]
        data_crs = self._obj.xindexes[coords].crs

        # transformation code taken from geopandas (BSD 3-clause license)
        if data_crs is None:
            raise ValueError(
                "Cannot transform naive geometries. "
                "Please set a CRS on the object first."
            )

        crs = CRS.from_user_input(crs)

        if data_crs.is_exact_same(crs):
            return self

        transformer = Transformer.from_crs(data_crs, crs, always_xy=True)

        has_z = shapely.has_z(data)

        result = np.empty_like(data)

        coordinates = shapely.get_coordinates(data[~has_z], include_z=False)
        new_coords_z = transformer.transform(coordinates[:, 0], coordinates[:, 1])
        result[~has_z] = shapely.set_coordinates(
            data[~has_z].copy(), np.array(new_coords_z).T
        )

        coords_z = shapely.get_coordinates(data[has_z], include_z=True)
        new_coords_z = transformer.transform(
            coords_z[:, 0], coords_z[:, 1], coords_z[:, 2]
        )
        result[has_z] = shapely.set_coordinates(
            data[has_z].copy(), np.array(new_coords_z).T
        )

        # return result
        return (
            self._obj.assign_coords({coords: result})
            .drop_indexes(coords)
            .set_xindex(coords, GeometryIndex, crs=crs)
        )
