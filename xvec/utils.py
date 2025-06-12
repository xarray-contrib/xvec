import numpy as np
import shapely
from pyproj import CRS, Transformer


def transform_geom(array: np.ndarray, crs_from: CRS, crs_to: CRS) -> np.ndarray:
    # transformation code taken from geopandas (BSD 3-clause license)
    if crs_from.is_exact_same(crs_to):
        return np.asarray(array)

    transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)

    has_z = shapely.has_z(array)

    result = np.empty_like(array)

    coordinates = shapely.get_coordinates(array[~has_z], include_z=False)
    new_coords = transformer.transform(coordinates[:, 0], coordinates[:, 1])
    result[~has_z] = shapely.set_coordinates(
        array[~has_z].copy(), np.array(new_coords).T
    )

    coords_z = shapely.get_coordinates(array[has_z], include_z=True)
    new_coords_z = transformer.transform(coords_z[:, 0], coords_z[:, 1], coords_z[:, 2])
    result[has_z] = shapely.set_coordinates(
        array[has_z].copy(), np.array(new_coords_z).T
    )

    return result
