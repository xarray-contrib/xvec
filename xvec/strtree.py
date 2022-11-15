import numpy
import shapely
import xarray
from xarray.core.indexes import IndexSelResult
from xarray.indexes import Index


class ShapelySTRTreeIndex(Index):
    def __init__(self, array, dim, crs):
        assert numpy.all(shapely.is_geometry(array))

        # only support 1-d coordinate for now
        assert len(array.shape) == 1

        self._tree = shapely.STRtree(numpy.ravel(array))
        self.dim = dim
        self.crs = crs

    @classmethod
    def from_variables(cls, variables, *, options):
        # only supports one coordinate of shapely geometries
        assert len(variables) == 1
        var = next(iter(variables.values()))

        return cls(var._data, var.dims[0], options["crs"])

    def sel(self, labels, method=None, tolerance=None):
        # We reuse here `method` and `tolerance` options of
        # `xarray.indexes.PandasIndex` as `predicate` and `distance`
        # options when `labels` is a single geometry.
        # Xarray currently doesn't support custom options
        # (see https://github.com/pydata/xarray/issues/7099)

        # only one coordinate supported
        assert len(labels) == 1
        label = next(iter(labels.values()))

        if isinstance(label, xarray.DataArray):
            label_array = label._variable._data
        elif isinstance(label, xarray.Variable):
            label_array = label._data
        elif isinstance(label, shapely.Geometry):
            label_array = numpy.array([label])
        else:
            label_array = numpy.array(label)

        # check for possible CRS of geometry labels
        # (by default assume same CRS than the index)
        if hasattr(label_array, "crs") and label_array.crs != self.crs:
            raise ValueError("conflicting CRS for input geometries")

        assert numpy.all(shapely.is_geometry(label_array))

        if method is None or method == "nearest":
            indices = self._tree.nearest(label_array)
        else:
            indices = self._tree.query(label, predicate=method, distance=tolerance)

            if indices.ndim == 2:
                indices = indices[1]

        # attach dimension names and/or coordinates to positional indexer
        if isinstance(label, xarray.Variable):
            indices = xarray.Variable(label.dims, indices)
        elif isinstance(label, xarray.DataArray):
            indices = xarray.DataArray(indices, coords=label._coords, dims=label.dims)

        return IndexSelResult({self.dim: indices})
