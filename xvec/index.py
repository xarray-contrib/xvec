from __future__ import annotations

from typing import Any, Hashable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import shapely
from pyproj import CRS
from xarray import DataArray, Variable
from xarray.core.indexing import IndexSelResult
from xarray.indexes import Index, PandasIndex


class GeoVectorIndex(Index):
    """An CRS-aware, Xarray-compatible index for vector geometries.

    This index can be set from any 1-dimensional coordinate of
    (shapely 2.0) :class:`shapely.Geometry` elements.

    It provides all the basic functionality of an
    :class:`xarray.indexes.PandasIndex`. In addition, it allows spatial
    filtering based on geometries (powered by :class:`shapely.STRtree`).

    Parameters
    ----------
    index : :class:`xarray.indexes.PandasIndex`
        An Xarray (pandas) index built from an array-like of
        :class:`shapely.Geometry` objects.
    crs : object
        The coordinate reference system. Any value accepted by
        :meth:`pyproj.crs.CRS.from_user_input`.

    """

    _index: PandasIndex
    _sindex: shapely.STRtree | None
    _crs: CRS

    def __init__(self, index: PandasIndex, crs: CRS):
        if not np.all(shapely.is_geometry(index.index)):
            raise ValueError("array must contain shapely.Geometry objects")

        self._crs = CRS.from_user_input(crs)
        self._index = index
        self._sindex = None

    @property
    def crs(self) -> CRS:
        """Returns the coordinate reference system of the index as a
        :class:`pyproj.crs.CRS` object.
        """
        return self._crs

    @property
    def sindex(self) -> shapely.STRtree:
        """Returns the spatial index, i.e., a :class:`shapely.STRtree` object.

        It may build the index before returning it.
        """
        if self._sindex is None:
            self._sindex = shapely.STRtree(self._index.index)
        return self._sindex

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, Variable],
        *,
        options: Mapping[str, Any],
    ):
        # TODO: try getting CRS from coordinate attrs or GeometryArray
        if "crs" not in options:
            raise ValueError("a CRS must be provided")

        index = PandasIndex.from_variables(variables, options={})
        return cls(index, crs=options["crs"])

    @classmethod
    def concat(
        cls,
        indexes: Sequence[GeoVectorIndex],
        dim: Hashable,
        positions: Iterable[Iterable[int]] | None = None,
    ) -> GeoVectorIndex:
        crss = [idx.crs for idx in indexes]

        if any([s != crss[0] for s in crss]):
            raise ValueError("conflicting CRS for coordinates to concat")

        indexes_ = [idx._index for idx in indexes]
        index = PandasIndex.concat(indexes_, dim, positions)
        return cls(index, crss[0])

    def create_variables(
        self, variables: Mapping[Any, Variable] | None = None
    ) -> dict[Hashable, Variable]:
        return self._index.create_variables(variables)

    def to_pandas_index(self) -> pd.Index:
        return self._index.index

    def isel(self, indexers: Mapping[Any, Any]):
        index = self._index.isel(indexers)

        if index is not None:
            return type(self)(index, self.crs)
        else:
            return None

    def _sel_sindex(self, labels, method, tolerance):
        # only one coordinate supported
        assert len(labels) == 1
        label = next(iter(labels.values()))

        if method != "nearest":
            if not isinstance(label, shapely.Geometry):
                raise ValueError(
                    "selection with another method than nearest only supports "
                    "a single geometry as input label."
                )

        if isinstance(label, DataArray):
            label_array = label._variable._data
        elif isinstance(label, Variable):
            label_array = label._data
        elif isinstance(label, shapely.Geometry):
            label_array = np.array([label])
        else:
            label_array = np.array(label)

        # check for possible CRS of geometry labels
        # (by default assume same CRS than the index)
        if hasattr(label_array, "crs") and label_array.crs != self.crs:
            raise ValueError("conflicting CRS for input geometries")

        assert np.all(shapely.is_geometry(label_array))

        if method == "nearest":
            indices = self.sindex.nearest(label_array)
        else:
            indices = self.sindex.query(label, predicate=method, distance=tolerance)

        # attach dimension names and/or coordinates to positional indexer
        if isinstance(label, Variable):
            indices = Variable(label.dims, indices)
        elif isinstance(label, DataArray):
            indices = DataArray(indices, coords=label._coords, dims=label.dims)

        return IndexSelResult({self._index.dim: indices})

    def sel(
        self, labels: dict[Any, Any], method=None, tolerance=None
    ) -> IndexSelResult:
        if method is None:
            return self._index.sel(labels)
        else:
            # We reuse here `method` and `tolerance` options of
            # `xarray.indexes.PandasIndex` as `predicate` and `distance`
            # options when `labels` is a single geometry.
            # Xarray currently doesn't support custom options
            # (see https://github.com/pydata/xarray/issues/7099)
            return self._sel_sindex(labels, method, tolerance)

    def equals(self, other: Index) -> bool:
        if not isinstance(other, GeoVectorIndex):
            return False
        if other.crs != self.crs:
            return False
        return self._index.equals(other._index)

    def join(
        self: GeoVectorIndex, other: GeoVectorIndex, how: str = "inner"
    ) -> GeoVectorIndex:
        index = self._index.join(other._index, how=how)
        return type(self)(index, self.crs)

    def reindex_like(
        self, other: GeoVectorIndex, method=None, tolerance=None
    ) -> dict[Hashable, Any]:
        return self._index.reindex_like(
            other._index, method=method, tolerance=tolerance
        )

    def roll(self, shifts: Mapping[Any, int]) -> GeoVectorIndex:
        index = self._index.roll(shifts)
        return type(self)(index, self.crs)

    def rename(self, name_dict, dims_dict):
        index = self._index.rename(name_dict, dims_dict)
        return type(self)(index, self.crs)

    def _repr_inline_(self, max_width):
        return f"{self.__class__.__name__}(crs={self.crs.to_string()})"
