from __future__ import annotations

import warnings
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import shapely
import xproj
from pyproj import CRS
from xarray import DataArray, Variable, get_options
from xarray.core.indexing import IndexSelResult
from xarray.indexes import Index, PandasIndex

from xvec.utils import transform_geom


class GeometryIndex(Index, xproj.ProjIndexMixin):
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
    crs : :class:`pyproj.crs.CRS` or any, optional
        The coordinate reference system. Any value accepted by
        :meth:`pyproj.crs.CRS.from_user_input`.

    """

    _index: PandasIndex
    _sindex: shapely.STRtree | None
    _crs: CRS | None

    def __init__(self, index: PandasIndex, crs: CRS | Any | None = None):
        if not np.all(shapely.is_geometry(index.index)):
            raise ValueError("array must contain shapely.Geometry objects")

        if crs is not None:
            crs = CRS.from_user_input(crs)

        self._crs = crs
        self._index = index
        self._sindex = None

    @property
    def crs(self) -> CRS | None:
        """Returns the coordinate reference system of the index as a
        :class:`pyproj.crs.CRS` object.
        """
        return self._crs

    def _proj_set_crs(
        self: GeometryIndex, spatial_ref: Hashable, crs: CRS
    ) -> GeometryIndex:
        # Returns a geometry index shallow copy with a replaced CRS, without transformation
        # (XProj integration via xproj.ProjIndexMixin)
        # Note: XProj already handles the case of overriding any existing CRS
        return GeometryIndex(self._index, crs=crs)

    def _proj_to_crs(
        self: GeometryIndex, spatial_ref: Hashable, crs: CRS
    ) -> GeometryIndex:
        # Returns a new geometry index with a replaced CRS and transformed geometries
        # (XProj integration via xproj.ProjIndexMixin)
        # Note: XProj already handles the case of overriding any existing CRS

        # XProj redirects to `._proj_set_crs()` if this index's CRS is undefined
        assert self.crs is not None

        result = transform_geom(np.asarray(self._index.index), self.crs, crs)
        index = PandasIndex(
            result, self._index.dim, coord_dtype=self._index.coord_dtype
        )
        return GeometryIndex(index, crs=crs)

    @property
    def sindex(self) -> shapely.STRtree:
        """Returns the spatial index, i.e., a :class:`shapely.STRtree` object.

        It may build the index before returning it if it hasn't been built before.
        """
        if self._sindex is None:
            self._sindex = shapely.STRtree(self._index.index)
        return self._sindex

    def _crs_mismatch_raise(
        self, other_crs: CRS | None, warn: bool = False, stacklevel: int = 3
    ) -> None:
        """Raise a CRS mismatch error or warning with the information
        on the assigned CRS.
        """
        srs = xproj.format_crs(self.crs, max_width=50)
        other_srs = xproj.format_crs(other_crs, max_width=50)

        # TODO: expand message with reproject suggestion
        msg = (
            "CRS mismatch between the CRS of index geometries "
            "and the CRS of input geometries.\n"
            f"Index CRS: {srs}\n"
            f"Input CRS: {other_srs}\n"
        )

        if warn:
            warnings.warn(msg, UserWarning, stacklevel=stacklevel)
        else:
            raise ValueError(msg)

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, Variable],
        *,
        options: Mapping[str, Any],
    ) -> GeometryIndex:
        # TODO: try getting CRS from coordinate attrs or GeometryArray or SRID

        index = PandasIndex.from_variables(variables, options={})
        return cls(index, crs=options.get("crs"))

    @classmethod
    def concat(
        cls,
        indexes: Sequence[GeometryIndex],
        dim: Hashable,
        positions: Iterable[Iterable[int]] | None = None,
    ) -> GeometryIndex:
        crs_set = {idx.crs for idx in indexes}
        crs = xproj.get_common_crs(crs_set)

        indexes_ = [idx._index for idx in indexes]
        index = PandasIndex.concat(indexes_, dim, positions)
        return cls(index, crs)

    def create_variables(
        self, variables: Mapping[Any, Variable] | None = None
    ) -> dict[Hashable, Variable]:
        return self._index.create_variables(variables)

    def to_pandas_index(self) -> pd.Index:
        return self._index.index

    def isel(self, indexers: Mapping[Any, Any]) -> GeometryIndex | None:
        index = self._index.isel(indexers)

        if index is not None:
            return type(self)(index, self.crs)
        else:
            return None

    def _sel_sindex(self, labels, method: str, tolerance) -> IndexSelResult:
        # only one entry expected
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

        if not np.all(shapely.is_geometry(label_array)):
            raise ValueError("labels must be shapely.Geometry objects")

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
        self,
        labels: dict[Any, Any],
        method: str | None = None,
        tolerance: int | float | Iterable[int | float] | None = None,
    ) -> IndexSelResult:
        if method is None:
            return self._index.sel(labels)
        else:
            # We reuse here `method` and `tolerance` options of
            # `xarray.indexes.PandasIndex` as `predicate` and `distance`
            # options when `labels` is a single geometry.
            # Xarray currently doesn't support custom options
            # (see https://github.com/pydata/xarray/issues/7099)
            return self._sel_sindex(labels, method, tolerance)  # type: ignore

    def equals(
        self, other: Index, *, exclude: frozenset[Hashable] | None = None
    ) -> bool:
        if not isinstance(other, GeometryIndex):
            return False
        if not self._proj_crs_equals(other, allow_none=True):
            return False
        return self._index.equals(other._index, exclude=exclude)

    def join(
        self: GeometryIndex, other: GeometryIndex, how: str = "inner"
    ) -> GeometryIndex:
        if not self._proj_crs_equals(other, allow_none=True):
            self._crs_mismatch_raise(other.crs)

        index = self._index.join(other._index, how=how)
        return type(self)(index, self.crs)

    def reindex_like(
        self,
        other: GeometryIndex,
        method: str | None = None,
        tolerance: int | float | Iterable[int | float] | None = None,
    ) -> dict[Hashable, Any]:
        if not self._proj_crs_equals(other, allow_none=True):
            self._crs_mismatch_raise(other.crs)

        return self._index.reindex_like(
            other._index, method=method, tolerance=tolerance
        )

    def roll(self, shifts: Mapping[Any, int]) -> GeometryIndex:
        index = self._index.roll(shifts)
        return type(self)(index, self.crs)

    def rename(
        self, name_dict: Mapping[Any, Hashable], dims_dict: Mapping[Any, Hashable]
    ) -> GeometryIndex:
        index = self._index.rename(name_dict, dims_dict)
        return type(self)(index, self.crs)

    def _repr_inline_(self, max_width: int) -> str:
        # TODO: remove when fixed in XArray
        if max_width is None:
            max_width = get_options()["display_width"]

        srs = xproj.format_crs(self.crs, max_width=max_width)
        return f"{self.__class__.__name__} (crs={srs})"

    def __repr__(self) -> str:
        srs = xproj.format_crs(self.crs)
        shape = self._index.index.shape[0]
        if shape == 0:
            return f"GeometryIndex([], crs={srs})"
        if shape < 10:
            wkts = [repr(g) for g in self._index.index]
        else:
            wkts = [repr(g) for g in self._index.index[:4]]
            wkts.append("...")
            wkts = wkts + [repr(g) for g in self._index.index[-4:]]

        if len(wkts) == 1:
            return f"GeometryIndex([{wkts[0]}], crs={srs})"
        joined = "\n ".join(wkts[1:])
        return f"GeometryIndex(\n[{wkts[0]}\n {joined}],\ncrs={srs})".replace(
            "\n", "\n" + " " * 4
        )
