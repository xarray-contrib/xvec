from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
import shapely
import xarray as xr
import xproj  # noqa: F401
from xarray.core.formatting import format_item
from xarray.plot.utils import _determine_cmap_params, label_from_attrs


def _setup_axes(n_rows, n_cols, arr, geometry, crs, subplot_kws, figsize):
    import matplotlib.pyplot as plt

    if figsize is None:
        if geometry:
            geoms = arr[geometry].data
        elif np.all(shapely.is_valid_input(arr.data)):
            geoms = arr.data
        else:
            geoms = arr[list(arr.xvec._geom_coords_all)[0]]

        bounds = shapely.total_bounds(np.asarray(geoms))

        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]

        max_dim = max(width, height)
        width /= max_dim
        height /= max_dim

        # get aspect for geographic coordinates
        if crs and crs.is_geographic:
            y_coord = np.mean([bounds[1], bounds[3]])
            height *= 1 / np.cos(y_coord * np.pi / 180)

        figsize = (n_cols * 3 * width, n_rows * 3 * height)

    return plt.subplots(
        n_rows,
        n_cols,
        sharex=True,
        sharey=True,
        figsize=figsize,
        subplot_kw=subplot_kws,
    )


def _get_axis_labels(crs):
    if crs:
        x_label = f"{crs.axis_info[0].name}\n[{crs.axis_info[0].unit_name}]"
        y_label = f"{crs.axis_info[1].name}\n[{crs.axis_info[1].unit_name}]"
        if crs.equals(4326):
            x_label, y_label = y_label, x_label
    else:
        x_label, y_label = "x", "y"
    return x_label, y_label


def _get_crs(arr, geometry=None):
    if isinstance(arr, xr.DataArray):
        if geometry:
            return arr[geometry].crs
        elif np.all(shapely.is_valid_input(arr.data)):
            return arr.proj.crs
        return arr.xindexes[list(arr.xvec._geom_coords_all)[0]].crs
    return arr[geometry].crs if hasattr(arr[geometry], "crs") else arr.proj.crs


def _setup_legend(fig, cmap_params, label=None):
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    if "norm" in cmap_params:
        if not cmap_params["norm"]:
            cmap_params["norm"] = Normalize(
                vmin=cmap_params["vmin"], vmax=cmap_params["vmax"]
            )
        n_cmap = cm.ScalarMappable(norm=cmap_params["norm"], cmap=cmap_params["cmap"])
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
        fig.colorbar(
            n_cmap,
            cax=cbar_ax,
            label=label,
            extend=cmap_params["extend"],
        )
    else:
        if "cmap" not in cmap_params:
            cmap_params["cmap"] = "tab10"

        mn = 0
        mx = len(cmap_params["categories"]) - 1

        norm = Normalize(vmin=mn, vmax=mx)

        n_cmap = cm.ScalarMappable(cmap=cmap_params["cmap"], norm=norm)
        patches = []
        for i in range(len(cmap_params["categories"])):
            patches.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker="o",
                    markersize=10,
                    markerfacecolor=n_cmap.to_rgba(i),
                    markeredgewidth=0,
                )
            )
        fig.get_axes()[-1].legend(
            numpoints=1,
            loc="upper left",
            handles=patches,
            labels=list(cmap_params["categories"]),
            bbox_to_anchor=(1.1, 1.05),
        )


def _plot_faceted(arr, axs, row, col, hue, geometry, cmap_params=None, **kwargs):
    if row and col:
        for i_r, row_val in enumerate(arr[row]):
            for i_c, col_val in enumerate(arr[col]):
                _plot_single_panel(
                    arr.sel({row: [row_val.values], col: [col_val.values]}).drop_vars(
                        [col, row]
                    ),
                    axs[i_r, i_c],
                    hue,
                    geometry,
                    cmap_params,
                    **kwargs,
                )
                if i_r == 0:
                    axs[0, i_c].set_title(
                        f"{col} = {format_item(arr[col][i_c].values)}",
                        fontsize="small",
                    )
                if i_c == len(arr[col]) - 1:
                    axs[i_r, -1].yaxis.set_label_position("right")
                    axs[i_r, -1].set_ylabel(
                        f"{row} = {format_item(arr[row][i_r].values)}",
                        fontsize="small",
                        rotation=270,
                        labelpad=12,
                    )
    else:
        axs_flat = axs.flatten()
        for i_c, col_val in enumerate(arr[col]):
            _plot_single_panel(
                arr.sel({col: col_val}).drop_vars([col]),
                axs_flat[i_c],
                hue,
                geometry,
                cmap_params,
                **kwargs,
            )
            axs_flat[i_c].set_title(
                f"{col} = {format_item(arr[col][i_c].values)}", fontsize="small"
            )
        return arr[col].shape[0]  # Return used axes count


def _plot_single_panel(arr, ax, hue, geometry, cmap_params, **kwargs):
    if geometry:
        if hue:
            vals = arr[hue].squeeze().data
            sub = arr[geometry]
            if geometry in arr.xvec._geom_coords_all:
                sub = sub.drop_vars(geometry)
            sub.xvec.to_geodataframe(geometry=geometry).plot(
                vals,
                ax=ax,
                vmin=cmap_params.get("vmin", None),
                vmax=cmap_params.get("vmax", None),
                cmap=cmap_params.get("cmap", None),
                categories=cmap_params.get("categories", None),
                **kwargs,
            )
        else:
            arr[geometry].xvec.to_geodataframe().plot(ax=ax, **kwargs)
    elif np.all(shapely.is_valid_input(arr.data)):
        if hue:
            arr.xvec.to_geodataframe().reset_index().plot(
                hue,
                ax=ax,
                vmin=cmap_params.get("vmin", None),
                vmax=cmap_params.get("vmax", None),
                cmap=cmap_params.get("cmap", None),
                categories=cmap_params.get("categories", None),
                **kwargs,
            )
        else:
            arr.xvec.to_geodataframe().plot(ax=ax, **kwargs)
    else:
        name = arr.name if arr.name else "value"
        geometry = arr.xvec._geom_coords_all[0]
        arr.xvec.to_geodataframe(name=name, geometry=geometry).plot(
            name,
            ax=ax,
            vmin=cmap_params.get("vmin", None),
            vmax=cmap_params.get("vmax", None),
            cmap=cmap_params.get("cmap", None),
            categories=cmap_params.get("categories", None),
            **kwargs,
        )


def _plot(
    arr,
    *,
    row: Hashable | None = None,
    col: Hashable | None = None,
    col_wrap: int | None = None,
    hue: Hashable | None = None,
    subplot_kws: dict[str, Any] | None = None,
    figsize: Iterable[float] | None = None,
    geometry: Hashable | None = None,
    vmin=None,
    vmax=None,
    cmap=None,
    center=None,
    robust: bool = False,
    extend=None,
    levels=None,
    norm=None,
    **kwargs,
):
    # Calculate grid dimensions
    if row and col:
        n_rows, n_cols = arr[row].shape[0], arr[col].shape[0]
    elif col:
        total = arr[col].shape[0]
        n_rows = int(np.ceil(total / col_wrap)) if col_wrap else 1
        n_cols = col_wrap if col_wrap else total
    else:
        n_rows = n_cols = 1

    # Get CRS and axis labels
    crs = _get_crs(arr, geometry)
    x_label, y_label = _get_axis_labels(crs)

    # Setup figure and axes
    fig, axs = _setup_axes(n_rows, n_cols, arr, geometry, crs, subplot_kws, figsize)

    # Setup color parameters if needed
    if hue or (
        not hue
        and isinstance(arr, xr.DataArray)
        and not np.all(shapely.is_valid_input(arr.data))
    ):
        array = arr[hue].data if hue else arr.data

        # object is categorical, not supported by _determine_cmap_params
        if array.dtype != "object":
            cmap_params = _determine_cmap_params(
                array,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                center=center,
                robust=robust,
                extend=extend,
                levels=levels,
                norm=norm,
            )
        else:
            cmap_params = {"categories": np.unique(array)}
            if cmap:
                cmap_params["cmap"] = cmap
    else:
        cmap_params = {}

    # Handle simple case - single geometry with no faceting
    if not col and isinstance(arr, xr.DataArray) and n_cols == 1 and n_rows == 1:
        if arr.ndim == 2:
            arr = arr.squeeze()
        arr.xvec.to_geodataframe(geometry=geometry, name="plotting").plot(
            arr.values,
            ax=axs,
            vmin=cmap_params.get("vmin", None),
            vmax=cmap_params.get("vmax", None),
            cmap=cmap_params.get("cmap", None),
            categories=cmap_params.get("categories", None),
            **kwargs,
        )
        axs.set_xlabel(x_label, fontsize="small")
        axs.set_ylabel(y_label, fontsize="small")

        # Add colorbar if needed
        if hue:
            _setup_legend(fig, cmap_params, label=hue)
        elif (
            isinstance(arr, xr.DataArray)
            # and not geometry
            and not np.all(shapely.is_valid_input(arr))
        ):
            _setup_legend(fig, cmap_params, label=label_from_attrs(arr))
        return fig, axs

    if not col and geometry in arr.xvec._geom_coords_all:
        arr[geometry].drop_vars([geometry]).xvec.to_geodataframe(name="plotting").plot(
            ax=axs, **kwargs
        )
        axs.set_xlabel(x_label, fontsize="small")
        axs.set_ylabel(y_label, fontsize="small")
        return fig, axs

    # Handle faceted plotting
    used_axes = _plot_faceted(arr, axs, row, col, hue, geometry, cmap_params, **kwargs)

    # Add common labels
    axs = axs.reshape(n_rows, n_cols)
    for i in range(n_cols):
        axs[-1, i].set_xlabel(x_label, fontsize="small")
    for i in range(n_rows):
        axs[i, 0].set_ylabel(y_label, fontsize="small")

    # Remove unused axes
    if not row and col:
        for ax in axs.flatten()[used_axes:]:
            fig.delaxes(ax)

    # Add colorbar if needed
    if hue:
        _setup_legend(fig, cmap_params, label=hue)
    elif (
        isinstance(arr, xr.DataArray)
        and not geometry
        and not np.all(shapely.is_valid_input(arr))
    ):
        _setup_legend(fig, cmap_params, label=label_from_attrs(arr))

    return fig, axs
