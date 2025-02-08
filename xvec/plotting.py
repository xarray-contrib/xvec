import matplotlib.pyplot as plt
import numpy as np
import shapely
import xarray as xr
import xproj  # noqa: F401
from matplotlib import cm
from matplotlib.colors import Normalize
from xarray.plot.utils import _determine_cmap_params, label_from_attrs


def _setup_axes(n_rows, n_cols, size, aspect, subplot_kws, figsize):
    if figsize is None:
        cbar_space = 1
        figsize = (n_cols * size * aspect + cbar_space, n_rows * size)
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
        return arr[list(arr.xvec._geom_coords_all)[0]].crs
    return arr[geometry].crs if hasattr(arr[geometry], "crs") else arr.proj.crs


def _setup_colorbar(fig, cmap_params, label=None):
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


def _plot_faceted(arr, axs, row, col, hue, geometry, alpha, cmap_params=None):
    if row and col:
        for i_r, row_val in enumerate(arr[row]):
            for i_c, col_val in enumerate(arr[col]):
                _plot_single_panel(
                    arr.sel({row: [row_val.item()], col: [col_val.item()]}).drop_vars(
                        [col, row]
                    ),
                    axs[i_r, i_c],
                    hue,
                    geometry,
                    alpha,
                    cmap_params,
                )
                if i_r == 0:
                    axs[0, i_c].set_title(
                        f"{col} = {arr[col][i_c].item()}", fontsize="small"
                    )
                if i_c == len(arr[col]) - 1:
                    axs[i_r, -1].yaxis.set_label_position("right")
                    axs[i_r, -1].set_ylabel(
                        f"{row} = {arr[row][i_r].item()}",
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
                alpha,
                cmap_params,
            )
            axs_flat[i_c].set_title(f"{col} = {arr[col][i_c].item()}", fontsize="small")
        return arr[col].shape[0]  # Return used axes count


def _plot_single_panel(arr, ax, hue, geometry, alpha, cmap_params):
    if geometry:
        if hue:
            vals = arr[hue].squeeze().data
            sub = arr[geometry]
            if geometry in arr.xvec._geom_coords_all:
                sub = sub.drop_vars(geometry)
            sub.xvec.to_geodataframe(geometry=geometry).plot(
                vals,
                ax=ax,
                vmin=cmap_params["vmin"],
                vmax=cmap_params["vmax"],
                cmap=cmap_params["cmap"],
                alpha=alpha,
            )
        else:
            arr[geometry].xvec.to_geodataframe().plot(ax=ax, alpha=alpha)
    elif np.all(shapely.is_valid_input(arr.data)):
        if hue:
            arr.xvec.to_geodataframe().reset_index().plot(
                hue,
                ax=ax,
                vmin=cmap_params["vmin"],
                vmax=cmap_params["vmax"],
                cmap=cmap_params["cmap"],
                alpha=alpha,
            )
        else:
            arr.xvec.to_geodataframe().plot(ax=ax, alpha=alpha)
    else:
        name = arr.name if arr.name else "value"
        geometry = arr.xvec._geom_coords_all[0]
        arr.xvec.to_geodataframe(name=name, geometry=geometry).plot(
            name,
            ax=ax,
            vmin=cmap_params["vmin"],
            vmax=cmap_params["vmax"],
            cmap=cmap_params["cmap"],
        )


def _plot(
    arr,
    row=None,
    col=None,
    col_wrap=None,
    ax=None,
    hue=None,
    subplot_kws=None,
    figsize=None,
    aspect=1,
    size=3,
    geometry=None,
    alpha=None,
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

    # Setup figure and axes
    fig, axs = _setup_axes(n_rows, n_cols, size, aspect, subplot_kws, figsize)

    # Get CRS and axis labels
    crs = _get_crs(arr, geometry)
    x_label, y_label = _get_axis_labels(crs)

    # Setup color parameters if needed
    cmap_params = _determine_cmap_params(arr[hue].data, **kwargs) if hue else None
    if (
        not hue
        and isinstance(arr, xr.DataArray)
        and not np.all(shapely.is_valid_input(arr.data))
    ):
        cmap_params = _determine_cmap_params(arr.data, **kwargs)

    # Handle simple case - single geometry with no faceting
    if not col and geometry in arr.xvec._geom_coords_all:
        arr[geometry].drop_vars([geometry]).xvec.to_geodataframe().plot(
            ax=axs, alpha=alpha
        )
        axs.set_xlabel(x_label, fontsize="small")
        axs.set_ylabel(y_label, fontsize="small")
        return fig, axs

    # Handle faceted plotting
    used_axes = _plot_faceted(arr, axs, row, col, hue, geometry, alpha, cmap_params)

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
        _setup_colorbar(fig, cmap_params, label=hue)
    elif (
        isinstance(arr, xr.DataArray)
        and not geometry
        and not np.all(shapely.is_valid_input(arr))
    ):
        _setup_colorbar(fig, cmap_params, label=label_from_attrs(arr))

    return fig, axs
