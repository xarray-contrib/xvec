import matplotlib.pyplot as plt
import numpy as np
import shapely
import xarray as xr
import xproj  # noqa: F401
from matplotlib import cm
from matplotlib.colors import Normalize
from xarray.plot.utils import _determine_cmap_params, label_from_attrs


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
    **kwargs,
):
    if row and col:
        if len(arr.dims) != 3:
            raise ValueError(
                "To plot with row and col, the DataArray needs to have "
                "3 dimensions, one of which is indexed by xvec.GeometryIndex."
            )
        n_rows = arr[row].shape[0]
        n_cols = arr[col].shape[0]
    elif col:
        if len(arr.dims) != 2:
            raise ValueError(
                "To plot with col, the DataArray needs to have "
                "2 dimensions, one of which is indexed by xvec.GeometryIndex."
            )
        total = arr[col].shape[0]
        if col_wrap:
            n_rows = int(np.ceil(total / col_wrap))
            n_cols = col_wrap
        else:
            n_rows = 1
            n_cols = total
    else:
        n_rows = 1
        n_cols = 1

    if isinstance(arr, xr.DataArray):
        if geometry:
            crs = arr[geometry].crs
        elif np.all(shapely.is_valid_input(arr.data)):
            crs = arr.proj.crs
        else:
            crs = arr[list(arr.xvec.geom_coords)[0]].crs
    else:
        crs = arr[geometry].crs if hasattr(arr[geometry], "crs") else arr.proj.crs

    if crs:
        x_label = f"{crs.axis_info[0].name}\n[{crs.axis_info[0].unit_name}]"
        y_label = f"{crs.axis_info[1].name}\n[{crs.axis_info[1].unit_name}]"

        if crs.equals(4326):
            x_label, y_label = y_label, x_label
    else:
        x_label = "x"
        y_label = "y"

    if figsize is None:
        cbar_space = 1
        figsize = (n_cols * size * aspect + cbar_space, n_rows * size)

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        sharex=True,
        sharey=True,
        figsize=figsize,
        subplot_kw=subplot_kws,
    )

    if geometry is not None:
        if (
            not col and geometry in arr.xvec._geom_coords_all
        ):  # Dataset with geometry variable
            arr[geometry].drop_vars([geometry]).xvec.to_geodataframe().plot(ax=axs)
            axs.set_xlabel(x_label, fontsize="small")
            axs.set_ylabel(y_label, fontsize="small")
        else:
            if hue:
                cmap_params = _determine_cmap_params(arr[hue].data, **kwargs)
            axs = axs.flatten()
            for i_c, col_val in enumerate(arr[col]):
                sliced = arr.sel({col: col_val}).drop_vars([col])
                if hue:
                    vals = sliced[hue].data
                    if geometry in arr.coords:
                        arr[geometry].drop_vars([geometry]).xvec.to_geodataframe().plot(
                            vals,
                            ax=axs[i_c],
                            vmin=cmap_params["vmin"],
                            vmax=cmap_params["vmax"],
                            cmap=cmap_params["cmap"],
                        )
                    else:
                        sliced[geometry].xvec.to_geodataframe().plot(
                            vals,
                            ax=axs[i_c],
                            vmin=cmap_params["vmin"],
                            vmax=cmap_params["vmax"],
                            cmap=cmap_params["cmap"],
                        )
                else:
                    sliced[geometry].xvec.to_geodataframe().plot(ax=axs[i_c])

                axs[i_c].set_title(f"{col} = {arr[col][i_c].item()}", fontsize="small")

            axs = axs.reshape(n_rows, n_cols)
            for i in range(n_cols):
                axs[-1, i].set_xlabel(x_label, fontsize="small")
            for i in range(n_rows):
                axs[i, 0].set_ylabel(y_label, fontsize="small")

            used_axes = arr[col].shape[0]
            for ax in axs.flatten()[used_axes:]:
                fig.delaxes(ax)

            if hue:
                if not cmap_params["norm"]:
                    cmap_params["norm"] = Normalize(
                        vmin=cmap_params["vmin"], vmax=cmap_params["vmax"]
                    )

                n_cmap = cm.ScalarMappable(
                    norm=cmap_params["norm"], cmap=cmap_params["cmap"]
                )

                fig.subplots_adjust(right=0.85)
                cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
                fig.colorbar(
                    n_cmap,
                    cax=cbar_ax,
                    label=hue,
                    extend=cmap_params["extend"],
                )

    elif isinstance(arr, xr.DataArray) and np.all(
        shapely.is_valid_input(arr.data)
    ):  # data array with geometry
        axs = axs.flatten()
        for i_c, col_val in enumerate(arr[col]):
            arr.sel({col: col_val}).drop_vars([col]).xvec.to_geodataframe().plot(
                ax=axs[i_c]
            )

            axs[i_c].set_title(f"{col} = {arr[col][i_c].item()}", fontsize="small")

        axs = axs.reshape(n_rows, n_cols)
        for i in range(n_cols):
            axs[-1, i].set_xlabel(x_label, fontsize="small")
        for i in range(n_rows):
            axs[i, 0].set_ylabel(y_label, fontsize="small")

        used_axes = arr[col].shape[0]
        for ax in axs.flatten()[used_axes:]:
            fig.delaxes(ax)

    else:
        name = arr.name if arr.name else "value"

        cmap_params = _determine_cmap_params(arr.data, **kwargs)

        if row and col:
            for i_r, row_val in enumerate(arr[row]):
                for i_c, col_val in enumerate(arr[col]):
                    arr.sel({row: row_val, col: col_val}).drop_vars(
                        [col, row]
                    ).xvec.to_geodataframe(name=name).plot(
                        name,
                        ax=axs[i_r, i_c],
                        vmin=cmap_params["vmin"],
                        vmax=cmap_params["vmax"],
                        cmap=cmap_params["cmap"],
                    )

            for i in range(n_cols):
                axs[0, i].set_title(f"{col} = {arr[col][i].item()}", fontsize="small")
                axs[-1, i].set_xlabel(x_label, fontsize="small")

            for i in range(n_rows):
                axs[i, -1].yaxis.set_label_position("right")
                axs[i, -1].set_ylabel(
                    f"{row} = {arr[row][i].item()}",
                    fontsize="small",
                    rotation=270,
                    labelpad=12,
                )
                axs[i, 0].set_ylabel(y_label, fontsize="small")

        else:
            axs = axs.flatten()
            for i_c, col_val in enumerate(arr[col]):
                arr.sel({col: col_val}).drop_vars([col]).xvec.to_geodataframe(
                    name=name
                ).plot(
                    name,
                    ax=axs[i_c],
                    vmin=cmap_params["vmin"],
                    vmax=cmap_params["vmax"],
                    cmap=cmap_params["cmap"],
                )
                axs[i_c].set_title(f"{col} = {arr[col][i_c].item()}", fontsize="small")

            axs = axs.reshape(n_rows, n_cols)
            for i in range(n_cols):
                axs[-1, i].set_xlabel(x_label, fontsize="small")
            for i in range(n_rows):
                axs[i, 0].set_ylabel(y_label, fontsize="small")

            used_axes = arr[col].shape[0]
            for ax in axs.flatten()[used_axes:]:
                fig.delaxes(ax)

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
            label=label_from_attrs(arr),
            extend=cmap_params["extend"],
        )
    return fig, axs
