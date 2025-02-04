import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from xarray.plot.utils import _determine_cmap_params, label_from_attrs


def _plot(
    arr,
    row=None,
    col=None,
    col_wrap=None,
    ax=None,
    subplot_kws=None,
    **kwargs,
):
    n_rows = arr[row].shape[0]
    n_cols = arr[col].shape[0]
    fig, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(8, 6))
    name = arr.name if arr.name else "value"

    cmap_params = _determine_cmap_params(arr.data, **kwargs)
    crs = arr[list(arr.xvec.geom_coords)[0]].crs

    if crs:
        x_label = f"{crs.axis_info[0].name}\n[{crs.axis_info[0].unit_name}]"
        y_label = f"{crs.axis_info[1].name}\n[{crs.axis_info[1].unit_name}]"
    else:
        x_label = "x"
        y_label = "y"

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
            f"{row} = {arr[row][i].item()}", fontsize="small", rotation=270, labelpad=12
        )
        axs[i, 0].set_ylabel(y_label, fontsize="small")

    if not cmap_params["norm"]:
        cmap_params["norm"] = Normalize(
            vmin=cmap_params["vmin"], vmax=cmap_params["vmax"]
        )
    n_cmap = cm.ScalarMappable(norm=cmap_params["norm"], cmap=cmap_params["cmap"])
    fig.colorbar(
        n_cmap, ax=axs, label=label_from_attrs(arr), extend=cmap_params["extend"]
    )
    return fig, axs
