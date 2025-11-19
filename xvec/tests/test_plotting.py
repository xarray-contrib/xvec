import geodatasets
import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr
import xproj  # noqa: F401

import xvec  # noqa: F401

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib.testing.decorators import image_comparison  # noqa: E402


@pytest.fixture()
def aggregated():
    ds = xr.tutorial.open_dataset("eraint_uvz")
    counties = gpd.read_file(geodatasets.get_path("geoda natregimes")).to_crs(4326)

    return ds.xvec.zonal_stats(
        counties.geometry,
        x_coords="longitude",
        y_coords="latitude",
        all_touched=True,
        method="rasterize",
    )


@pytest.fixture()
def glaciers():
    glaciers_df = gpd.read_file(
        "https://github.com/loreabad6/post/raw/refs/heads/main/inst/extdata/svalbard.gpkg"
    )
    glaciers_df["year"] = pd.to_datetime(glaciers_df["year"].astype(int), format="%Y")
    return (
        glaciers_df.set_index(["name", "year"])
        .to_xarray()
        .proj.assign_crs(spatial_ref=glaciers_df.crs)
    )


@image_comparison(baseline_images=["col"], extensions=["png"], style=[], tol=0.01)
def test_col(aggregated):
    f, ax = aggregated.v.sel(month=1).xvec.plot(col="level")

    assert ax.shape == (1, 3)
    ax0 = ax[0][0]
    assert ax0.get_xlabel() == "Geodetic longitude\n[degree]"
    assert ax0.get_ylabel() == "Geodetic latitude\n[degree]"
    assert ax0.get_title() == "level = 200"


@image_comparison(baseline_images=["colwrap"], extensions=["png"], style=[], tol=0.01)
def test_colwrap(aggregated):
    f, ax = aggregated.u.sel(month=1).xvec.plot(col="level", col_wrap=2)

    assert ax.shape == (2, 2)
    assert ax[1][0].get_xlabel() == "Geodetic longitude\n[degree]"
    assert ax[0][0].get_ylabel() == "Geodetic latitude\n[degree]"
    assert ax[0][0].get_title() == "level = 200"


@image_comparison(baseline_images=["col_row"], extensions=["png"], style=[], tol=0.01)
def test_col_row(aggregated):
    f, ax = aggregated.v.xvec.plot(col="month", row="level")

    assert ax.shape == (3, 2)
    assert ax[2][0].get_xlabel() == "Geodetic longitude\n[degree]"
    assert ax[0][0].get_ylabel() == "Geodetic latitude\n[degree]"
    assert ax[0][0].get_title() == "month = 1"
    assert ax[0][1].get_title() == "month = 7"
    assert ax[0][1].get_ylabel() == "level = 200"


@image_comparison(baseline_images=["1d"], extensions=["png"], style=[], tol=0.01)
def test_1d(aggregated):
    f, ax = aggregated.z.sel(level=200, month=1).xvec.plot()

    assert ax.get_xlabel() == "Geodetic longitude\n[degree]"
    assert ax.get_ylabel() == "Geodetic latitude\n[degree]"


@image_comparison(
    baseline_images=["void_dimension"], extensions=["png"], style=[], tol=0.01
)
def test_void_dimension():
    ds = xr.tutorial.open_dataset("eraint_uvz").load()
    counties = gpd.read_file(geodatasets.get_path("geoda natregimes")).to_crs(4326)

    ds.sel(month=1, level=[200]).z.xvec.zonal_stats(
        counties.geometry,
        x_coords="longitude",
        y_coords="latitude",
        all_touched=True,
        method="rasterize",
    ).xvec.plot()


@image_comparison(baseline_images=["unnamed"], extensions=["png"], style=[], tol=0.01)
def test_unnamed():
    ds = xr.tutorial.open_dataset("eraint_uvz").load()
    counties = gpd.read_file(geodatasets.get_path("geoda natregimes")).to_crs(4326)

    arr = ds.sel(month=1, level=[200]).z
    arr.name = None

    arr.xvec.zonal_stats(
        counties.geometry,
        x_coords="longitude",
        y_coords="latitude",
        all_touched=True,
        method="rasterize",
    ).sel(level=200).xvec.plot()


@image_comparison(baseline_images=["var_geom"], extensions=["png"], style=[], tol=0.01)
def test_var_geom(glaciers):
    f, ax = glaciers.geometry.xvec.plot(col="year")

    assert ax.shape == (1, 3)
    ax0 = ax[0][0]
    assert ax0.get_xlabel() == "Easting\n[metre]"
    assert ax0.get_ylabel() == "Northing\n[metre]"
    assert ax0.get_title() == "year = 1936-01-01"


@image_comparison(
    baseline_images=["var_geom_facet"], extensions=["png"], style=[], tol=0.01
)
def test_var_geom_facet(glaciers):
    f, ax = glaciers.geometry.xvec.plot(col="name", row="year")

    assert ax.shape == (3, 5)
    assert ax[2][0].get_xlabel() == "Easting\n[metre]"
    assert ax[0][0].get_ylabel() == "Northing\n[metre]"
    assert ax[0][0].get_title() == "name = Austre Brøggerbreen"
    assert ax[0][-1].get_ylabel() == "year = 1936-01-01"


@image_comparison(
    baseline_images=["var_geom_ds"], extensions=["png"], style=[], tol=0.01
)
def test_var_geom_ds(glaciers):
    f, ax = glaciers.xvec.plot(col="year", geometry="geometry")

    assert ax.shape == (1, 3)
    ax0 = ax[0][0]
    assert ax0.get_xlabel() == "Easting\n[metre]"
    assert ax0.get_ylabel() == "Northing\n[metre]"
    assert ax0.get_title() == "year = 1936-01-01"


@image_comparison(baseline_images=["hue"], extensions=["png"], style=[], tol=0.01)
def test_hue(glaciers):
    f, ax = glaciers.xvec.plot(col="year", geometry="geometry", hue="fwidth")

    assert ax.shape == (1, 3)
    ax0 = ax[0][0]
    assert ax0.get_xlabel() == "Easting\n[metre]"
    assert ax0.get_ylabel() == "Northing\n[metre]"
    assert ax0.get_title() == "year = 1936-01-01"


@image_comparison(
    baseline_images=["geom_switching"], extensions=["png"], style=[], tol=0.01
)
def test_geom_switching(glaciers):
    glaciers_w_sum = glaciers.xvec.summarize_geometry(
        dim="name", geom_array="geometry", aggfunc="concave_hull", ratio=0.25
    )

    f, ax = glaciers_w_sum.xvec.plot(geometry="summary_geometry")
    assert ax.get_xlabel() == "Easting\n[metre]"
    assert ax.get_ylabel() == "Northing\n[metre]"


@image_comparison(
    baseline_images=["categorical"],
    extensions=["png"],
    style=[],
    savefig_kwarg=dict(bbox_inches="tight"),
)
def test_categorical(glaciers):
    f, ax = glaciers.xvec.plot(col="year", geometry="geometry", hue="name")

    assert ax.shape == (1, 3)
    ax0 = ax[0][0]
    assert ax0.get_xlabel() == "Easting\n[metre]"
    assert ax0.get_ylabel() == "Northing\n[metre]"
    assert ax0.get_title() == "year = 1936-01-01"


@image_comparison(
    baseline_images=["single_custom_geometry"],
    extensions=["png"],
    style=[],
    savefig_kwarg=dict(bbox_inches="tight"),
)
def test_single_custom_geometry(glaciers):
    glaciers = glaciers.xvec.summarize_geometry(
        dim="name", geom_array="geometry", aggfunc="concave_hull", ratio=0.25
    )
    f, ax = glaciers["length"].sum("year").xvec.plot(geometry="summary_geometry")

    assert ax.get_xlabel() == "Easting\n[metre]"
    assert ax.get_ylabel() == "Northing\n[metre]"
