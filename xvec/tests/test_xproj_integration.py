import shapely
import xproj  # noqa: F401


def test_xproj_crs(geom_dataset):
    assert geom_dataset.xindexes["geom"].crs.equals(geom_dataset.proj("geom").crs)


def test_xproj_map_crs(geom_dataset_no_crs):
    ds_with_crsindex = geom_dataset_no_crs.proj.assign_crs(spatial_ref=26915)

    # set crs
    ds = ds_with_crsindex.proj.map_crs(spatial_ref=["geom"])
    assert ds.proj("geom").crs.equals(ds.proj("spatial_ref").crs)

    # to crs
    geom_array_4326 = shapely.from_wkt(
        ["POINT (-97.488735 0.000018)", "POINT (-97.488717 0.000036)"]
    )

    ds_with_crsindex2 = ds.proj.assign_crs(spatial_ref=4326, allow_override=True)
    ds2 = ds_with_crsindex2.proj.map_crs(
        spatial_ref=["geom"], transform=True, allow_override=True
    )
    assert ds2.proj("geom").crs.equals(ds2.proj("spatial_ref").crs)
    assert shapely.equals_exact(geom_array_4326, ds2.geom, tolerance=1e-5).all()
