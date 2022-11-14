# Vector data cubes for Xarray

> Where raster data cubes refer to data cubes with raster (x- and y-, or lon- and lat-) dimensions, vector data cubes are n-D arrays that have (at least) a single spatial dimension that maps to a set of (2-D) vector geometries. ([Edzer Pebesma](https://r-spatial.org/r/2022/09/12/vdc.html))

Xvec combines [Xarray](http://xarray.pydata.org) n-D arrays and [shapely 2.0](https://shapely.readthedocs.io/en/latest/) planar vector geometries to create a support for vector data cubes in Python. See [this post](https://r-spatial.org/r/2022/09/12/vdc.html) by Edzer Pebesma on an introduction of the concept.

Xvec is in an early stage of development and we do not advise to use it yet as the API will likely change.

## Resources

- <https://github.com/corteva/rioxarray/issues/588>
- <https://github.com/dcherian/crsindex/blob/main/crsindex.ipynb>
- <https://hackmd.io/Zxw_zCa7Rbynx_iJu6Y3LA?view>
