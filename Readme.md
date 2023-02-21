# Vector data cubes for Xarray

> Where raster data cubes refer to data cubes with raster (x- and y-, or lon- and lat-) dimensions, vector data cubes are n-D arrays that have (at least) a single spatial dimension that maps to a set of (2-D) vector geometries. ([Edzer Pebesma](https://r-spatial.org/r/2022/09/12/vdc.html))

Xvec combines [Xarray](http://xarray.pydata.org) n-D arrays and [shapely 2](https://shapely.readthedocs.io/en/latest/) planar vector geometries to create a support for vector data cubes in Python. See [this post](https://r-spatial.org/r/2022/09/12/vdc.html) by Edzer Pebesma on an introduction of the concept or the introduction of their implementation in Xvec in our [documentation](https://xvec.readthedocs.io/en/latest/intro.html).

## Installing

Xvec is not released yet and can be installed from GitHub.

```sh
pip install git+https://github.com/martinfleis/xvec.git
```

We recommend installing its dependencies using `mamba` or `conda` before.

```sh
mamba install xarray shapely pyproj -c conda-forge
```
