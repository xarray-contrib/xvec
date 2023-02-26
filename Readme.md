# Vector data cubes for Xarray

> Where raster data cubes refer to data cubes with raster (x- and y-, or lon- and lat-) dimensions, vector data cubes are n-D arrays that have (at least) a single spatial dimension that maps to a set of (2-D) vector geometries. ([Edzer Pebesma](https://r-spatial.org/r/2022/09/12/vdc.html))

Xvec combines [Xarray](http://xarray.pydata.org) n-D arrays and [shapely 2](https://shapely.readthedocs.io/en/latest/) planar vector geometries to create a support for vector data cubes in Python. See [this post](https://r-spatial.org/r/2022/09/12/vdc.html) by Edzer Pebesma on an introduction of the concept or the introduction of their implementation in Xvec in our [documentation](https://xvec.readthedocs.io/en/latest/intro.html).

## Project status

The project is in the early stage of development and its API may still change.

## Installing

You can install Xvec from PyPI using `pip` or from conda-forge using `mamba` or `conda`:

```sh
pip install xvec
```

Or (recommended):

```sh
mamba install xvec -c conda-forge
```

### Development version

The development version can be installed from GitHub.

```sh
pip install git+https://github.com/xarray-contrib/xvec.git
```

We recommend installing its dependencies using `mamba` or `conda` before.

```sh
mamba install xarray shapely pyproj -c conda-forge
```
