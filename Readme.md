# Vector data cubes for Xarray

In geospatial analysis, data cubes can be of two sorts. The first is a raster data cube, typically represented by an [Xarray](https://docs.xarray.dev/en/stable/) DataArray indexed either by `x` or `y` dimensions or `latitude` and `longitude`. The second is a vector data cube, which is an n-D array that has either at least one dimension indexed by a 2-D array of vector geometries ([Pebesma, 2022](https://r-spatial.org/r/2022/09/12/vdc.html)) or contains geometries as variables (e.g. moving features or time-evolving shapes), possibly both.

We can distinguish between two types of geometries in a DataArray or Dataset:

- **coordinate geometry** - an array (typically one dimensional) is used as coordinates along one or more dimensions. A typical example would be an outcome of zonal statistics of a multi-dimensional raster, avoiding the need for _flattenning_ of the array to a data frame.
- **variable geometry** - an array (typicially multi-dimensional) is used as a variable within a DataArray. This may encode evolving shapes of lava flows in time, trajectories, or growth of city limits.

The Xvec package brings support for both of these to the Xarray ecosystem. It uses [Shapely](https://shapely.readthedocs.io/en/stable/) package, allowing a seamless interface between Xvec and [GeoPandas](https://geopandas.org/). See [this post](https://r-spatial.org/r/2022/09/12/vdc.html) by Edzer Pebesma on an introduction of the concept of coordinate geometry or [introduction](https://xvec.readthedocs.io/en/latest/intro.html) page in Xvec documentation.

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
