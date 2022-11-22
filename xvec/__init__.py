from importlib.metadata import PackageNotFoundError, version

from .index import GeometryIndex  # noqa

try:
    __version__ = version("xvec")
except PackageNotFoundError:  # noqa
    # package is not installed
    pass
