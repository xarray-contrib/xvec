from importlib.metadata import PackageNotFoundError, version

from .accessor import XvecAccessor  # noqa: F401
from .index import GeometryIndex  # noqa: F401

try:
    __version__ = version("xvec")
except PackageNotFoundError:  # noqa
    # package is not installed
    pass
