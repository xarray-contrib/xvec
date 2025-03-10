.. _reference:


API reference
=============

The API reference provides an overview of all public objects, functions and
methods and Xarray accessors implemented in Xvec.

Indexing
--------

.. currentmodule:: xvec

.. autosummary::
   :toctree: generated/

   GeometryIndex
   GeometryIndex.crs
   GeometryIndex.sindex


.. currentmodule:: xarray

Dataset.xvec
------------

.. _dsattr:

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

    Dataset.xvec.geom_coords
    Dataset.xvec.geom_coords_indexed


.. _dsmeth:

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    Dataset.xvec.set_geom_indexes
    Dataset.xvec.is_geom_variable
    Dataset.xvec.set_crs
    Dataset.xvec.to_crs
    Dataset.xvec.query
    Dataset.xvec.to_geodataframe
    Dataset.xvec.to_geopandas
    Dataset.xvec.extract_points
    Dataset.xvec.zonal_stats
    Dataset.xvec.encode_cf
    Dataset.xvec.decode_cf
    Dataset.xvec.encode_wkb
    Dataset.xvec.decode_wkb
    Dataset.xvec.summarize_geometry
    Dataset.xvec.plot


DataArray.xvec
--------------

.. _daattr:

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

    DataArray.xvec.geom_coords
    DataArray.xvec.geom_coords_indexed


.. _dameth:

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    DataArray.xvec.set_geom_indexes
    DataArray.xvec.is_geom_variable
    DataArray.xvec.to_crs
    DataArray.xvec.set_crs
    DataArray.xvec.query
    DataArray.xvec.mask
    DataArray.xvec.to_geodataframe
    DataArray.xvec.to_geopandas
    DataArray.xvec.extract_points
    DataArray.xvec.zonal_stats
    DataArray.xvec.encode_wkb
    DataArray.xvec.decode_wkb
    DataArray.xvec.summarize_geometry
    DataArray.xvec.plot