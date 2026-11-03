.. NeoFOAM documentation master file, created by
   sphinx-quickstart on Sat Dec 16 15:22:11 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NeoFOAM!
===================

The NeoFOAM project has set itself the goal of bringing modern software development methods to the core libraries of OpenFOAM.
By reimplementing the `libfiniteVolume` and `libOpenFOAM` we want to deliver a code that:

* is compliant with modern C++20;
* is extensively unit-tested;
* is platform portable and GPU ready;
* is highly extensible via Plugins

We aim for a high level of interoperability with OpenFOAM, however, if reasonable, NeoFOAM might deviate from the OpenFOAM API.
NeoFOAM is a community-driven project and we welcome contributions from everyone.

Table of Contents
^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 2

   self
   installation
   contributing
   basics/index
   dsl/index
   timeIntegration/index
   finiteVolume/cellCentred/index
   datastructures/index
   mpi_architecture

Compatibility with OpenFOAM
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We are aiming for a high level of compatibility with OpenFOAM.
However, we don't expect binary or ABI compatibility.
This means that NeoFOAM won't produce a `libfiniteVolume.so` and `libOpenFOAM.so` which could serve as a plugin replacement for existing `libfiniteVolume.so` and `libOpenFOAM.so`.
Instead, we aim for source compatibility, i.e. the possibility to compile application OpenFOAM code like pimpleFoam and others against the NeoFOAM libraries.
This approach is demonstrated in the `FoamAdapter <https://github.com/exasim-project/FoamAdapter>`_ repository.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
