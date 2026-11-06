.. NeoN documentation master file, created by
   sphinx-quickstart on Sat Dec 16 15:22:11 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NeoN!
===================

NeoN is an open-source ``C++20`` library for developing high-performance computational fluid dynamics (CFD) software on heterogeneous computing systems, from multi-core CPUs to NVIDIA, AMD and Intel GPUs.
It provides data structures, parallel algorithms and a domain-specific language (DSL) for the finite volume method (FVM).

Statement of Need
^^^^^^^^^^^^^^^^^

Many widely used CFD codes were originally designed for single-core or moderately parallel CPUs.
On modern high-performance computing systems, which increasingly rely on GPUs, CFD workloads are typically limited by memory bandwidth rather than by compute capability, and porting existing codes to accelerators often requires backend-specific rewrites.
At the same time, the finite-volume abstractions of established frameworks are usually tied to a specific solver infrastructure and are difficult to reuse elsewhere.

NeoN addresses this gap by providing a reusable, performance-portable finite-volume core:

* **Equation-level DSL:** partial differential equations are written close to their mathematical form, for example

  .. code-block:: cpp

     auto expr = dsl::imp::ddt(U) + dsl::imp::div(phi, U) - dsl::imp::laplacian(nu, U);

  Here, ``imp`` denotes implicit operators, which contribute coefficients to the system matrix, while ``exp`` denotes explicit operators, which contribute to the right-hand side.
  Expressions are evaluated lazily, deferring matrix assembly until it is needed and avoiding unnecessary temporaries.
* **Discretisation and assembly:** discretisation schemes generate stencil coefficients that are assembled into standard sparse matrix formats (e.g., CSR), with the resulting linear systems solved by external libraries such as `Ginkgo <https://github.com/ginkgo-project/ginkgo>`_.
* **Performance portability:** `Kokkos <https://github.com/kokkos/kokkos>`_ provides execution on serial CPUs, multi-threaded CPUs and GPUs from a single code base, while MPI supports distributed-memory parallelism.
* **Python access:** the DSL can also be used from Python, see :doc:`python_bindings`.

Target Audience
^^^^^^^^^^^^^^^

NeoN is intended for

* developers of new CFD solvers that should run efficiently on CPUs and GPUs without architecture-specific code;
* researchers developing and testing discretisation schemes and numerical methods for the finite volume method;
* maintainers of existing CFD frameworks who want to modernise them incrementally. `NeoFOAM <https://github.com/exasim-project/NeoFOAM>`_ demonstrates this for OpenFOAM-based solvers;
* Python users who want to prototype finite-volume discretisations.

NeoN is not a standalone CFD application.
Time loops, solver control and I/O are handled by the application that uses NeoN; complete solvers built on NeoN are available in `NeoFOAM <https://github.com/exasim-project/NeoFOAM>`_.

Key Features
^^^^^^^^^^^^

* extensively unit-tested;
* extensible via runtime-selectable components (e.g., discretisation schemes and boundary conditions), see :doc:`basics/registerclass`;
* designed to be integrated into other CFD frameworks through its API, as demonstrated by `NeoFOAM <https://github.com/exasim-project/NeoFOAM>`_.

NeoN is a community-driven project and we welcome contributions from everyone, see :doc:`contributing`.

Table of Contents
^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 2

   self
   installation
   python_bindings
   contributing
   basics/index
   dsl/index
   timeIntegration/index
   finiteVolume/cellCentred/index
   datastructures/index
   linearAlgebra/index
   mpi_architecture
   ci

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
