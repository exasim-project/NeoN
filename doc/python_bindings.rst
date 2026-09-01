Python bindings (``neon_pde``)
==============================

NeoN provides Python bindings that expose the core containers, executors,
fields and DSL to Python. The distribution is named ``neon_pde`` on PyPI, but
the import package is ``neon``:

.. code-block:: python

   import neon
   neon.initialize()
   exec = neon.CPUExecutor()
   v = neon.ScalarVector(exec, 10, 1.0)
   neon.finalize()

The package supports CPython 3.9–3.13 and depends on NumPy.

.. contents::
   :local:
   :depth: 1

Installing
----------

CPU wheels from PyPI
^^^^^^^^^^^^^^^^^^^^^

Pre-built CPU wheels are published to PyPI for Linux (x86-64 and ARM64),
Windows (AMD64) and macOS (Apple Silicon and Intel). These are the default
wheels for most users and require no CUDA toolkit or GPU:

.. code-block:: bash

   pip install neon_pde

CPU wheels are built with the serial and OpenMP (CPU) Kokkos back ends. MPI and
the optional HPC dependencies (PETSc, ADIOS2, SUNDIALS) are disabled so the
package installs without external HPC libraries.

Conda packages from prefix.dev
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same releases are published as conda packages to the
`prefix.dev <https://prefix.dev>`_ channel, which is the most convenient way to
get NeoN into a `pixi <https://pixi.sh>`_ project. The conda package is named
``neon-pde`` (the import package is still ``neon``):

.. code-block:: bash

   pixi add neon-pde -c https://prefix.dev/exasim-project -c conda-forge

or, in ``pixi.toml``:

.. code-block:: toml

   [project]
   channels = ["https://prefix.dev/exasim-project", "conda-forge"]

   [dependencies]
   neon-pde = "*"

The channel provides CPython 3.10-3.13 packages for ``linux-64``,
``linux-aarch64``, ``osx-64`` and ``osx-arm64``. There is no ``win-64`` conda
package, and no Python 3.9 one — use the PyPI wheel for those. (3.9 is excluded
because conda-forge builds ``nanobind``, which the stub generation step needs,
only for Python 3.10 and newer.)

Unlike the wheels, the conda package installs the NeoN C++ runtime into
``$CONDA_PREFIX/lib`` and its headers and CMake package files into
``$CONDA_PREFIX/include`` and ``$CONDA_PREFIX/lib/cmake/NeoN``, so a C++ project
in the same environment can consume it with ``find_package(NeoN)``.

GPU conda packages carry the accelerator in their build string, so the flavour is
selected by matching on it:

.. code-block:: bash

   # CUDA 12.8, CPython 3.12, NVIDIA Ampere (sm_80), linux-64
   pixi add "neon-pde=*=cuda_py312*" -c https://prefix.dev/exasim-project -c conda-forge

The CUDA package depends on the ``__cuda`` virtual package, so it only resolves
on a machine whose NVIDIA driver is new enough. A ROCm flavour
(``rocm_py312*``) can be built on demand, but it ships Kokkos HIP **without** the
Ginkgo solver backend: conda-forge provides ``hip-devel`` and ``hipcc`` but none
of the ROCm math libraries (rocBLAS, rocSPARSE, rocThrust, rocPRIM) that
Ginkgo's HIP backend requires. It is therefore not part of a tagged release and
must be dispatched manually.

CUDA wheels from GitHub Releases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GPU wheels are **not** published to PyPI. They are large and depend on the
system NVIDIA driver, so they are attached to the matching
`GitHub Release <https://github.com/exasim-project/NeoN/releases>`_ instead.
CUDA wheels carry a local version suffix such as ``+cuda128``.

The current CUDA wheel targets:

* CUDA 12.8
* CPython 3.12
* Linux x86-64 (``manylinux_2_34``)
* NVIDIA Ampere, compute capability ``sm_80``

Download the wheel from the release page and install it, or install it directly
by URL:

.. code-block:: bash

   pip install https://github.com/exasim-project/NeoN/releases/download/v0.1.0/neon_pde-0.1.0+cuda128-cp312-cp312-manylinux_2_34_x86_64.whl

The CUDA wheel requires a host NVIDIA driver that provides ``libcuda.so.1``.
That library is intentionally *not* bundled into the wheel, since it must match
the installed driver. A local CUDA toolkit is **not** required at runtime.

Because GitHub-hosted runners have no GPU, CUDA wheels are not runtime-tested in
CI and must be validated on a machine with a compatible NVIDIA driver and GPU.

Building from source
^^^^^^^^^^^^^^^^^^^^^

Building the bindings compiles the NeoN C++ library through
`scikit-build-core <https://scikit-build-core.readthedocs.io>`_, so it needs a
C++20 compiler and CMake ≥ 3.22 (see :doc:`installation`). From the repository
root:

.. code-block:: bash

   pip install .                # CPU build using the production preset

Build options are forwarded to CMake with ``--config-settings``. To build a
CUDA-enabled package matching the released wheels:

.. code-block:: bash

   pip install . \
     --config-settings=cmake.define.NeoN_BUILD_PYTHON_BINDINGS=ON \
     --config-settings=cmake.define.Kokkos_ENABLE_CUDA=ON \
     --config-settings=cmake.define.Kokkos_ARCH_AMPERE80=ON \
     --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=80 \
     --config-settings=cmake.define.CMAKE_CUDA_STANDARD=20

For local development, an editable install rebuilds on import:

.. code-block:: bash

   pip install -e .[dev]

Building the conda package
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The conda package is described by ``recipe/recipe.yaml`` and built with
`rattler-build <https://rattler.build>`_. ``ci/build_conda_packages.sh`` wraps it:
it derives the package version from ``pyproject.toml``, writes the per-build
variant configuration (python version, GPU flavour, C runtime floor) and invokes
``rattler-build``:

.. code-block:: bash

   ci/install_rattler_build.sh                       # optional, installs a pinned release
   ci/build_conda_packages.sh --python 3.12 --gpu cpu
   ci/build_conda_packages.sh --python 3.12 --gpu cuda --cuda-version 12.8

Packages land in ``output/<subdir>/``. The build clones Kokkos, Ginkgo and
Umpire through CPM, so it needs network access — unlike a conda-forge build,
which would have to vendor them.

Verifying an installation
-------------------------

The bindings expose the version and the compiled back-end feature flags. Use
them to confirm which executors a wheel supports:

.. code-block:: python

   import neon
   print(neon.__version__)      # e.g. 0.1.0 (or 0.1.0+cuda128 for a CUDA wheel)
   print(neon.__has_serial__)   # serial executor available
   print(neon.__has_cpu__)      # CPU (OpenMP) executor available
   print(neon.__has_gpu__)      # GPU (CUDA/HIP) executor available

The repository ships an equivalent smoke test at ``ci/check_installed_wheel.py``
that CI runs against every repaired CPU wheel.

Running the binding tests
-------------------------

The Python test suite lives in ``test/bindings`` and is driven by ``pytest``:

.. code-block:: bash

   pip install .[test]
   pytest test/bindings

Tests marked ``gpu`` are skipped automatically when no GPU executor is
available.

Versioning and release workflow
-------------------------------

The package version in ``pyproject.toml`` is the single source of truth. CMake
reads it during configuration and it becomes ``neon.__version__``.

Wheels are produced by the ``.github/workflows/python_wheels.yaml`` GitHub
Actions workflow using ``cibuildwheel``:

* **Stable releases** are triggered by tags named ``v*.*.*`` (e.g. ``v0.1.0``).
  CPU wheels are published to PyPI via Trusted Publishing, and CUDA wheels are
  attached to the GitHub Release for the tag.
* **Manual builds** (``workflow_dispatch``) produce a development version with a
  ``.dev`` suffix. Dispatch with ``build_wheels``, ``build_cpu`` and
  ``publish_repository=testpypi`` to rehearse a publish against TestPyPI without
  touching the production index.
* The workflow does **not** run on ordinary branch pushes or pull requests.

Conda packages are produced by ``.github/workflows/conda_packages.yaml`` on the
same triggers and with the same version scheme, so a wheel and a conda package
built from one commit report the same ``neon.__version__``. On a ``v*.*.*`` tag
the CPU and CUDA packages are built and uploaded to prefix.dev; a manual
dispatch can build any subset and only publishes when ``publish`` is set.
Publishing needs two repository settings: the ``PREFIX_DEV_CHANNEL`` variable
and the ``PREFIX_DEV_API_KEY`` secret (an API key with write access to that
channel).

For the full CI/CD description, including recovery paths for re-publishing
already-built artifacts, see :doc:`ci`.
