.. SPDX-FileCopyrightText: 2026 NeoN authors
..
.. SPDX-License-Identifier: Unlicense

Split compilation: device and host-only translation units
=========================================================

NeoN is compiled in two halves.

**Device translation units** include Kokkos, may launch kernels (``NEON_LAMBDA``,
``parallelFor``) and are built with the device toolchain at ``CMAKE_CXX_STANDARD`` (C++20). This is
the level Kokkos 5 accepts for ``nvcc``, which currently gates CUDA builds to ``-std=c++20``.

**Host-only translation units** never see a Kokkos header. They hold configuration, database,
logging and most of the Python bindings, and are compiled at ``NeoN_HOST_CXX_STANDARD`` (20, 23 or
26, default 20). This lets that part of the code base use newer language and library features,
for example static reflection, without waiting for the GPU toolchains.

Both halves are linked into the same ``libNeoN`` (and ``_neon`` Python module), so there is no
additional shared-library boundary.

Configuring
-----------

CPU and HIP builds only need the option:

.. code-block:: bash

   cmake --preset production -DNeoN_HOST_CXX_STANDARD=26

CUDA builds need Kokkos in CMake-language mode, because ``nvcc_wrapper`` as ``CMAKE_CXX_COMPILER``
only accepts C++20. The host compiler then compiles the host-only half while ``nvcc`` uses a host
compiler it supports for the device half:

.. code-block:: bash

   cmake --preset production \
     -DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON \
     -DCMAKE_CXX_COMPILER=g++-16 \
     -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
     -DNeoN_HOST_CXX_STANDARD=26

Configuring ``NeoN_HOST_CXX_STANDARD`` other than 20 with ``nvcc_wrapper`` as the C++ compiler
stops with an error that points here.

When two host compilers are involved, the final link uses the newer one and its ``libstdc++``.
Keep the types passed between the halves to long-stable vocabulary types (``std::string``,
``std::vector``, ``std::unordered_map``, ``std::shared_ptr``, ``std::variant``, ``std::any``,
``std::span`` and plain structs). ``libstdc++`` gives no ABI guarantee for experimental standard
modes.

Rules for code
--------------

* Headers used by both halves must not include Kokkos. Annotate functions that have to be callable
  from kernels with ``NEON_INLINE_FUNCTION`` from ``NeoN/core/portability.hpp`` instead of
  ``KOKKOS_INLINE_FUNCTION``, and use ``NEON_ABORT`` instead of ``Kokkos::abort``.
* Shared headers stay valid C++20. Newer features belong in host-only ``.cpp`` files or headers
  only they include.
* Shared class definitions must not change members or layout depending on the half. Only function
  annotations (``NEON_HOST_DEVICE``) differ.
* Kokkos types for an executor come from ``NeoN/core/executor/kokkosExecutor.hpp``
  (``kokkosExecSpace<E>``, ``underlyingExec(e)``, ``createKokkosView(e, ptr, n)``), which only
  device translation units may include.
* Layout-relevant compile definitions (``NF_WITH_MPI_SUPPORT``, ``NeoN_DP_SCALAR``, ...) are set
  on ``NeoN_config_api``, which both halves see. Add new ones there, not on ``NeoN_public_api``.

Adding a host-only translation unit
-----------------------------------

Add the file to ``NeoN_HOST_SRCS`` in ``src/CMakeLists.txt`` (or to
``NeoN_PYTHON_HOST_BINDING_SRCS`` in ``src/bindings/CMakeLists.txt``). Host-only TUs are built by
``neon_add_host_object_library`` from ``cmake/HostOnly.cmake``, which

* links only ``NeoN_config_api``, so there is no Kokkos include path and no
  ``-DKOKKOS_DEPENDENCE`` (``kokkos_launch_compiler`` leaves these TUs with the host compiler),
* defines ``NEON_HOST_ONLY_TU=1``,
* force-includes ``src/hostOnlyGuard.hpp``, which poisons the identifier ``Kokkos`` on GCC and
  Clang.

The guard makes a mistake a hard error even when Kokkos is found on a default include path, as in
conda or system installs:

* ``attempt to use poisoned "Kokkos"``: a header reachable from the TU includes Kokkos. Make it
  Kokkos-free or move the TU to ``NeoN_SRCS``.
* ``parallelAlgorithms.hpp launches Kokkos kernels ...``: the TU tries to launch a kernel. Move the
  kernel into a device TU and call it through a non-template function.

MSVC has no equivalent of ``#pragma GCC poison``; there the missing include path is the only guard.
