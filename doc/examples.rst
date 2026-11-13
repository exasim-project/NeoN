Examples
========

The `examples <https://github.com/exasim-project/NeoN/tree/develop/examples>`_ directory contains small, self-contained programs that show how NeoN is used.
Each example is a single source file, solves a problem with a known exact solution and checks its result.

Build them by adding ``-DNeoN_BUILD_EXAMPLES=ON`` when configuring NeoN (see :doc:`installation`):

.. code-block:: bash

    cmake --preset production -DNeoN_WITH_MPI=OFF -DNeoN_BUILD_EXAMPLES=ON
    cmake --build --preset production
    ./build/production/bin/examples/neon_example_steadyDiffusion

All examples are also registered with CTest and can be run with ``ctest -R example_``.
The examples require Ginkgo (``NeoN_WITH_GINKGO=ON``, the default).

Steady diffusion
----------------

Solves the steady heat equation with a uniform source,

.. math::

    -\frac{d}{dx}\left(\Gamma \frac{dT}{dx}\right) = S, \qquad T(0) = 0, \quad T(1) = 1,

and shows the basic workflow of every NeoN application: create a mesh, set up boundary conditions and fields, choose the schemes and the linear solver via dictionaries, write the equation with the DSL and solve it.
Repeating the solve on refined meshes shows second-order convergence:

.. code-block:: text

      nCells    L2 error    order
      10        0.0125
      20        0.003125    2
      40        0.00078125  2
      80        0.000195312 2

.. literalinclude:: ../examples/steadyDiffusion.cpp
   :language: cpp
   :lines: 12-

.. note::

   Once the discretisation has cached the geometric quantities it needs, the mesh releases its points, face centres and cell centres to save (device) memory.
   Copy them beforehand if you need them later, as the examples do for evaluating the exact solution.

Further examples
----------------

Complete solvers built on NeoN, e.g. for incompressible flows, can be found in `NeoFOAM <https://github.com/exasim-project/NeoFOAM>`_.
