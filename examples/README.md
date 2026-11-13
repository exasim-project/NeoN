# NeoN examples

Small, self-contained programs that show what NeoN can do. Each example is a single `.cpp` file,
solves a problem with a known exact solution and checks its result, so it also serves as a test.

| Example | What it shows |
| --- | --- |
| [steadyDiffusion.cpp](steadyDiffusion.cpp) | Mesh, fields and boundary conditions; writing an equation with the DSL; solving it with Ginkgo; checking second-order convergence |

## Building and running

Enable the examples when configuring NeoN:

    cmake --preset production -DNeoN_WITH_MPI=OFF -DNeoN_BUILD_EXAMPLES=ON
    cmake --build --preset production

The executables are written to `build/<preset>/bin/examples`, for example:

    ./build/production/bin/examples/neon_example_steadyDiffusion

which prints

    Running on CPUExecutor

      nCells    L2 error    order
      10        0.0125
      20        0.003125    2
      40        0.00078125  2
      80        0.000195312 2

The examples require Ginkgo (`NeoN_WITH_GINKGO=ON`, the default).
All examples are registered with CTest, so `ctest -R example_` runs them and checks their results.

To run on a GPU, configure NeoN with the matching Kokkos backend (e.g. `-DKokkos_ENABLE_CUDA=ON`);
the examples then pick the GPU executor automatically, without changes to the code.

More examples of complete solvers built on NeoN, e.g. incompressible flow solvers, can be found in
[NeoFOAM](https://github.com/exasim-project/NeoFOAM).
