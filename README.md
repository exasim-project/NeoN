**[Quickstart](#quickstart)** |
**[Integration](#integration-with-other-cfd-frameworks)** |
**[Documentation](https://exasim-project.com/NeoN/latest)** |
**[Roadmap](https://github.com/orgs/exasim-project/projects/1/views/8)** |

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14608520.svg)](https://doi.org/10.5281/zenodo.14608520)
[![c++ standard](https://img.shields.io/badge/c%2B%2B-20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization) [![Gitter](https://img.shields.io/badge/Gitter-8A2BE2)](https://matrix.to/#/#NeoFOAM:gitter.im)
[![doxygen](https://img.shields.io/badge/Doxygen-8A2BE2)](https://exasim-project.com/NeoN/latest/doxygen/html/index.html)

![image](https://raw.githubusercontent.com/exasim-project/NeoN/refs/heads/color-logo/assets/NeonLogoCrop.png)

NeoN is an open-source, high-performance C++ CFD library for modern heterogeneous computing systems. It currently provides data structures, parallel algorithms, and numerical infrastructure for the finite volume method.

By combining finite-volume abstractions with modern C++ and performance-portable backends, NeoN enables developers to build maintainable, scalable, and high-performance fluid-flow solvers without writing architecture-specific code for each hardware platform.

Its modular architecture enables performance-portable execution and solver backends, currently using [Kokkos](https://github.com/kokkos/kokkos) and [Ginkgo](https://github.com/ginkgo-project/ginkgo) while remaining flexible to adopt alternative technologies as the framework evolves.

> [!IMPORTANT]
> The NeoN project needs you!
> If you're interested in contributing to NeoN please open a PR! If you have any questions on where to start please contact us here or on [gitter](https://matrix.to/#/#NeoFOAM:gitter.im).
> All participants are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Key Features

- Execution on
  - serial CPU
  - multithreaded CPU
  - MPI-based distributed systems
  - GPUs from NVIDIA, AMD, and Intel
- Portability across Linux, macOS, and Windows
- Performance-portable parallel and memory abstractions
- Unified GPU execution model for NVIDIA, AMD, and Intel GPUs

---

## Quickstart

To build the C++ library from source you need CMake ≥ 3.22, a C++20 compiler (_gcc >= 13_ or _clang >= 19_), [Ninja](https://ninja-build.org/) (only when using the CMake presets) and, optionally, MPI (disable with `-DNeoN_WITH_MPI=OFF`). Kokkos, Ginkgo and other C++ dependencies are managed through `CPM <https://github.com/cpm-cmake/CPM.cmake>` and fetched automatically when suitable system installations are unavailable.

    git clone https://github.com/exasim-project/NeoN.git && cd NeoN
    cmake --preset production -DNeoN_WITH_MPI=OFF
    cmake --build --preset production

Alternatively, try NeoN from Python via the prebuilt CPU package from PyPI (Python 3.9–3.13):

    pip install neon_pde

```python
import neon
neon.initialize()
executor = neon.CPUExecutor()
v = neon.ScalarVector(executor, 10, 1.0)
print(neon.__version__, v.size())
del v, executor
neon.finalize()
```

[![workflows/Build on linux](https://github.com/exasim-project/NeoN/actions/workflows/build_on_ubuntu.yaml/badge.svg?branch=main)](https://github.com/exasim-project/NeoN/actions/workflows/build_on_ubuntu.yaml?query=branch%3Amain)
[![workflows/Build on OSX](https://github.com/exasim-project/NeoN/actions/workflows/build_on_macos.yaml/badge.svg?branch=main)](https://github.com/exasim-project/NeoN/actions/workflows/build_on_macos.yaml?query=branch%3Amain)
[![workflows/Build on windows](https://github.com/exasim-project/NeoN/actions/workflows/build_on_windows.yaml/badge.svg?branch=main)](https://github.com/exasim-project/NeoN/actions/workflows/build_on_windows.yaml?query=branch%3Amain)

Further installation options (GPU builds, conda/pixi packages, CUDA wheels, Spack), the full list of requirements and how to run the tests are described in the [installation guide](https://exasim-project.com/NeoN/latest/installation.html) and the [Python bindings guide](https://exasim-project.com/NeoN/latest/python_bindings.html).

## Integration with other CFD Frameworks

Currently, NeoN is not a standalone CFD framework.
It is designed to be used with other CFD Frameworks.
Examples how to integrate NeoN into CFD frameworks and how to write applications is demonstrated in the [NeoFOAM](https://github.com/exasim-project/NeoFOAM) repository.

## Documentation

The online documentation can be found [here](https://exasim-project.com/NeoN/latest). Since NeoN is evolving quickly, the documentation might not always reflect the latest state.

## Contributing and Getting Help

- **Contributing:** see the [contributing guide](https://exasim-project.com/NeoN/latest/contributing.html) for the code style, the pull request workflow and how to build the documentation.
- **Reporting bugs and requesting features:** open an issue on the [GitHub issue tracker](https://github.com/exasim-project/NeoN/issues). The contributing guide lists [what to include in a bug report](https://exasim-project.com/NeoN/latest/contributing.html#reporting-issues-and-getting-help).
- **Questions and help:** ask on [GitHub Discussions](https://github.com/exasim-project/NeoN/discussions) or in the [Gitter chat](https://matrix.to/#/#NeoFOAM:gitter.im).
