**[Requirements](#requirements)** |
**[Compilation](#Compilation)** |
**[Integration](#integration-with-other-cfd-frameworks)** |
**[Documentation](https://exasim-project.com/NeoN/latest)** |
**[Roadmap](https://github.com/orgs/exasim-project/projects/1/views/8)** |

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14608521.svg)](https://doi.org/10.5281/zenodo.14608521)
[![c++ standard](https://img.shields.io/badge/c%2B%2B-20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization) [![Gitter](https://img.shields.io/badge/Gitter-8A2BE2)](https://matrix.to/#/#NeoFOAM:gitter.im)
[![doxygen](https://img.shields.io/badge/Doxygen-8A2BE2)](https://exasim-project.com/NeoN/latest/doxygen/html/index.html)

![image](https://raw.githubusercontent.com/exasim-project/NeoN/refs/heads/main/assets/NeonLogoCrop.png)

# NeoN

NeoN is an open-source, high-performance C++ CFD library for modern heterogeneous computing systems. It currently provides data structures, parallel algorithms, and numerical infrastructure for the finite volume method.

By combining finite-volume abstractions with modern C++ and performance-portable backends, NeoN enables developers to build maintainable, scalable, and high-performance fluid-flow solvers without writing architecture-specific code for each hardware platform.

Its modular architecture enables performance-portable execution and solver backends, currently using [Kokkos](https://github.com/kokkos/kokkos) and [Ginkgo](https://github.com/ginkgo-project/ginkgo) while remaining flexible to adopt alternative technologies as the framework evolves.

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

> [!IMPORTANT]
> The NeoN project needs you!
> If you're interested in contributing to NeoN please open a PR! If you have any questions on where to start please contact us here or on [gitter](https://matrix.to/#/#NeoN:gitter.im).

## Requirements

NeoN has the following requirements

*  _cmake > 3.22_
*  _gcc >= 13_ or  _clang >= 19_
*  _Kokkos 5.0.2_

For GPU support
* NVIDIA: CUDA _12+_
* AMD: ROCm _6.4.1_
* Intel: oneAPI Base Toolkit _2024.2_

For development it is required to use [pre-commit](https://pre-commit.com/).

### C++ dependencies

C++ dependencies like Kokkos are handled via [CPM](https://github.com/cpm-cmake/CPM.cmake) and are cloned at the configuration step.
However, the cmake build process will prefer system wide installed C++ dependencies like Kokkos, cxxopts, etc.
If you prefer to clone, configure and build dependencies your self consider setting `-DCPM_USE_LOCAL_PACKAGES = OFF`, see [CPM](https://github.com/cpm-cmake/CPM.cmake) for more details.

## Compilation

[![workflows/Build on linux](https://github.com/exasim-project/NeoN/actions/workflows/build_on_ubuntu.yaml/badge.svg?branch=main)](https://github.com/exasim-project/NeoN/actions/workflows/build_on_ubuntu.yaml?query=branch%3Amain)
[![workflows/Build on OSX](https://github.com/exasim-project/NeoN/actions/workflows/build_on_macos.yaml/badge.svg?branch=main)](https://github.com/exasim-project/NeoN/actions/workflows/build_on_macos.yaml?query=branch%3Amain)
[![workflows/Build on windows](https://github.com/exasim-project/NeoN/actions/workflows/build_on_windows.yaml/badge.svg?branch=main)](https://github.com/exasim-project/NeoN/actions/workflows/build_on_windows.yaml?query=branch%3Amain)

NeoN uses cmake to build, thus the standard cmake procedure should work.
From a build directory you can execute

    cmake <DesiredBuildFlags> ..
    cmake --build .
    cmake --install .

Additionally, we provide several Cmake presets to set commmonly required flags if you compile NeoN in combination with Kokkos.

    cmake --list-presets # To list existing presets
    cmake --preset production # To configure for production use
    cmake --build --preset production # To compile for production use


### Executing Tests

We provide a set of unit tests which can be executed via ctest or

    cmake --build . --target test

### Installing the Python bindings

NeoN ships Python bindings as the `neon_pde` distribution (imported as `neon`).
The package requires Python 3.9–3.13 and NumPy.

**CPU (from PyPI).** Pre-built CPU wheels are published to PyPI for Linux
(x86-64/ARM64), Windows (AMD64) and macOS (Apple Silicon/Intel):

    pip install neon_pde

**Conda / pixi (from prefix.dev).** The same releases are published as conda
packages named `neon-pde` to a [prefix.dev](https://prefix.dev) channel, which
also installs the NeoN C++ runtime, headers and CMake package files into the
environment:

    pixi add neon-pde -c https://prefix.dev/exasim-project -c conda-forge

    # CUDA 12.8, CPython 3.12, NVIDIA Ampere (sm_80), linux-64
    pixi add "neon-pde=*=cuda_py312*" -c https://prefix.dev/exasim-project -c conda-forge

Conda packages cover CPython 3.10–3.13 on `linux-64`, `linux-aarch64`, `osx-64`
and `osx-arm64`; Windows and Python 3.9 are wheel-only.

**CUDA (from GitHub Releases).** GPU wheels are *not* published to PyPI because
they are large and depend on the NVIDIA driver. They are attached to the
corresponding [GitHub Release](https://github.com/exasim-project/NeoN/releases)
and carry a local version suffix such as `+cuda128`. Download the wheel matching
your Python version, or install it directly by URL:

    # CUDA 12.8, CPython 3.12, Linux x86-64, NVIDIA Ampere (sm_80)
    pip install https://github.com/exasim-project/NeoN/releases/download/v0.1.0/neon_pde-0.1.0+cuda128-cp312-cp312-manylinux_2_34_x86_64.whl

The CUDA wheel needs a host NVIDIA driver providing `libcuda.so.1` (it is
intentionally not bundled). It does not require a local CUDA toolkit at runtime.

**From source.** Building the bindings uses `scikit-build-core` and compiles the
C++ library, so a C++20 compiler and CMake ≥ 3.22 are required:

    pip install .                       # CPU build (production preset)

    # CUDA build (matching the released wheels)
    pip install . --config-settings=cmake.define.Kokkos_ENABLE_CUDA=ON \
                  --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=80 \
                  --config-settings=cmake.define.CMAKE_CUDA_STANDARD=20

Verify an installation with:

```python
import neon
print(neon.__version__, neon.__has_serial__, neon.__has_cpu__, neon.__has_gpu__)
```

See [the documentation](https://exasim-project.com/NeoN/latest) for the full
install-and-release workflow.

### Python Wheels

The Python distribution name is `neon_pde`, which produces wheel filenames
starting with `neon_pde`. The import package remains `neon`.

The package version in `pyproject.toml` is the source of truth. CMake reads that
version during configuration, and the generated `neon.__version__` uses the same
value.

GitHub Actions uses cibuildwheel to build wheels for release tags and explicitly
requested manual builds. Stable releases use tags named like `v0.1.0`. Manual
non-tag builds use development versions like `0.1.1.dev202605270217123`. The
wheel workflow does not currently run for ordinary branch pushes, pull requests,
or on a nightly schedule.

CPU wheels are built for:

* Linux x86-64
* Linux ARM64
* Windows AMD64
* macOS Apple Silicon
* macOS Intel

The CPU matrix covers CPython 3.9 through 3.13. CPU wheels use the plain package
version and are published to PyPI only for stable `v*.*.*` tags.

CUDA wheels are currently limited to CUDA 12.8 on Linux x86-64 with CPython
3.12. They use a local version suffix such as `0.1.0+cuda128`, are uploaded as
workflow artifacts, and are attached to the GitHub Release for stable tags.

### Conda packages

Conda packages are built from `recipe/recipe.yaml` with
[rattler-build](https://rattler.build) by
`.github/workflows/conda_packages.yaml`, on the same triggers and with the same
version scheme as the wheels, and are published to prefix.dev. Build one locally
with:

    ci/install_rattler_build.sh                       # optional, pinned release
    ci/build_conda_packages.sh --python 3.12 --gpu cpu

The GPU flavour is part of the build string (`cpu_py312_*`, `cuda_py312_*`,
`rocm_py312_*`), so all flavours coexist in one channel. The ROCm flavour is
experimental and manual-dispatch only: conda-forge has no rocBLAS/rocSPARSE/
rocThrust/rocPRIM, so it ships Kokkos HIP without the Ginkgo solver backend.
GitHub-hosted runners do not provide a GPU, so the CUDA wheel build does not run
runtime tests against the resulting wheel.

## Integration with other CFD Frameworks

Currently, NeoN is not a standalone CFD framework.
It is designed to be used with other CFD Frameworks.
Examples how to integrate NeoN into CFD frameworks and how to write applications is demonstrated in the [NeoFOAM](https://github.com/exasim-project/NeoFOAM) repository.

## Documentation

An online documentation can be found [here](https://exasim-project.com/NeoN/latest), be cautious since this repository is currently evolving the documentation might not always reflect the latest stage.

For building the documentation further dependencies like doxygen and sphinx are requirement.
The list of requirements can be found [here](https://github.com/exasim-project/NeoN/actions/workflows/build_doc.yaml)
