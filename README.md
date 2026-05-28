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

### Pixi Workspace

For local development, NeoN can be driven via `pixi.toml`. Pixi provides the
developer tools, such as Python, CMake, Ninja, a C++ compiler, nanobind, pytest,
and optional CUDA compiler packages, in an isolated environment.

Typical commands are:

    pixi install
    pixi shell
    pixi run build
    pixi run test
    pixi run -e cuda build-cuda
    pixi run -e cuda test-cuda

Pixi is only the local development leg. It is not used to publish Python wheels
or upload releases.

### Python Wheels

The Python distribution name in this test repository is
`andrei-maftei-testneon`, which produces wheel filenames starting with
`andrei_maftei_testneon`. The import package remains `neon`.

The package version in `pyproject.toml` is the source of truth. CMake reads that
version during configuration, and the generated `neon.__version__` uses the same
value.

GitHub Actions uses cibuildwheel to build wheels after pushes, nightly schedules,
manual dispatches, and release tags. Stable releases use tags named like
`v0.1.0`. Nightly builds use development versions like
`0.1.1.dev202605270217123`.

CPU wheels use the plain package version and are published to PyPI only for
stable `v*.*.*` tags. CUDA wheels use local version suffixes such as
`0.1.0+cuda128` and `0.1.0+cuda130`; these are uploaded to the GitHub Release
for stable tags and are also available as workflow artifacts.

## Integration with other CFD Frameworks

Currently, NeoN is not a standalone CFD framework.
It is designed to be used with other CFD Frameworks.
Examples how to integrate NeoN into CFD frameworks and how to write applications is demonstrated in the [NeoFOAM](https://github.com/exasim-project/NeoFOAM) repository.

## Documentation

An online documentation can be found [here](https://exasim-project.com/NeoN/latest), be cautious since this repository is currently evolving the documentation might not always reflect the latest stage.

For building the documentation further dependencies like doxygen and sphinx are requirement.
The list of requirements can be found [here](https://github.com/exasim-project/NeoN/actions/workflows/build_doc.yaml)
