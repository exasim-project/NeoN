# CLAUDE.md — NeoN

NeoN is a C++20 framework for building CFD software, providing executor-based parallelism (Serial/CPU/GPU via Kokkos), finite volume discretisation, and a DSL for assembling PDE systems. Python bindings via nanobind.

## Build Commands

```bash
# Configure + build (presets: develop, production, profiling)
cmake --preset develop
cmake --build --preset develop

# Run all tests
ctest --preset develop

# Run a single test
ctest --preset develop -R <test_name>

# Exclude benchmarks from test run
ctest --preset develop -E bench
```

**Presets**: `develop` (Debug, tests+warnings+AMReX on), `production` (Release), `profiling` (RelWithDebInfo, benchmarks on).

## Python Bindings (blockamr) Build Commands

```bash
# Full rebuild (C++ + Python, slow — only needed for C++ changes or first install)
uv pip install -e ".[all]" --no-build-isolation -v > log.build 2>&1

# Sync Python files only (fast — use after editing src/blockamr/*.py)
cp -r src/blockamr/*.py .venv/lib/python3.12/site-packages/blockamr/
cp -r src/blockamr/**/*.py .venv/lib/python3.12/site-packages/blockamr/

# Run examples/tests
uv run --no-sync python example/blockamr/amr_mesh_demo.py

# Run blockamr tests
uv run --no-sync python -m pytest test/blockamr/
```

**Important**: `uv pip install -e` triggers scikit-build-core which regenerates cmake cache with a new temp path, causing a full AMReX recompilation (~3 min). For Python-only changes, copy files directly instead.

## Key CMake Options

| Option | Default | Description |
|---|---|---|
| `NeoN_WITH_KOKKOS` | ON | Platform portability (always needed) |
| `NeoN_WITH_GINKGO` | ON | Linear algebra backend |
| `NeoN_WITH_PETSC` | OFF | Alternative LA backend |
| `NeoN_WITH_SUNDIALS` | OFF | ODE solver support |
| `NeoN_WITH_AMREX` | OFF | AMReX mesh support |
| `NeoN_WITH_UMPIRE` | ON | Memory manager (experimental) |
| `NeoN_WITH_SPDLOG` | OFF | Logging |
| `NeoN_BUILD_TESTS` | OFF | Unit tests (Catch2) |
| `NeoN_BUILD_BENCHMARKS` | OFF | Benchmarks |
| `NeoN_BUILD_PYTHON_BINDINGS` | OFF | nanobind Python bindings |
| `NeoN_ENABLE_MPI_SUPPORT` | ON | MPI support |

Dependencies are managed via CPM (cmake/CxxThirdParty.cmake, versions in cmake/Versions.cmake).

## Architecture

### Directory Layout

```
include/NeoN/
  core/           # executor, database, memory, MPI, primitives, vector
  dsl/            # expression DSL: explicit (exp::) and implicit (imp::) operators
  fields/         # field base class, boundary data
  finiteVolume/   # cell-centred FV: operators, boundary conditions, interpolation, stencils
    cellCentred/
      operators/  # ddt, div, laplacian, grad, sourceterm
      fields/     # VolumeField, SurfaceField
      boundary/   # surface and volume boundary conditions
  linearAlgebra/  # sparse matrix, linear system, Ginkgo/PETSc solvers
  mesh/           # unstructured mesh
  timeIntegration/
src/              # implementation files mirror include/ structure
test/             # Catch2 tests mirroring module structure
src/bindings/     # nanobind Python bindings
```

### Key Concepts

- **Executor**: `std::variant<SerialExecutor, CPUExecutor, GPUExecutor>` — all computation dispatches through an executor. Kokkos underneath.
- **Fields**: `VolumeField<T>` (cell-centred), `SurfaceField<T>` (face). Value types: `scalar`, `Vec3`, `Tensor`.
- **DSL**: `NeoN::dsl::exp::` (explicit) and `NeoN::dsl::imp::` (implicit) namespaces. Operators: `ddt()`, `div()`, `laplacian()`, `grad()`, `source()`.
- **Linear Algebra**: `LinearSystem` (matrix A, vectors x, b). Backends: Ginkgo (default), PETSc.
- **Database**: Document-based key-value store (`Database` → `Collection` → `Document`). `FieldCollection` and `OldTimeCollection` for field management.
- **Mesh**: `UnstructuredMesh` only.

## Naming Conventions (enforced by clang-tidy)

- **Classes/Structs/Enums**: `CamelCase` — `VolumeField`, `LinearSystem`
- **Functions/Variables/Parameters**: `camelBack` — `createDefaultExecutor()`, `faceFlux`
- **Global variables**: `UPPER_CASE`
- **Template parameters**: `CamelCase` — `ValueType`
- **Namespaces**: `NeoN::dsl::exp::`, `NeoN::finiteVolume::cellCentred::`

## Licensing (REUSE-compliant)

All files require SPDX headers:

- **C++/Python/docs**: MIT — `SPDX-License-Identifier: MIT`
- **CMake/config files**: Unlicense — `SPDX-License-Identifier: Unlicense`
- Copyright line: `SPDX-FileCopyrightText: 2023 - 2026 NeoN authors`

## Formatting & Checks

- **clang-format**: LLVM-based style (see `.clang-format`)
- **cmake-format**: for CMake files (see `.cmake-format.py`)
- **pre-commit**: clang-format, cmake-format, REUSE check, typos
- Run pre-commit: `pre-commit run --all-files`

## Testing

Tests use **Catch2**. The `NeoN_unit_test()` CMake function registers tests with CTest. MPI tests specify `MPI_SIZE`. Test sources mirror the module structure under `test/`.

## Python Bindings

Package name: `neon_pde`. Built with scikit-build-core + nanobind. Binding sources in `src/bindings/`. Install with:

```bash
pip install -e . --no-build-isolation
```

Requires `NeoN_BUILD_PYTHON_BINDINGS=ON`.
