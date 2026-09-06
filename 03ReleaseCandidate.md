<!--
SPDX-FileCopyrightText: 2026 NeoN authors

SPDX-License-Identifier: MIT
-->

# NeoN v0.3.0rc1 — Release Candidate

NeoN is a portable C++20 library of CFD primitives — Kokkos-backed containers, cell-centred
finite-volume operators, a PDE DSL and Ginkgo linear solvers. It is the next-generation
high-performance backend for CFD frameworks like NeoFOAM and OpenFOAM, not a standalone solver.

When v0.2 was released we announced that v0.3 would focus on performance optimisations, multi-GPU
and Python support. This release candidate delivers a first implementation of the latter two, and
we would like them exercised on real cases before v0.3.0 is tagged.

- Tag: [`v0.3.0rc1`](https://github.com/exasim-project/NeoN/releases/tag/v0.3.0rc1)
- Python package: [`neon_pde` on PyPI](https://pypi.org/project/neon_pde/)
- Full changelog: [`CHANGELOG.md`](https://github.com/exasim-project/NeoN/blob/develop/CHANGELOG.md)

## Highlights

### Distributed support (experimental)

NeoN now runs across multiple MPI ranks, and therefore across multiple GPUs.

- Processor boundaries carry exact processor-face geometry, with non-orthogonal corrected and
  limited surface-normal gradient corrections across rank boundaries
  ([#528](https://github.com/exasim-project/NeoN/pull/528)).
- The linear system is assembled distributed through Ginkgo. The off-diagonal matrix is built with
  local row indices and non-local columns are widened on the executor, removing host round-trips
  from the assembly path.
- Boundary conditions gained a one-time `set()` / per-iteration `update()` interface
  ([#528](https://github.com/exasim-project/NeoN/pull/528)).
- Mesh partitioning (`include/NeoN/distributed/partitioning.hpp`) and a communication pattern
  abstraction (`communicationPattern.hpp`), on top of the existing `Environment` and half/full
  duplex communication buffers.
- Strong and weak scaling benchmarks under `benchmarks/distributed/`, and a distributed test suite
  under `test/distributed/` driven by `mpirun`.

Correctness fixes in this cycle covered multi-patch (scotch) halo exchange, the `ddtFluxCorr`
processor-face correction, processor boundary conditions on coupled patches, and row-sorted
non-local COO so the distributed apply is correct on CUDA
([#528](https://github.com/exasim-project/NeoN/pull/528)).

### Python bindings

NeoN is now importable from Python as the `neon` package
([#382](https://github.com/exasim-project/NeoN/pull/382)).

- Exposes executors, containers, fields, meshes, surface interpolation, the DSL, the linear algebra
  layer and the document database.
- Implemented with [nanobind](https://github.com/wjakob/nanobind), so the bindings stay a thin
  layer over the C++ API.
- 25 pre-built wheels: CPython 3.9–3.13 on Linux (x86-64 and ARM64), macOS (Apple Silicon and
  Intel) and Windows (AMD64).
- No compiler, Kokkos installation or CUDA toolkit is needed to get started.

```bash
pip install --pre neon_pde
```

```python
import neon

neon.initialize()
executor = neon.CPUExecutor()
vector = neon.ScalarVector(executor, 10, 1.0)
print(neon.__version__, neon.__has_serial__, neon.__has_cpu__, neon.__has_gpu__)
del vector, executor          # Kokkos aborts if allocations outlive finalize
neon.finalize()
```

### Also in this cycle

- `slip` and `inletOutlet` volume boundary conditions
  ([#565](https://github.com/exasim-project/NeoN/pull/565)).
- Corrected and limited-corrected face-normal gradient schemes
  ([#514](https://github.com/exasim-project/NeoN/pull/514)) and a `linearUpwind` scheme
  ([#548](https://github.com/exasim-project/NeoN/pull/548)).
- Experimental cell-based assembly strategies and COO/CSR matrix support
  ([#471](https://github.com/exasim-project/NeoN/pull/471),
  [#486](https://github.com/exasim-project/NeoN/pull/486)).
- Tensor and symmTensor primitives, `Su`-type source terms, and passing fields to boundary
  conditions ([#428](https://github.com/exasim-project/NeoN/pull/428)).
- DSL expression optimiser infrastructure for operator fusing
  ([#452](https://github.com/exasim-project/NeoN/pull/452)).
- Experimental Umpire memory management ([#455](https://github.com/exasim-project/NeoN/pull/455)),
  a uniform mesh generator ([#475](https://github.com/exasim-project/NeoN/pull/475)) and an L1-norm
  stopping criterion ([#538](https://github.com/exasim-project/NeoN/pull/538)).
- Kokkos bumped to 5.0.2, Ginkgo to 2.0.0.

## What we would like tested

This is a release candidate, and it is most useful to us in other people's hands. Concretely:

1. **Distributed results against serial.** Run the same case on one rank and on many, and report any
   divergence. Non-orthogonal meshes and multi-patch decompositions are the most valuable, since
   that is where the processor-face corrections are newest.
2. **Rank counts and hardware we do not have.** Our CI runners are CPU-only; the distributed CUDA
   path is not runtime-tested in CI at all. Multi-GPU runs on real hardware are the single most
   useful thing anyone can contribute right now.
3. **The Python API.** Does it feel natural from Python, or does it read like C++ in disguise? Which
   objects are missing? Where does the lifetime model surprise you?
4. **Wheel installation.** Report any platform or Python version where `pip install --pre neon_pde`
   fails or the extension does not import.

Issues, discussions and pull requests are all welcome at
<https://github.com/exasim-project/NeoN>.

## Known limitations

- **Distributed support is experimental.** The API is expected to change before it is considered
  stable.
- **The Python wheels are built without MPI.** `NeoN_WITH_MPI=OFF`, along with PETSc, ADIOS2 and
  SUNDIALS, so the distributed features are C++-only for now. Distributed and Python are two
  separate features in this release, not one combined one.
- **No GPU wheels are published for this release candidate.** CUDA wheels are built on tags but are
  attached to a GitHub Release, and none exists for `v0.3.0rc1` yet. GPU users must build from
  source; see `doc/python_bindings.rst`.
- **CUDA wheels, when published, are narrow by design:** CUDA 12.8, CPython 3.12, Linux x86-64,
  NVIDIA Ampere (`sm_80`). They are not runtime-tested in CI because GitHub-hosted runners have no
  GPU.
- **No conda/pixi packages yet.** The rattler-build recipe and prefix.dev publishing workflow exist
  on a branch but are not merged, and the channel is not live. Targeted at v0.3.0 final.

## Contributors

This pre-release would not have been possible without: Gregor Olenik, Hendrik Hetmann,
Chih-Ta Wang, Dheeraj Raghunathan, Andrei Maftei and Henning Scheufler.

NeoN is a community-driven project and we welcome feedback and contributions.

---

## Appendix: announcement copy

Ready to paste. Plain text, matching the style of the previous NeoFOAM release posts: no emoji, no
bold, hashtags inline in the prose, hyphen bullets, URLs inline.

### LinkedIn (2,216 characters; limit 3,000)

```text
The NeoN team is proud to announce the v0.3.0rc1 pre-release https://github.com/exasim-project/NeoN/releases/tag/v0.3.0rc1 of NeoN, the next-generation high-performance backend for #CFD frameworks like #NeoFOAM and #OpenFOAM. When we released v0.2 we announced that v0.3 would focus on multi-GPU and Python support. This release candidate delivers a first implementation of both, and before we tag the final release we would like your help testing them.

Distributed support (experimental):
- Runs across multiple MPI ranks, and therefore across multiple GPUs;
- Processor boundaries with exact processor-face geometry, including non-orthogonal corrected and limited surface-normal gradient corrections across rank boundaries;
- Distributed linear system assembly through #Ginkgo, with the off-diagonal matrix built on the executor rather than on the host;
- Mesh partitioning, together with strong and weak scaling benchmarks.

Python bindings:
- Executors, containers, fields, meshes, the DSL, and the linear algebra layer are now available from Python;
- Implemented with nanobind, keeping the bindings a thin layer over the C++ API;
- Installable with "pip install --pre neon_pde" https://pypi.org/project/neon_pde/, with pre-built wheels for CPython 3.9 to 3.13 on Linux (x86-64 and ARM64), macOS (Apple Silicon and Intel), and Windows;
- No compiler, #Kokkos installation, or CUDA toolkit needed to get started.

This is a release candidate rather than a final release, and it is most useful to us in other people's hands. Please run it on your meshes, at your rank counts, and on your hardware, and tell us where the distributed results disagree with a serial run, where the Python API feels awkward, or where a wheel refuses to install. Issues, discussions, and pull requests are all welcome at https://github.com/exasim-project/NeoN.

NeoN is a community-driven project and we welcome feedback and contributions. A big thanks to all contributors who made this pre-release possible: Gregor Olenik, Hendrik Hetmann, Chih-Ta Wang, Dheeraj Raghunathan, Andrei Maftei, and Henning Scheufler.

A full list of changes can be found in the CHANGELOG https://github.com/exasim-project/NeoN/blob/develop/CHANGELOG.md.
```

### Mastodon (494 characters as Mastodon counts links; limit 500)

```text
NeoN v0.3.0rc1 — release candidate 🚀

Portable C++20 CFD primitives (Kokkos + Ginkgo), the numerics layer behind NeoFOAM.

🌐 Distributed/MPI (experimental): processor boundaries with exact proc-face geometry, non-orthogonal corrected snGrad across ranks, distributed assembly via Ginkgo, partitioning + scaling benchmarks.

🐍 Python bindings (nanobind): fields, meshes, DSL, solvers.
pip install --pre neon_pde

Please stress-test it and tell us what breaks.

https://github.com/exasim-project/NeoN
#CFD #HPC
```

### X / Bluesky (262 characters raw; 255 on X, where links count as 23)

```text
NeoN v0.3.0rc1 — release candidate for our portable C++20 CFD library.

New: experimental distributed/MPI support, and Python bindings.

pip install --pre neon_pde

It's an RC for a reason. Please test it and tell us what breaks.

github.com/exasim-project/NeoN
```

### Suggested visual

Both previous NeoFOAM release posts carried an image (a validation plot and a vortex-shedding
animation). A strong-scaling plot generated from `benchmarks/distributed/strongScaling.cpp` would
fit that pattern and would directly illustrate the headline feature.
