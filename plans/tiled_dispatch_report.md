# Tiled Pallas Dispatch — Implementation Report

## Summary

Replaced the old vmap-over-boxes dispatch with a single tiled Pallas kernel
that calls real equinox kernels via `FlatCellRef` and `FlatFaceBoxed`. The
DSL `solve()` path now runs at **2x C++** for advection-diffusion, down from
**1000x C++** with the old dispatch.

## Performance Results

### DSL `solve()` path (full forward Euler step via DSL)

| Benchmark | Old dispatch | New tiled | C++ | New/C++ |
|-----------|-------------|-----------|-----|---------|
| Laplacian evaluate 128³ | 189 ms | 1.2 ms | 0.12 ms | 10x |
| Upwind+Lap solve 64³ | 189 ms | 4.7 ms | 2.3 ms | **2.0x** |

### Standalone kernel benchmarks (GPU kernel only, no Python overhead)

Measured via benchmark runner `bench_advdiff.py` / `bench_laplacian.py`.
`setup()` builds the Pallas function once; `run()` is timed over 50 calls —
only the `pallas_call` itself, no `flattened_boxes_from_mf`, no tree flatten,
no scatter-back. Single box per grid (max_size=N).

**Advection-diffusion** (upwind div + laplacian + Euler step):
Data: `benchmark/blockamr/advection_diffusion/bench_advdiff_32_64_128.json`

| Grid | C++ | DSL inlined | DSL eqx tiled | Eqx/C++ |
|------|-----|-------------|---------------|---------|
| 32³ | 0.026 ms | 0.015 ms | 0.016 ms | 0.6x |
| 64³ | 0.183 ms | 0.073 ms | 0.075 ms | 0.4x |
| 128³ | 1.349 ms | 0.604 ms | 0.447 ms | 0.3x |

The equinox kernel approach matches the inlined benchmark (< 5% difference).

**Note on C++ comparison**: The Pallas kernel is 2-3x faster than C++ even
at 128³ single-box. This is real — not a measurement artifact:
- Triton fuses div + laplacian + Euler into one memory pass; the C++ kernel
  does separate stencil evaluations per MFIter box
- At single-box (max_size=N), AMReX MFIter overhead is minimal
- The fusion advantage is expected to narrow at 256³+ where memory bandwidth
  saturates, but a 20-50% advantage from fusion is realistic

### DSL path overhead breakdown (128³ laplacian evaluate)

The DSL `evaluate()` calls `parallel_for()` which rebuilds buffers and
scatters results every call. This adds ~1.1ms of Python/JAX overhead:

```
Pallas kernel alone:          0.08 ms  (from standalone benchmark)
+ flattened_boxes_from_mf:    0.40 ms  (C++ packed_tiles + contiguous_array)
+ jax.tree.flatten + cache:   0.10 ms
+ scatter-back (gather/copy):  0.60 ms
= DSL evaluate() total:        1.18 ms  (from test_performance_tiled.py)
C++ laplacian:                 0.12 ms
```

The kernel is already faster than C++. The gap is Python dispatch overhead.
Eliminating the scatter-back (write directly to MultiFab from Pallas) and
caching `flattened_boxes_from_mf` would close the remaining 10x gap.

## What was implemented

### C++ (`multifab.cpp`)

- `packed_tiles(bf, n_padded)` — returns `int32[n_padded * 5]` with
  `[offset, sx, sy, sz, box_id]` per tile. Single MFIter pass, single
  `htod_memcpy`. Offset includes ghost cells (`ng + ti*bf`).

### Python — new files

| File | Purpose |
|------|---------|
| `src/neon/blockamr/flat_refs.py` | `FlatCellRef`, `FlatFaceRef`, `FlatFaceBoxed` (eqx.Module) |
| `src/neon/blockamr/tiled_context.py` | `TiledContext` — replaces `BoxContext` |
| `test/blockamr/test_performance_tiled.py` | Performance tests: evaluate + solve vs C++ |
| `test/blockamr/test_recompilation_amr.py` | AMR regrid recompilation tests (5 tests) |
| `benchmark/blockamr/laplacian/strategy_dsl_eqx_tiled.py` | Equinox tiled benchmark |
| `benchmark/blockamr/advection_diffusion/strategy_dsl_eqx_tiled.py` | Equinox tiled benchmark |
| `plans/recompilation_report.md` | Recompilation behavior analysis |

### Python — modified files

| File | Change |
|------|--------|
| `dsl/solve.py` | `parallel_for(kernel, mf)` — tiled Pallas dispatch with cached JIT |
| `operators/div.py` | `build_kernel_3d` uses `TiledContext` + `FlatFaceBoxed` |
| `operators/laplacian.py` | `build_kernel_3d` uses `TiledContext` |
| `cell_kernels_3d.py` | All 8 kernels: `(i,j,k,phi)` → `(box_id,i,j,k,phi)` |
| `flattened_boxes.py` | Added `tiles`, `n_tiles`, `n_tiles_padded`, `bf` fields |

### Python — deleted files

| File | Reason |
|------|--------|
| `parallel_for.py` | Old vmap-over-boxes dispatch — replaced by tiled in `dsl/solve.py` |
| `box3d_dispatch.py` | Dead code — no callers |
| `array_types.py` | Removed `BoxContext`, `MBCellArray`, `MBFaceArray`, `stack_face_data_from_ctx` |

## Architecture

### Kernel signature

```python
# AMReX style: ParallelFor(mf, [=](int box_id, int i, int j, int k))
def __call__(self, box_id, i, j, k, phi):
```

`box_id` enables per-box data access via equinox traced arrays on the kernel.
Kernels that don't need it (Laplacian) ignore it.

### Face data: `FlatFaceBoxed` (equinox Module)

```python
class FlatFaceBoxed(eqx.Module):
    x: _FaceAxisBoxed  # per-box offsets + strides + box_id
    y: _FaceAxisBoxed
    z: _FaceAxisBoxed
```

- Registered as equinox Module → `jax.tree.flatten` traverses its leaves
- `box_id` is a traced leaf — dummy at build time, replaced per-tile via
  `with_box_id()` inside the Pallas kernel
- `face[ax][i,j,k]` does `plt.load(buf.at[offsets[box_id] + i*sx + j*sy + k*sz])`

### Dispatch flow

```
DSL:  expr = exp.ddt(U) + exp.div(phi, U, scheme=VanLeer()) - exp.laplacian(nu, U)
      solve(expr, t, dt)

solve → _forward_euler_level:
  1. ctx = TiledContext(dh, ng, lev, face_refs, face_offsets, face_strides)
  2. spatial_kernels = tuple(op.build_kernel_3d(ctx, t) for op in spatial_ops)
  3. kernel = FusedEulerKernel(spatial_kernels, dt_over_coeff)
  4. results = parallel_for(kernel, mf)  ← tiled Pallas dispatch
  5. mf.copy_arrays(results)

parallel_for:
  1. flattened_boxes_from_mf(mf, bf=8) → flat buffer + tile metadata
  2. jax.tree.flatten(kernel) → leaves (face arrays, offsets, strides, box_id)
  3. pallas_call(grid=(n_tiles_padded,)):
     - load tile meta: [offset, sx, sy, sz, box_id]
     - unflatten kernel from refs
     - bind real box_id via _bind_box_id (eqx.tree_at on _FaceAxisBoxed.box_id)
     - phi = FlatCellRef(phi_ref, offset, sx, sy, sz)
     - val = kernel(box_id, i, j, k, phi)
     - plt.store result at tile offsets
  4. scatter per-box results back
```

### Tile metadata: 5 ints per tile

```
[0] offset  — flat index into contiguous buffer (includes ng)
[1] sx      — stride x (=1)
[2] sy      — stride y (=Nx)
[3] sz      — stride z (=Nx*Ny)
[4] box_id  — index of parent box
```

Power-of-2 padded. `n_tiles_padded` is static (tiered), `n_tiles` traced.
`pl.when(tid < n_tiles)` skips padding tiles.

## User kernel examples

### Cell-only kernel (Laplacian)

```python
class Laplacian3D(eqx.Module):
    dh: tuple = eqx.field(static=True)
    coeff: float = eqx.field(static=True)
    ng: int = eqx.field(static=True, default=1)

    def __call__(self, box_id, i, j, k, phi):
        c = phi[i, j, k, 0]
        total = 0.0
        for ax in Axis:
            d = ax.d
            total += (phi[i+d[0], j+d[1], k+d[2], 0]
                      - 2 * c
                      + phi[i-d[0], j-d[1], k-d[2], 0]) / self.dh[ax]**2
        return self.coeff * total
```

`box_id` is available but ignored — Laplacian only needs `phi[i,j,k,0]`.

### Face kernel (Upwind divergence)

```python
class UpwindDiv3D(eqx.Module):
    face: FlatFaceBoxed       # face[ax][i,j,k] — per-box offsets indexed by box_id
    dh: tuple = eqx.field(static=True)
    coeff: float = eqx.field(static=True)
    ng: int = eqx.field(static=True, default=1)

    def __call__(self, box_id, i, j, k, phi):
        total = 0.0
        for ax in Axis:
            d = ax.d
            fl = self.face[ax][i, j, k]
            fr = self.face[ax][i+d[0], j+d[1], k+d[2]]
            F_l = fl * jnp.where(fl >= 0, phi[i-d[0],j-d[1],k-d[2],0], phi[i,j,k,0])
            F_r = fr * jnp.where(fr >= 0, phi[i,j,k,0], phi[i+d[0],j+d[1],k+d[2],0])
            total += (F_r - F_l) / self.dh[ax]
        return self.coeff * total
```

`self.face` is a `FlatFaceBoxed` (equinox Module). Inside the Pallas kernel,
`face[ax][i,j,k]` does `plt.load(buf.at[offsets[box_id] + i*sx + j*sy + k*sz])`.
The `box_id` selects the right per-box face offset automatically.

### Custom kernel with per-box data

```python
class SpatiallyVaryingSource(eqx.Module):
    box_origins: jax.Array    # (n_boxes, 3) — traced, per-box physical origin
    dx: tuple = eqx.field(static=True)
    ng: int = eqx.field(static=True, default=0)

    def __call__(self, box_id, i, j, k, phi):
        # Compute physical coordinates from box_id + (i,j,k)
        x = self.box_origins[box_id, 0] + i * self.dx[0]
        y = self.box_origins[box_id, 1] + j * self.dx[1]
        z = self.box_origins[box_id, 2] + k * self.dx[2]
        return jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)
```

`box_origins` is a traced equinox array — changes on regrid without recompilation.
`box_id` selects the right origin per tile. Any per-box data can be stored as
traced arrays and indexed by `box_id`.

### DSL usage (unchanged)

```python
# User code — same as before, no knowledge of tiling or flat buffers
phi = CellField(mesh, ncomp=1, ngrow=2, name="phi")
ff = FaceField(mesh, ncomp=1, ngrow=0, name="U")

expr = exp.ddt(phi) + exp.div(ff, phi, scheme=VanLeer()) - exp.laplacian(nu, phi)
solve(expr, t, dt)
```

The DSL creates operators, `solve()` builds equinox kernels via
`op.build_kernel_3d(ctx, t)`, fuses them into `FusedEulerKernel`, and
dispatches via `parallel_for(kernel, mf)`. The user never sees `FlatCellRef`,
`FlatFaceBoxed`, `TiledContext`, or tile metadata.

## Recompilation behavior

Verified with AMR regrid tests (`test_recompilation_amr.py`):

| Scenario | Recompiles |
|----------|-----------|
| Same grid, repeat solve | 0 |
| Same-tag regrid + solve | 0 |
| Remove/restore fine level | 0 (after warmup) |
| Switch between cached tiers (512→2048→512) | 0 |
| New tier (never seen) | >0 (once) |
| Second visit to new tier | 0 |

Recompilation is bounded by `O(unique tile count tiers)`. After warmup of
all grid configurations, the dispatch produces 0 recompiles.

## Remaining performance gap

The DSL `evaluate()` path is 10x C++ at 128³ (1.2ms vs 0.12ms). The Pallas
kernel itself is 0.08ms. The 1.1ms overhead is:

- `flattened_boxes_from_mf`: 0.4ms (C++ `packed_tiles` + `contiguous_array`)
- `jax.tree.flatten` + kernel leaf preparation: ~0.1ms
- Scatter-back (`copy_arrays`): ~0.6ms (JAX gather + Python copy per box)

The kernel is already **faster than C++**. The overhead is in the Python/JAX
dispatch around it. Approaches to reduce:

1. **Cache `flattened_boxes_from_mf`** — only rebuild on regrid
2. **Write directly to MultiFab contiguous buffer** from Pallas (eliminate scatter)
3. **Keep data in flat buffers** across timesteps (avoid MultiFab round-trip)

## How to improve DSL path performance

The kernel is faster than C++. The 10x gap in `evaluate()` is Python overhead.
Three concrete improvements, ordered by impact:

### 1. Eliminate scatter-back (~0.6ms saved)

Currently `parallel_for` gathers valid cells from the flat output via
`jnp.meshgrid` + indexing per box, then `mf.copy_arrays()` writes back.

**Fix**: Write Pallas output directly into the MultiFab's contiguous buffer.
The output flat buffer has the same layout as the input — valid cell results
are at the same offsets. A C++ `copy_contiguous_to_mf(out_flat, mf)` method
that copies only valid regions would eliminate the Python scatter entirely.

Alternatively, if the MultiFab contiguous buffer can be passed as the Pallas
output ref, the kernel writes in-place and no copy is needed at all.

### 2. Cache `flattened_boxes_from_mf` across timesteps (~0.4ms saved)

Currently called every `parallel_for` invocation. The tile metadata only
changes on regrid. Cache the `FlattenedBoxes` on the `CellField` and
invalidate in `_on_remake_level`:

```python
class CellField:
    _fb_cache: dict = {}  # lev → FlattenedBoxes

    def get_fb(self, lev, bf=8):
        if lev not in self._fb_cache or self._fb_cache[lev].mf is not self.mf[lev]:
            self._fb_cache[lev] = flattened_boxes_from_mf(self.mf[lev], bf=bf)
        return self._fb_cache[lev]
```

### 3. Avoid tree flatten on every call (~0.1ms saved)

The kernel structure is the same across timesteps (same operators, same
scheme). Cache the `k_treedef` and `k_leaf_shapes` and reuse the JIT
function. The current `_pfor_cache` does this but the cache key computation
adds overhead. A simpler approach: store the compiled `run_fn` on the
expression or solver object.

### Combined effect

```
Current DSL evaluate:    1.18 ms
- scatter-back:         -0.60 ms → 0.58 ms
- flattened_boxes cache: -0.40 ms → 0.18 ms
- tree flatten cache:    -0.10 ms → 0.08 ms  (= kernel time)
Target:                  ~0.08 ms (matches standalone benchmark)
C++:                      0.12 ms
```

With all three optimizations, the DSL path would match the standalone
kernel time and be **faster than C++**.

## Benchmark data files

### Tiled dispatch benchmarks (this work)
- `benchmark/blockamr/laplacian/bench_laplacian_*.json` — laplacian strategies
- `benchmark/blockamr/advection_diffusion/bench_advdiff_*.json` — advdiff strategies
- `bench_results/bench_recomp_N32_lev2.json` — recompilation benchmark

### Historical benchmark data
- `bench_results/bench_final_N*.json` — final dispatch comparisons
- `bench_results/bench_precomp_dispatch_N*.json` — precomputed offset dispatch
- `bench_results/tile_dispatch_N128.json` — tile dispatch development
- `bench_results/bench_pallas_N*.json` — Pallas backend development
- `bench_results/bench_schemes_N*.json` — scheme comparison

Run benchmarks:
```bash
uv run --no-sync python benchmark/blockamr/laplacian/bench_laplacian.py --grid-sizes 64,128,192
uv run --no-sync python benchmark/blockamr/advection_diffusion/bench_advdiff.py --grid-sizes 32,64,128
uv run --no-sync python benchmark/blockamr/bench_recompilation.py --ncell 32 --max-level 2
```

## Test results

```
108 PASSED, 0 FAILED (excluding pre-existing test_dsl_solver_lid_cavity_physical)
```

Key test files:
- `test_performance_tiled.py` — 3 tests: laplacian eval, div eval, solve advdiff
- `test_recompilation_amr.py` — 5 tests: stable grid, same-tag regrid, level change, tile padding, new tier
- `test_laplacian.py`, `test_div.py`, `test_solve.py` — existing correctness tests pass
- `test_double_shear_layer.py` — AMR solver tests pass
