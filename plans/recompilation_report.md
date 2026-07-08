# Recompilation Report — AMR Regrid Behavior

## Summary

Recompilation with AMR regrid is **predictable and bounded**. Once all grid
configurations have been seen (warmup), subsequent visits produce **0 recompiles**.
The 3 recompiles/step observed in `bench_recompilation.py` come from
`update_face_fluxes` (jax.vmap on velocity function), not from the DSL solve path.

## Test Results (`test_recompilation_amr.py`)

Using `AmrMesh` with `CellField`, `FaceField`, and the DSL `solve()` path
(which dispatches via `parallel_for` under the hood):

| Phase | Action | Recompiles | Why |
|-------|--------|------------|-----|
| Warmup | First solve on 2-level grid | N (compiles) | First trace of each static shape |
| Phase 1 | 3× solve on same grid | **0** | Same static shapes → cached |
| Phase 2 | Regrid with same tagging + solve | **0** | Box layout unchanged (4+16 boxes) |
| Phase 3 | Remove fine level + solve | 0* | Level-0 shape already cached |
| Phase 4 | Restore fine level + solve | **0** | 2-level shape already cached from warmup |

\* Phase 3 produces 0 recompiles because the level-0 dispatch was already
compiled during warmup. The fine level is simply not dispatched.

## Benchmark Analysis (`bench_recompilation.py`, N=32, max_level=2)

### Per-step recompilation pattern

```
Backend   Step 0 (regrid)  Steps 1-19   Total
C++       0                0/step       0
JAX       144              0/step*      1452
Pallas    39               3/step       96
```

\*JAX only recompiles on regrid steps (0,2,4,...) with varying counts (34-346).
Pallas has a constant 3 recompiles per step on all steps (including non-regrid).

### Root cause of the 3/step on Pallas

The 3 recompiles per step come from `update_face_fluxes` → `FaceFluxUpdater`
→ `jax.vmap(vel_func)`, **not** from `solve()`. This is because:

1. `update_face_fluxes` creates new `jnp.meshgrid` coordinate arrays each call
2. The batched velocity evaluation uses `jax.jit` + `jax.vmap` with these arrays
3. 3 levels × 1 face-flux compilation = 3 recompiles per step

The DSL solve path itself produces **0 recompiles** after warmup.

### What triggers recompilation in the solve path

The solve path (`_dispatch_level` → `parallel_for`) recompiles when any
**static** field on `BoxContext` or `MBCellArray` changes:

| Static field | Changes when | Impact |
|-------------|-------------|--------|
| `BoxContext.n_padded` | Box count crosses power-of-2 boundary | Retrace |
| `BoxContext.shapes` | Box dimensions change (rare on same level) | Retrace |
| `BoxContext.max_shape` | Largest box changes size | Retrace |
| `BoxContext.dh` | Cell spacing changes (new AMR level) | Retrace |
| `FusedEulerKernel.dt_over_coeff` | dt changes | Retrace |

All traced (non-static) fields — `box_offsets`, `cell_buf`, face data —
change freely on regrid without recompilation.

### Tile padding and tier crossing verified

Tile counts pad to the next power of 2. When regrid changes the tile count
across a power-of-2 boundary, JAX recompiles once for the new tier, then
caches it. Subsequent visits to any previously-seen tier produce 0 recompiles.

Tested with AMR regrid crossing the 512→2048 tier:

```
Config A (small tag): lev1 = 63 boxes, 504 tiles, padded=512
Config B (large tag): lev1 = 156 boxes, 1248 tiles, padded=2048

After warmup of both tiers:
  Switch to A: 0 recompiles
  Switch to B: 0 recompiles
  Switch to A: 0 recompiles
```

The padded tile array shape (`n_padded * 5`) is static and determines the
JAX trace. The actual tile count `n_tiles` is traced (used in `pl.when`)
and changes freely without recompilation.

### New tier recompiles once, then is cached

Tested with 3 tag widths producing 3 different padded tiers (256, 1024, 2048):

```
Width=0.10: 160 tiles, padded=256   → first visit: 147 recompiles, second: 0
Width=0.25: 648 tiles, padded=1024  → first visit:  35 recompiles, second: 0
Width=0.40: 1920 tiles, padded=2048 → first visit:  35 recompiles, second: 0

Revisiting all 3 cached tiers: 0, 0, 0 recompiles
```

First-visit recompile count decreases after tier 1 because level-0 dispatch
is already cached — only the new level-1 tier triggers a retrace.

This is continuously verified in `test_recompilation_amr.py::test_amr_new_tier_recompiles_once`.

### Recompilation is bounded by O(unique configurations)

After warmup covers all grid configurations that appear during the simulation,
the solve path produces **0 recompiles**. For a typical AMR run with fixed
`max_level` and `max_grid_size`, the number of unique configurations is:

```
n_configs ≤ max_level × n_power_of_2_tiers
```

For max_level=2 with box counts in the range [1, 128], this is at most
2 × 7 = 14 configurations. After warmup, all are cached.

## Approaches to Eliminate Face Flux Recompilation

The 3 recompiles/step are the dominant cost. Two code paths exist:

### Current: `update_face_fluxes` (no caching)

```python
def _fill_face_component(comp, d, vel_func, dx, prob_lo, t):
    for mfi in MFIterator(comp.mf):
        # Rebuild coordinates every call
        X, Y, Z = jnp.meshgrid(...)       # ← new shapes each regrid
        vel = vel_func(X, Y, Z, t)        # ← unbatched, per-box JIT
```

Problem: `jnp.meshgrid` creates new JAX arrays with shapes tied to box
dimensions. When the box layout changes on regrid, these are new shapes →
retrace. Even without regrid, calling `vel_func` per-box means N separate
JIT entries instead of one batched call.

### Current: `FaceFluxUpdater` (precomputed coordinates, batched)

```python
class FaceFluxUpdater:
    def __init__(self, face_fluxes, vel_func, geom):
        # Precompute coordinates per box, group by shape, stack
        all_X = jnp.stack([...])           # (n_boxes_in_group, nx, ny, nz)

    @jax.jit
    def _batched_vel(all_X, all_Y, all_Z, t):
        return jax.vmap(vel_func)(...)     # single batched call
```

Better: coordinates are precomputed once in `__init__`, batched by shape group.
But `AmrFaceFluxUpdater._rebuild()` creates a new `FaceFluxUpdater` on regrid
→ new `jnp.stack` → new stacked shapes → retrace.

### Approach A: Pad coordinate arrays to stable shapes

Same idea as `MBCellArray`: pad the stacked coordinate arrays to a power-of-2
box count and the max face dimensions. On regrid, the padded shape stays the
same (unless box count crosses a tier) — only the contents change.

```python
class StableFaceFluxUpdater:
    def __init__(self, face_field, vel_func, geom, max_boxes=None):
        # Precompute per-box coords, pad to (max_boxes, max_nx, max_ny, max_nz)
        # max_boxes = next_power_of_2(n_boxes) — static, rarely changes
        self._coords = ...   # padded, traced
        self._n_valid = ...  # traced scalar

    def rebuild(self, face_field, geom):
        # Update coords contents in-place (same padded shape)
        # Only recompiles if max_boxes tier changes

    @jax.jit
    def _eval(coords_x, coords_y, coords_z, n_valid, t):
        return jax.vmap(vel_func)(...)  # padded vmap, mask with n_valid
```

Recompiles: only when box count crosses a power-of-2 boundary (same as solve).

### Approach B: Compute face fluxes inside the Pallas kernel

Instead of precomputing face velocity on a separate MultiFab, evaluate the
velocity function inside the stencil kernel itself. The kernel already has
`box_id` and tile position → can compute physical coordinates on the fly:

```python
class DivWithVelocity(eqx.Module):
    vel_func: callable   # static
    prob_lo: tuple       # static
    dx: tuple            # static

    def __call__(self, box_id, i, j, k, phi):
        # Compute face-centre coordinates from (i,j,k) + geometry
        x_face = self.prob_lo[0] + (i + 0.0) * self.dx[0]  # x-face
        # Evaluate velocity inline
        u = self.vel_func(x_face, ...)
        # Compute divergence
```

Eliminates face MultiFab entirely. No precomputation, no recompilation from
face data. The velocity function is a static field (baked into the kernel),
and all coordinate computation uses traced `(i, j, k)` values.

Trade-off: more compute per cell (velocity evaluation repeated at every face),
but no memory traffic for face buffers and zero face-related recompilation.

### Approach C: Use `parallel_for` with `box_id` for face flux fill

Use the new `box_id` parameter to write face fluxes via `parallel_for` instead
of the Python MFIterator loop. Store per-box coordinate offsets as traced
equinox arrays on the kernel:

```python
class FaceFluxKernel(eqx.Module):
    box_lo: jax.Array    # (n_boxes, 3) traced — box origin in index space
    dx: tuple            # static
    prob_lo: tuple       # static
    ng: int = eqx.field(static=True, default=0)

    def __call__(self, box_id, i, j, k, phi):
        lo = self.box_lo[box_id]
        x = self.prob_lo[0] + (lo[0] + i + 0.5) * self.dx[0]
        y = self.prob_lo[1] + (lo[1] + j + 0.5) * self.dx[1]
        z = self.prob_lo[2] + (lo[2] + k + 0.5) * self.dx[2]
        u, v, w = vel_func(x, y, z, t)
        return u  # for x-face; separate kernel per direction
```

`box_lo` is traced (changes on regrid, same padded shape → 0 recompiles).
The `parallel_for` machinery handles batching over boxes via vmap.

### Recommendation

**Approach A** for immediate fix (minimal code change — pad existing
`FaceFluxUpdater` coordinate arrays to stable shapes).

**Approach C** for the tiled dispatch follow-up — face fluxes become a
`parallel_for` kernel like everything else, using `box_id` for per-box
coordinate offsets. This aligns with the tile metadata design where all
per-box data lives on the kernel as traced equinox arrays.

## Other Recommendations

1. **The `box_id` parameter** added in this PR flows through the existing
   vmap-over-boxes dispatch without adding recompilation. It's a traced
   value (`jnp.arange(n_padded)`), not static.

2. **The `packed_tiles` C++ method** and `FlattenedBoxes.tiles` field are
   available for the tiled Pallas dispatch follow-up. They are traced arrays
   with static `n_tiles_padded` (power-of-2 tiered) — same recompilation
   properties as the existing dispatch.
