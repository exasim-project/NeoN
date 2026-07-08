# Uniform Buffer Dispatch: One Kernel for All Box Sizes

## Problem

With AMR, boxes on a level have varying dimensions (all multiples of blocking factor).
Current approach buckets by shape → one compiled kernel per unique `(Nx, Ny, Nz)`.
With blocking factor 4 on a 64³ domain, up to 4096 distinct shapes → many recompilations
and small vmap batches with no GPU occupancy benefit.

## Core Idea

Pad all boxes on a level into a single uniform buffer of shape
`(n_boxes, MAX_NX, MAX_NY, MAX_NZ, nc)`. Per-box valid extents
`(nx, ny, nz)` are carried as **JAX array data**, not static metadata.
The kernel is compiled once for the max shape; `jnp.where` masks out
padding cells.

```
Box 0: 16×8×12 cells   → slot (MAX, MAX, MAX), valid = (16, 8, 12)
Box 1:  8×16×8 cells   → slot (MAX, MAX, MAX), valid = ( 8, 16,  8)
Box 2: 12×12×16 cells  → slot (MAX, MAX, MAX), valid = (12, 12, 16)
                           ↑ same compiled kernel for all three
```

## Data Structures

```python
class BoxMeta(eqx.Module):
    """Per-box valid extents as JAX arrays. vmap slices these per box."""
    nx: jnp.ndarray    # (n_boxes,) valid x cells (including ghosts)
    ny: jnp.ndarray    # (n_boxes,) valid y cells (including ghosts)
    nz: jnp.ndarray    # (n_boxes,) valid z cells (including ghosts)


class MBCellArray(eqx.Module):
    """Uniform-shape multi-box cell data."""
    data: jnp.ndarray       # (n_boxes, MAX_NX, MAX_NY, MAX_NZ, nc)
    meta: BoxMeta
    ng: int = eqx.field(static=True)
    n_valid: int = eqx.field(static=True)


class MBFaceArray(eqx.Module):
    """Uniform-shape multi-box face data."""
    x: jnp.ndarray          # (n_boxes, MAX_NX+1, MAX_NY, MAX_NZ)
    y: jnp.ndarray          # (n_boxes, MAX_NX, MAX_NY+1, MAX_NZ)
    z: jnp.ndarray          # (n_boxes, MAX_NX, MAX_NY, MAX_NZ+1)
    meta: BoxMeta
    n_valid: int = eqx.field(static=True)
```

## Kernel Dispatch

Existing 3D kernels are unchanged — they still use `self.phi[i, j, k, 0]`.
The masking happens in `parallel_for`:

```python
def parallel_for(stencil, mb: MBCellArray, *, backend=Backend.AUTO):
    ng = stencil.ng
    MAX_NX, MAX_NY, MAX_NZ = mb.data.shape[1:4]
    max_bx = MAX_NX - 2 * ng
    max_by = MAX_NY - 2 * ng
    max_bz = MAX_NZ - 2 * ng
    n_cells = max_bx * max_by * max_bz

    def one_cell(stencil, ci, nx, ny, nz):
        bx, by, bz = nx - 2 * ng, ny - 2 * ng, nz - 2 * ng
        iz = ci % max_bz
        iy = (ci // max_bz) % max_by
        ix = ci // (max_bz * max_by)
        valid = (ix < bx) & (iy < by) & (iz < bz)
        return jnp.where(valid, stencil(ng + ix, ng + iy, ng + iz), 0.0)

    def one_box(phi_box, nx, ny, nz):
        s = eqx.tree_at(lambda s: s.phi.data, stencil, phi_box)
        return jax.vmap(lambda ci: one_cell(s, ci, nx, ny, nz))(
            jnp.arange(n_cells)
        ).reshape(max_bx, max_by, max_bz)

    results = jax.vmap(one_box)(mb.data, mb.meta.nx, mb.meta.ny, mb.meta.nz)
    return MBCellArray(data=results, meta=mb.meta, ng=0, n_valid=mb.n_valid)
```

**One `jax.jit` trace → one XLA compilation → reused for any box count and box sizes.**

## Writing Results Back

Only valid cells are written back to the MultiFab:

```python
def copy_to_multifab(result: MBCellArray, mf):
    for b in range(result.n_valid):
        nx, ny, nz = result.meta.nx[b], result.meta.ny[b], result.meta.nz[b]
        valid_slice = result.data[b, :nx, :ny, :nz, :]
        mf.copy_array(b, valid_slice)
```

## Building the Uniform Buffer

### Option A: dynamic_update_slice (no full copy)

```python
buffer = jnp.zeros((n_padded, MAX_NX, MAX_NY, MAX_NZ, nc))
for b, idx in enumerate(box_indices):
    box = reshape_to_cell_array(contiguous, offset[idx], Nx, Ny, Nz, nc)
    buffer = jax.lax.dynamic_update_slice(buffer, box.data[None], (b, 0, 0, 0, 0))
```

XLA alias analysis can avoid physical copies of the full buffer on each update.

### Option B: AMReX uniform boxes (zero copy)

Set `max_grid_size` so all boxes are identical → contiguous array is already
`n_boxes × Nx × Ny × Nz × nc` → one `reshape` (metadata-only, zero copy).

### Option C: Pre-allocate + scatter

Allocate the max buffer once, reuse across timesteps. After regridding,
re-scatter box data into the existing buffer. Between regrids, only the
data values change, not the layout.

## Compilation Behaviour

| Event | Recompiles? | Why |
|-------|-------------|-----|
| New timestep, same boxes | No | Same shapes, data changes freely |
| AMR regrid, same max size | No | n_boxes and nx/ny/nz are data |
| AMR regrid, larger max | Yes (once) | MAX shape is array shape → new trace |
| Different level (different max) | One per level | Cached after first call |
| Adding/removing boxes | No | n_boxes padded to fixed count |

To avoid even the "larger max" recompilation, pre-allocate to a generous
max (e.g. `max_grid_size` from AMReX config) that won't grow.

## Tradeoffs vs Bucketed Dispatch

| | Bucketed (current) | Uniform buffer |
|---|---|---|
| Compilations | One per unique shape | One per level |
| Wasted compute | None | Padding cells masked out |
| vmap batch size | Small (few boxes per bucket) | Full level (all boxes) |
| GPU occupancy | Poor (many small launches) | Good (one large launch) |
| Code complexity | Bucketing + regrouping | Simple pad + mask |
| Memory | Tight (no padding) | ~2× worst case (half-size boxes) |

## When Uniform Buffer Wins

- Many distinct box shapes (AMR with small blocking factor)
- GPU execution (large vmap batch → full occupancy)
- Frequent regridding (no recompilation cost)

## When Bucketed Dispatch Wins

- Few distinct shapes (uniform `max_grid_size`)
- CPU execution (wasted compute on padding matters more)
- Memory-constrained (no padding overhead)
