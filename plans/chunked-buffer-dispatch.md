# Chunked Buffer Dispatch — Flat Element-Level Iteration

| | |
|---|---|
| **Complexity** | Medium — kernel interface refactor touches all operators |
| **Impact** | High — eliminates per-bucket kernel launch overhead and tier padding waste |
| **Risk** | Medium — memory access coalescing on GPU needs benchmarking |
| **Priority** | High |

## Problem

In `_forward_euler_level` and `evaluate`, we loop over buckets in Python:

```python
for bucket in buckets:
    kernels = tuple(op.build_kernel(bucket, t) for op in expr.spatial_ops)
    result = process_bucket(bucket, dt_over_coeff, kernels)
    _scatter_results(all_results, result, bucket)
```

Each `process_bucket` call is a separate `@jax.jit` invocation → a separate kernel launch. The nested vmap (`vmap(boxes) ∘ vmap(cells)`) requires grouping boxes by cell-count tier, which splits work across multiple launches.

## Proposed Solution: Flat Element-Level Dispatch

Instead of the nested box→cell vmap, **consume `contiguous_array` directly** with a single flat element index. Derive an `element_offset` mapping from the existing `offsets` array so each element knows which box it belongs to and what its local cell index is.

### Core Data Structure

Add to `FlattenedBoxes` (or as a new companion):

```python
class FlatElementMap(eqx.Module):
    """Maps flat valid-cell indices to buffer positions and box metadata."""

    cell_buf: Array           # the contiguous_array (traced)
    elem_to_buf_idx: Array    # (total_valid_cells,) → index into cell_buf
    elem_to_box: Array        # (total_valid_cells,) → box index
    elem_to_cell_idx: Array   # (total_valid_cells,) → local cell_idx within box

    # Per-box metadata (looked up via elem_to_box)
    box_Nx: Array             # (n_boxes,)
    box_Ny: Array             # (n_boxes,)
    box_Nz: Array             # (n_boxes,)
    box_offsets: Array        # (n_boxes,) into cell_buf

    ng: int = eqx.field(static=True)
    total_valid_cells: int = eqx.field(static=True)
    n_boxes: int = eqx.field(static=True)
```

### Construction from FlattenedBoxes

```python
def build_element_map(fb):
    """Build flat valid-cell → buffer mapping from FlattenedBoxes."""
    ng = fb.n_grow
    elem_box = []
    elem_cell = []

    for b in range(fb.n_boxes):
        Nx, Ny, Nz = fb.shapes[b][:3]
        vNx, vNy, vNz = Nx - 2*ng, Ny - 2*ng, Nz - 2*ng
        n_valid = vNx * vNy * vNz
        elem_box.extend([b] * n_valid)
        elem_cell.extend(range(n_valid))

    return FlatElementMap(
        cell_buf=fb.contiguous_array,
        elem_to_box=jnp.array(elem_box, dtype=jnp.int32),
        elem_to_cell_idx=jnp.array(elem_cell, dtype=jnp.int32),
        box_Nx=jnp.array([s[0] for s in fb.shapes], dtype=jnp.int32),
        box_Ny=jnp.array([s[1] for s in fb.shapes], dtype=jnp.int32),
        box_Nz=jnp.array([s[2] for s in fb.shapes], dtype=jnp.int32),
        box_offsets=fb.offsets,
        ng=ng,
        total_valid_cells=len(elem_box),
        n_boxes=fb.n_boxes,
    )
```

### Flat Dispatch Kernel

Replace the nested vmap with a single vmap over flat element indices:

```python
@jax.jit
def process_flat(elem_map, dt_over_coeff, kernels):
    """Process all valid cells across all boxes in one kernel launch."""
    ncomp = kernels[0].ncomp

    def process_one_element(flat_idx):
        box_idx = elem_map.elem_to_box[flat_idx]
        cell_idx = elem_map.elem_to_cell_idx[flat_idx]
        Nx = elem_map.box_Nx[box_idx]
        Ny = elem_map.box_Ny[box_idx]
        Nz = elem_map.box_Nz[box_idx]
        box_offset = elem_map.box_offsets[box_idx]

        phi = CellAccessor(
            elem_map.cell_buf, box_offset, cell_idx,
            Nx, Ny, Nz, elem_map.ng,
        )
        total = 0.0
        for k in kernels:
            total = total + k(phi)  # needs flat-compatible kernel interface
        return phi.center - dt_over_coeff * total

    return jax.vmap(process_one_element)(jnp.arange(elem_map.total_valid_cells))
```

One kernel launch processes the entire level. No per-box loop, no bucket grouping.

### Chunked Variant

For very large levels where `total_valid_cells` is huge, chunk the flat index range:

```python
def process_flat_chunked(elem_map, dt_over_coeff, kernels, chunk_size=65536):
    """Process flat element map in fixed-size chunks."""
    total = elem_map.total_valid_cells
    results = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk_result = _process_chunk(elem_map, dt_over_coeff, kernels, start, end)
        results.append(chunk_result)
    return jnp.concatenate(results)
```

Each chunk is one kernel launch of predictable size. `chunk_size` is static → no recompilation as long as only the last chunk differs.

### Writing Results Back

The flat result array maps 1:1 to valid cells. Scatter back via `elem_to_box`:

```python
def scatter_flat_to_boxes(result, elem_map, fb):
    """Reshape flat result → per-box arrays for copy_arrays."""
    ng = elem_map.ng
    all_results = [None] * fb.n_boxes
    offset = 0
    for b in range(fb.n_boxes):
        Nx, Ny, Nz = fb.shapes[b][:3]
        vNx, vNy, vNz = Nx - 2*ng, Ny - 2*ng, Nz - 2*ng
        n_valid = vNx * vNy * vNz
        box_data = result[offset:offset + n_valid]
        all_results[b] = box_data.reshape(vNz, vNy, vNx).transpose(2, 1, 0)[:, :, :, None]
        offset += n_valid
    return all_results
```

## Key Difference from Bucket Approach

| Aspect | Current (Buckets) | Flat Element-Level |
|--------|-------------------|-------------------|
| Dispatch unit | box (nested vmap) | element (single vmap) |
| Grouping | boxes by cell-count tier | none needed |
| Kernel launches per level | N_tiers | 1 (or N_chunks) |
| Padding waste | within-tier cell padding | none — exact valid cell count |
| Static shape dependency | `(max_boxes, n_cells_padded)` | `total_valid_cells` (or `chunk_size`) |
| `CellAccessor` | box_offset from bucket | box_offset looked up via `elem_to_box` |

## Changes Required

| File | Change |
|------|--------|
| `flattened_boxes.py` | Add `FlatElementMap` class and `build_element_map()` |
| `bucket_dispatch.py` | Add `process_flat()` and `evaluate_flat()` functions |
| `solve.py` | `_forward_euler_level`: build element map, call `process_flat`, scatter results |
| `solve.py` | `evaluate`: same pattern with `evaluate_flat` |
| Operator kernels | Need flat-compatible interface — kernel receives box metadata per-element instead of per-bucket `for_box()` binding |

## Implementation Steps

1. **Add `FlatElementMap`** and `build_element_map()` to `flattened_boxes.py`.

2. **Adapt kernel interface**: Currently kernels use `k.for_box(bucket, box_idx)` to bind per-box data (dh, face offsets, etc.). For flat dispatch, kernels need a `for_element(elem_map, flat_idx)` that looks up box metadata via `elem_to_box`. This is the main refactor.

3. **Add `process_flat()`** to `bucket_dispatch.py` — single vmap over `jnp.arange(total_valid_cells)`.

4. **Add `scatter_flat_to_boxes()`** — trivial sequential unpack since elements are ordered by box.

5. **Update `_forward_euler_level`** and `evaluate` in `solve.py`.

6. **Test**: Bit-identical output against current bucket dispatch.

7. **Benchmark**: Measure kernel launch reduction and wall-clock improvement.

8. **(Optional) Add chunked variant** if `total_valid_cells` exceeds GPU thread limits.

## Risks / Considerations

- **Kernel interface refactor**: The `for_box()` → `for_element()` change touches all operator kernels (div, laplacian, grad, etc.). Each kernel currently binds per-box face data, dh, etc. in `for_box()`. The flat version must look these up per-element via `elem_to_box` indirection.
- **Recompilation**: `total_valid_cells` is static. On regrid it changes → recompile. Using fixed `chunk_size` avoids this (only the last chunk's size varies, pad it).
- **Random access pattern**: Elements from different boxes access different regions of `cell_buf`. On GPU this may cause less coalesced memory access than the bucket approach where all elements in a vmap batch come from nearby boxes. Needs benchmarking.
- **Face field access**: `FaceAccessor` also needs the flat element lookup — face buffers have their own offsets array that must be indexed by box.
- **Multi-component**: The ncomp>1 path loops over components — same pattern, just with `comp` parameter on `CellAccessor`.

## Expected Impact

For a level with 256 boxes across 4 tiers:
- **Before**: 4 kernel launches, each with padding waste within its tier
- **After**: 1 kernel launch, zero padding waste (exact valid cell count)

Eliminates both kernel launch overhead AND tier padding waste simultaneously.
