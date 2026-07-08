# GPU Tile Table Construction

## Context

The current `tile_table` C++ binding (`multifab.cpp:650`) builds per-tile
metadata on the **host** (CPU), then copies to GPU via `htod_memcpy`. For
a 192^3 grid with bf=8 this creates 13,824 tiles × 9 arrays × 8 bytes =
~1MB of metadata, copied host→device every regrid.

For the tile-based Pallas dispatch with `FlatCellRef`, we need:
- **Cell tile**: `(offset, sx, sy, sz, sc)` — 5 ints per tile
- **Face tile**: `(fx_off, fx_sx, fx_sy, fx_sz, fy_off, ..., fz_off, ...)` — 12 ints
- **Output**: `(out_offset)` — 1 int per tile
- **Total**: 18 ints per tile, packed contiguous

The host→device copy is fast (~0.1ms for 1MB) but avoidable. On regrid,
AMReX already has the box metadata on GPU. We can compute tile metadata
directly with an AMReX `ParallelFor` kernel.

## Current Implementation

```
multifab.cpp:tile_table():
  1. MFIter loop on CPU: count tiles
  2. Allocate host arrays (new int64_t[n_padded])
  3. MFIter loop on CPU: fill tile descriptors
  4. Pad with tile 0 copies
  5. htod_memcpy each array to GPU
  6. Return as JAX device arrays
```

**Cost**: Two MFIter passes + 9 host allocations + 9 H2D copies.
At 13K tiles this is negligible (~1ms), but at 100K+ tiles (AMR with
many small boxes) it becomes noticeable.

## Proposed: GPU-Side Tile Table Construction

### Approach 1: AMReX ParallelFor (simplest)

Use AMReX's `ParallelFor` to fill the tile table in a single GPU kernel:

```cpp
void buildTileTableGPU(
    MultiFab& mf,
    int bf,
    int64_t* d_packed,      // output: (n_padded, FIELDS_PER_TILE) on device
    int64_t* d_box_starts,  // input: prefix sum of tiles per box
    int n_padded)
{
    // Pre-compute on host: per-box tile count and prefix sum
    // This is O(n_boxes) — very fast
    Vector<int> tiles_per_box;
    Vector<int64_t> box_offsets;
    int64_t cell_offset = 0;

    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto bx = mf[mfi].box();
        int Nx = bx.length(0), Ny = bx.length(1), Nz = bx.length(2);
        int ng = mf.nGrow();
        int vNx = Nx - 2*ng, vNy = Ny - 2*ng, vNz = Nz - 2*ng;
        tiles_per_box.push_back((vNx/bf) * (vNy/bf) * (vNz/bf));
        box_offsets.push_back(cell_offset);
        cell_offset += (int64_t)Nx * Ny * Nz * mf.nComp();
    }

    // Prefix sum (on host, O(n_boxes))
    Vector<int64_t> tile_starts(tiles_per_box.size() + 1, 0);
    for (int i = 0; i < tiles_per_box.size(); ++i)
        tile_starts[i+1] = tile_starts[i] + tiles_per_box[i];

    int n_tiles = tile_starts.back();
    // Copy tile_starts and box_offsets to device (tiny: ~n_boxes ints)

    // GPU kernel: each thread computes one tile's metadata
    ParallelFor(n_tiles, [=] AMREX_GPU_DEVICE (int t) {
        // Binary search to find which box this tile belongs to
        int box_id = upper_bound(d_box_starts, n_boxes, t) - 1;
        int local_t = t - d_box_starts[box_id];

        // Box dimensions
        int Nx = ..., Ny = ..., Nz = ...;  // from per-box metadata
        int vNx = Nx - 2*ng, vNy = Ny - 2*ng, vNz = Nz - 2*ng;
        int tiles_y = vNy / bf, tiles_z = vNz / bf;

        int ti = local_t / (tiles_y * tiles_z);
        int tj = (local_t / tiles_z) % tiles_y;
        int tk = local_t % tiles_z;

        int ci = ng + ti * bf;
        int cj = ng + tj * bf;
        int ck = ng + tk * bf;

        int64_t sx = 1, sy = Nx, sz = (int64_t)Nx * Ny;

        // Pack into contiguous output
        int base = t * FIELDS_PER_TILE;
        d_packed[base + 0] = box_offsets[box_id] + ci*sx + cj*sy + ck*sz;
        d_packed[base + 1] = sx;
        d_packed[base + 2] = sy;
        d_packed[base + 3] = sz;
        d_packed[base + 4] = t * bf * bf * bf;  // sequential output offset
        // ... face offsets similarly
    });

    // Pad remaining slots (n_tiles..n_padded) with tile 0 values
    if (n_tiles < n_padded) {
        ParallelFor(n_padded - n_tiles, [=] AMREX_GPU_DEVICE (int i) {
            int src = 0;  // copy from tile 0
            int dst = (n_tiles + i) * FIELDS_PER_TILE;
            for (int f = 0; f < FIELDS_PER_TILE; ++f)
                d_packed[dst + f] = d_packed[src + f];
        });
    }
}
```

**Advantages**:
- Tile metadata computed on GPU in a single kernel launch
- Only box-level metadata (n_boxes × few ints) needs host→device copy
- Padding done on GPU

**Complexity**: Needs per-box metadata accessible on device (box dimensions,
cell offsets, face offsets). This can be a small device array updated on
regrid.

### Approach 2: Two-Level Metadata (no binary search)

Avoid the binary search by using a box-to-tile indirection:

```
Level 1: Per-box metadata (on device, ~n_boxes × 20 ints)
    box_cell_offset, Nx, Ny, Nz, nc,
    fx_offset, fx_Nx, fx_Ny, fx_Nz,
    fy_offset, fy_Nx, fy_Ny, fy_Nz,
    fz_offset, fz_Nx, fz_Ny, fz_Nz,
    tile_start_idx, n_tiles_in_box

Level 2: Per-tile → box mapping (on device, n_tiles × 1 int)
    tile_to_box[t] = box_id
```

The Pallas kernel reads `tile_to_box[tile_id]` to get its box, then
reads box metadata from Level 1. This avoids pre-computing per-tile
strides entirely — the kernel computes them from box dimensions.

```python
def pallas_kernel(phi_ref, box_meta_ref, tile_to_box_ref, out_ref):
    tile_id = pl.program_id(0)
    box_id = plt.load(tile_to_box_ref.at[tile_id])

    # Read box metadata
    bm = box_id * BOX_META_SIZE
    cell_off = plt.load(box_meta_ref.at[bm + 0])
    Nx       = plt.load(box_meta_ref.at[bm + 1])
    Ny       = plt.load(box_meta_ref.at[bm + 2])
    Nz       = plt.load(box_meta_ref.at[bm + 3])
    sx = 1; sy = Nx; sz = Nx * Ny

    # Compute tile position from local index
    local_t = tile_id - plt.load(box_meta_ref.at[bm + TILE_START_IDX])
    tiles_z = (Nz - 2*ng) // bf
    tiles_y = (Ny - 2*ng) // bf
    tk = local_t % tiles_z
    tj = (local_t // tiles_z) % tiles_y
    ti = local_t // (tiles_z * tiles_y)

    ci = ng + ti * bf; cj = ng + tj * bf; ck = ng + tk * bf
    off = cell_off + ci * sx + cj * sy + ck * sz

    # ... stencil computation with FlatCellRef(phi_ref, off, sx, sy, sz)
```

**Advantages**:
- Box metadata is tiny (~n_boxes × 20 ints), easily fits in L2 cache
- `tile_to_box` is just n_tiles ints — can be built with a single
  `ParallelFor` or even `thrust::fill` per box segment
- No per-tile stride arrays needed — strides computed from box dims
- Box metadata can be **persistent** — allocated once, updated in-place on regrid

**Disadvantages**:
- Extra indirection (2 loads instead of 1) per tile to get box metadata
- Division/modulo for tile position within box (but these are on constants
  after the box_id load, so latency-hidden)

### Approach 3: Persistent Pre-Allocated Buffers

Pre-allocate the tile metadata buffer at max capacity on GPU. On regrid,
update in-place:

```cpp
class TileTableGPU {
    int64_t* d_packed;       // device: (max_tiles, FIELDS_PER_TILE)
    int64_t* d_box_meta;     // device: (max_boxes, BOX_META_SIZE)
    int*     d_tile_to_box;  // device: (max_tiles,)
    int max_tiles;
    int max_boxes;
    int n_tiles;             // current actual count

    void rebuild(MultiFab& cell_mf, MultiFab* face_mfs[3], int bf) {
        // Update box metadata (host → device, tiny)
        // Launch ParallelFor to fill tile table
        // Update n_tiles
        // No allocation — buffers are persistent
    }
};
```

The JAX/Pallas kernel uses `BlockSpec(block_shape=(max_tiles * FIELDS,))`
which is **static** — never recompiles. The actual tile count is passed as
`n_tiles` (dynamic), and `pl.when(tile_id < n_tiles)` skips padding.

## Recommended Approach

**Approach 2 (Two-Level Metadata)** is the best fit because:

1. **Minimal data**: only n_boxes × 20 ints on device (not n_tiles × 18)
2. **No tile-level precomputation**: strides derived from box metadata at
   kernel runtime — Triton optimizes this since it's uniform within a box
3. **Easy GPU construction**: `tile_to_box` array built with one `ParallelFor`
   per regrid; box metadata is just fab_metadata packaged differently
4. **Cache-friendly**: box metadata fits in L1/L2, accessed by all tiles
   in the same box
5. **Persistent allocation**: both arrays are fixed-size (max_boxes, max_tiles)
   — zero allocation on regrid

## Implementation Steps

1. **C++ `TileTableGPU` class** in `src/bindings/blockAMR/tile_table_gpu.cpp`
   - Pre-allocate `d_box_meta` and `d_tile_to_box` at construction
   - `rebuild()` method: update box metadata + tile_to_box via ParallelFor
   - Return as JAX device arrays (zero-copy via nanobind ndarray)

2. **Python `TileTableGPU` wrapper** in `src/neon/blockamr/tile_table.py`
   - `rebuild_from_multifab(mf, face_mfs, bf)` — calls C++ rebuild
   - Exposes `box_meta`, `tile_to_box`, `n_tiles` as JAX arrays

3. **Update Pallas dispatch** to use two-level metadata pattern
   - `parallel_for_tiled()` reads box_meta + tile_to_box
   - Computes per-tile offsets/strides from box dimensions inside kernel

4. **Wire into field lifecycle**
   - `CellField._on_new_level()` triggers `TileTableGPU.rebuild()`
   - Mesh regrid updates tile_to_box without reallocation

## Key Files

- `src/bindings/blockAMR/multifab.cpp:650` — current host-side tile_table
- `src/neon/blockamr/tile_table.py` — Python TileTable wrapper
- `benchmark/blockamr/flat_refs.py` — FlatCellRef/FlatFaceRef
- `benchmark/blockamr/plans/tile_dispatch_pallas.md` — dispatch architecture
