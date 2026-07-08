# Plan: Zero-Recompile Double vmap Dispatch

| | |
|---|---|
| **Complexity** | Low — changes to 3 files, no kernel code changes |
| **Impact** | High — eliminates JIT recompilation spikes on AMR regrid |
| **Risk** | Low — kernel `__call__` and `for_box` unchanged |
| **Priority** | High |

## Goal

Modify the double vmap dispatch so that `process_bucket` never recompiles during AMR regrid. Currently, 4 out of 6 static fields in `BucketContext` change on regrid, triggering JIT recompilation.

## Current Static Fields (all trigger recompile)

```python
class BucketContext(eqx.Module):
    # Traced (safe — never trigger recompile):
    box_offsets: Array      # per-box buffer offsets
    cell_buf: Array         # flat contiguous buffer
    Nx_arr, Ny_arr, Nz_arr  # per-box grown dimensions
    n_cells_arr: Array      # per-box valid cell count
    dh_arr: Array           # per-box cell spacing

    # Static (trigger recompile when changed):
    ng: int                 # ghost cells — stable ✓
    n_cells_padded: int     # tier ceiling — stable with tiers ✓
    max_boxes: int          # box count padding — changes on regrid ✗
    n_valid: int            # actual box count — changes on regrid ✗
    box_indices: tuple      # MF index mapping — changes on regrid ✗
    lev: int                # AMR level — stable ✓
```

## Target: Only Stable Static Fields

After the refactor:

```python
class BucketContext(eqx.Module):
    # Traced:
    box_offsets: Array
    cell_buf: Array
    Nx_arr, Ny_arr, Nz_arr: Array
    n_cells_arr: Array
    dh_arr: Array
    box_indices_arr: Array   # NEW: traced int32 array replaces static tuple

    # Static (stable — rarely change):
    ng: int                  # ghost cells — never changes
    n_cells_padded: int      # cell tier — changes only on tier boundary
    max_boxes: int           # box tier (128, 256, ...) — changes only on tier boundary
    lev: int                 # AMR level — never changes
```

Removed from static:
- `n_valid` → not needed in the JIT kernel (only used Python-side for scatter)
- `box_indices` → replaced by traced `box_indices_arr`

## Changes Required

### 1. `flattened_boxes.py` — BucketContext

```python
class BucketContext(eqx.Module):
    box_offsets: Array
    cell_buf: Array
    Nx_arr: Array
    Ny_arr: Array
    Nz_arr: Array
    n_cells_arr: Array
    dh_arr: Array
    box_indices_arr: Array              # traced: (max_boxes,) int32

    ng: int = eqx.field(static=True)
    n_cells_padded: int = eqx.field(static=True)
    max_boxes: int = eqx.field(static=True)
    lev: int = eqx.field(static=True, default=0)

    def replace_buf(self, new_buf):
        return eqx.tree_at(lambda s: s.cell_buf, self, new_buf)

    @property
    def n_valid(self):
        """Derive from n_cells_arr: count boxes with n_cells > 0."""
        # Only used Python-side for scatter — not JIT traced
        return int((self.n_cells_arr > 0).sum())
```

### 2. `flattened_boxes.py` — build_fixed_buckets

```python
MAX_BOXES_START = 128

def _box_tier(n):
    """Smallest box tier >= n. Starts at 128, grows by 2x."""
    t = MAX_BOXES_START
    while t < n:
        t *= 2
    return t

def build_fixed_buckets(fb, dh, lev=0):
    """Tier-bucketed cells + tiered box count. Minimal recompilation."""
    ng = fb.n_grow
    n_boxes = len(fb.offsets)

    # Group by cell-count tier (same as current)
    tier_groups = {}
    for b in range(n_boxes):
        Nx, Ny, Nz = fb.shapes[b][:3]
        vNx, vNy, vNz = Nx-2*ng, Ny-2*ng, Nz-2*ng
        n_cells = vNx * vNy * vNz
        tier = _cell_tier(n_cells)
        tier_groups.setdefault(tier, []).append(
            (b, int(fb.offsets[b]), Nx, Ny, Nz, n_cells))

    result = []
    for tier, boxes in tier_groups.items():
        n_valid = len(boxes)
        mb = _box_tier(n_valid)  # 128, 256, 512, ... — rarely changes

        offsets = [b[1] for b in boxes]
        indices = [b[0] for b in boxes]
        Nx_list = [b[2] for b in boxes]
        Ny_list = [b[3] for b in boxes]
        Nz_list = [b[4] for b in boxes]
        nc_list = [b[5] for b in boxes]

        # Pad to mb — dummy boxes replicate first box
        pad = mb - n_valid
        offsets += [offsets[0]] * pad
        indices += [indices[0]] * pad
        Nx_list += [Nx_list[0]] * pad
        Ny_list += [Ny_list[0]] * pad
        Nz_list += [Nz_list[0]] * pad
        nc_list += [0] * pad  # dummy boxes: 0 valid cells

        dh_data = [list(dh)] * mb

        bucket = BucketContext(
            box_offsets=jnp.array(offsets[:mb], dtype=jnp.int32),
            cell_buf=fb.contiguous_array,
            Nx_arr=jnp.array(Nx_list[:mb], dtype=jnp.int32),
            Ny_arr=jnp.array(Ny_list[:mb], dtype=jnp.int32),
            Nz_arr=jnp.array(Nz_list[:mb], dtype=jnp.int32),
            n_cells_arr=jnp.array(nc_list[:mb], dtype=jnp.int32),
            dh_arr=jnp.array(dh_data, dtype=jnp.float64),
            box_indices_arr=jnp.array(indices[:mb], dtype=jnp.int32),
            ng=ng,
            n_cells_padded=tier,
            max_boxes=mb,
            lev=lev,
        )
        result.append((bucket, n_valid))  # return real n_valid for scatter

    return result
```

### 3. `operators/div.py` — Div.build_kernel

The Div operator uses `bucket.box_indices` (static tuple) to build `face_offsets`. Replace with `bucket.box_indices_arr` (traced):

```python
def build_kernel(self, bucket, t):
    lev = bucket.lev
    face_fb = FlattenedFaceBoxes.from_face_field(self.face_field, lev)

    # Build face_offsets from traced box_indices_arr
    # For each direction, gather face offsets by box index
    face_offsets = tuple(
        face_fb.offsets[d][bucket.box_indices_arr]  # traced gather, not Python list comp
        for d in range(3)
    )

    ng_face = self.face_field[lev][0].mf.n_grow()
    return self.scheme.build_kernel(
        face_bufs=face_fb.bufs, face_offsets=face_offsets,
        Nx=bucket.Nx_arr, Ny=bucket.Ny_arr, Nz=bucket.Nz_arr,
        ng=bucket.ng, dh=bucket.dh_arr, coeff=self.coeff,
        ncomp=self.cell_field.ncomp, ng_face=ng_face,
    )
```

Key change: `face_fb.offsets[d][bucket.box_indices_arr]` is a **traced gather** from a traced array with a traced index array. This replaces the Python list comprehension `[int(face_fb.offsets[d][mf_idx]) for mf_idx in bucket.box_indices]` which created a static array from a static tuple.

### 4. `operators/laplacian.py` — _build_gamma_buffer

Same pattern: replace `bucket.box_indices` iteration with `bucket.box_indices_arr` gather for the gamma buffer offsets.

### 5. `dsl/solve.py` — scatter

Scatter needs the real `n_valid` (not the padded `max_boxes`). Since `n_valid` is no longer in BucketContext, it's returned alongside the bucket from `build_fixed_buckets`:

```python
def _forward_euler_level(expr, cell_field, lev, t, dt, ddt_coeff):
    mf = cell_field.mf[lev]
    fb = flattened_boxes_from_mf(mf)
    dh = tuple(float(d) for d in cell_field.mesh.geom(lev).cell_size())

    for bucket, n_valid in build_fixed_buckets(fb, dh, lev=lev):
        kernels = tuple(op.build_kernel(bucket, t) for op in expr.spatial_ops)
        result = process_bucket(bucket, dt / ddt_coeff, kernels)
        # Scatter only real boxes (n_valid), skip dummy padding
        _scatter_results(all_results, result, bucket, n_valid)

    mf.copy_arrays(all_results)
```

### 6. `bucket_dispatch.py` — process_bucket

**No changes needed.** The kernel vmaps over `max_boxes` × `n_cells_padded`. Dummy boxes (n_cells_arr=0) compute stencils on replicated data — the result is discarded by scatter. The `jnp.where(cell_idx < actual_n_cells, result, 0.0)` masking already handles this.

## Recompilation Behavior After Refactor

| Event | Static fields affected | Recompile? |
|---|---|---|
| Normal timestep | None | **No** |
| Regrid (same box count tier) | None — n_valid, box_indices are traced | **No** |
| Regrid (box count crosses tier) | `max_boxes` (128→256) | **Yes** (once, ~1 per run) |
| Regrid (new cell size tier) | `n_cells_padded` | **Yes** (rare) |
| New AMR level | `lev`, `ng` | **Yes** (once at startup) |

**Expected: 0 recompiles during steady-state simulation.** At most `log₂(max_boxes/128)` + number of cell tiers encountered ≈ 2-5 total across an entire run.

## What Stays the Same

- `CellAccessor`, `FaceAccessor`, `StencilAxis` — unchanged
- All kernel `__call__` methods — unchanged
- All kernel `for_box` methods — unchanged (they read from traced arrays, not static fields)
- `process_bucket`, `evaluate_bucket` — unchanged
- All divergence scheme kernels (Upwind, Linear, VanLeer, QUICK) — unchanged
- Test suite — should pass with no changes (build_fixed_buckets is additive)

## Migration Path

1. Add `box_indices_arr` as a traced field to `BucketContext` (keep `box_indices` tuple for backward compat)
2. Add `build_fixed_buckets` as a new function (keep `build_buckets` unchanged)
3. Update `Div.build_kernel` to use `box_indices_arr` when available
4. Update `solve.py` to use `build_fixed_buckets`
5. Run full test suite — verify bit-identical results
6. Benchmark recompilation with moving-band tagging — verify 0 recompiles
7. Remove `build_buckets` and `box_indices` tuple once validated

## Verification

```bash
# Correctness
uv run --no-sync python -m pytest test/blockamr/ -x -q

# Recompilation
uv run --no-sync python benchmark/blockamr/bench_dispatch_strategies.py \
    --ncell 32 --max-level 2 --max-size 16 --steps 50 --regrid-interval 2

# Expected output for strategy D (tiered+fixed):
#   normal [ms]  regrid [ms]  recompiles
#       ~160        ~160          0
```
