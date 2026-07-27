# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

import blockamr
import equinox as eqx
from ..cell_kernels_3d import FusedEulerKernel, CombinedSource
from ..flat_refs import FlatCellRef
from ..tiled_context import TiledContext


def forward_euler(spatial_kernels, dt_over_coeff):
    """Build a fused forward Euler kernel: phi_new = phi - dt * sum(spatial_kernels)."""
    return FusedEulerKernel(spatial_kernels=spatial_kernels, dt_over_coeff=dt_over_coeff)


# ---------------------------------------------------------------------------
# Tiled Pallas dispatch using TileLayout
# ---------------------------------------------------------------------------

BF = 8
NUM_WARPS = 8
NUM_STAGES = 1


def set_tile_size(bf):
    """Set the Pallas tile size (default 8). Must be a power of 2."""
    global BF
    BF = bf


def _forward_euler_level(expr, cell_field, lev, t, dt, ddt_coeff):
    """One forward Euler step for all boxes on one AMR level."""
    dh = tuple(float(d) for d in cell_field.mesh.geom(lev).cell_size())
    ng = cell_field.mf[lev].n_grow()
    dt_over_coeff = dt / ddt_coeff

    ctx = TiledContext(dh=dh, ng=ng, lev=lev)
    spatial_kernels = tuple(op.build_kernel_3d(ctx, t) for op in expr.spatial_ops)
    kernel = FusedEulerKernel(spatial_kernels, dt_over_coeff)

    parallel_for(kernel, cell_field, lev)


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def _run_pallas(k_treedef, n_tiles, n_padded, total_phi, bf, phi_flat, tiles, cvs, *k_leaves):
    """JIT'd Pallas dispatch. Static args define the compilation key.

    cvs: per-box cell-valid-start offsets (int32, n_boxes_padded).
    """
    n_k = k_treedef.num_leaves
    tile_vol = bf**3

    def pallas_kernel(*refs):
        from ..flat_refs import _FaceAxisBoxed

        fn = jax.tree.unflatten(k_treedef, refs[:n_k])
        phi_ref = refs[n_k]
        tiles_ref = refs[n_k + 1]
        cvs_ref = refs[n_k + 2]
        out_ref = refs[n_k + 3]

        tid = pl.program_id(0)
        base = tid * 5
        cell_off = plt.load(tiles_ref.at[base + 0])
        c_sx = plt.load(tiles_ref.at[base + 1])
        c_sy = plt.load(tiles_ref.at[base + 2])
        c_sz = plt.load(tiles_ref.at[base + 3])
        bid = plt.load(tiles_ref.at[base + 4])

        @pl.when(tid < n_tiles)
        def _():
            li = jnp.arange(bf)[:, None, None]
            lj = jnp.arange(bf)[None, :, None]
            lk = jnp.arange(bf)[None, None, :]

            # Compute box-local valid indices from tile offset
            cell_vs = plt.load(cvs_ref.at[bid])
            delta = cell_off - cell_vs
            vi0 = delta % c_sy
            vj0 = (delta // c_sy) % (c_sz // c_sy)
            vk0 = delta // c_sz
            gi = vi0 + li
            gj = vj0 + lj
            gk = vk0 + lk

            phi = FlatCellRef(phi_ref, cell_vs, c_sx, c_sy, c_sz)

            # Bind real box_id on face refs (replaces dummy box_id=0)
            fn_bound = fn
            face_axes = [
                leaf
                for leaf in jax.tree.leaves(fn, is_leaf=lambda x: isinstance(x, _FaceAxisBoxed))
                if isinstance(leaf, _FaceAxisBoxed)
            ]
            if face_axes:
                fn_bound = eqx.tree_at(
                    lambda k: tuple(
                        leaf.box_id
                        for leaf in jax.tree.leaves(
                            k, is_leaf=lambda x: isinstance(x, _FaceAxisBoxed)
                        )
                        if isinstance(leaf, _FaceAxisBoxed)
                    ),
                    fn,
                    tuple(bid for _ in face_axes),
                )

            val = fn_bound(bid, gi, gj, gk, phi)
            oi = cell_vs + gi * c_sx + gj * c_sy + gk * c_sz
            plt.store(out_ref.at[oi.reshape(tile_vol)], val=val.reshape(tile_vol))

    k_leaf_shapes = tuple(leaf.shape for leaf in k_leaves)
    n_cvs = cvs.shape[0]
    in_specs = [pl.BlockSpec(s, lambda i, _n=len(s): (0,) * _n) for s in k_leaf_shapes] + [
        pl.BlockSpec((total_phi,), lambda i: (0,)),
        pl.BlockSpec((n_padded * 5,), lambda i: (0,)),
        pl.BlockSpec((n_cvs,), lambda i: (0,)),
    ]

    return pl.pallas_call(
        pallas_kernel,
        out_shape=jax.ShapeDtypeStruct((total_phi,), phi_flat.dtype),
        grid=(n_padded,),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((total_phi,), lambda i: (0,)),
        compiler_params=plt.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES),
    )(*k_leaves, phi_flat, tiles, cvs)


def _gather_valid(flat, box_off, Nx, Ny, Nz, ng):
    """Extract valid (non-ghost) cells from a flat Fortran-ordered buffer."""
    vNx, vNy, vNz = Nx - 2 * ng, Ny - 2 * ng, Nz - 2 * ng
    ix = jnp.arange(vNx) + ng
    it = jnp.arange(vNy) + ng
    iz = jnp.arange(vNz) + ng
    idx = ix[:, None, None] + Nx * it[None, :, None] + Nx * Ny * iz[None, None, :]
    return flat[box_off + idx.reshape(-1)].reshape(vNx, vNy, vNz)


def _extract_valid_boxes(flat, meta, ng, ncomp):
    """Extract valid cells from flat output for all boxes. Returns list of arrays."""
    results = []
    for m in meta:
        off, Nx, Ny, Nz = int(m[0]), m[1], m[2], m[3]
        if ncomp == 1:
            results.append(_gather_valid(flat, off, Nx, Ny, Nz, ng))
        else:
            bM = Nx * Ny * Nz
            comps = [_gather_valid(flat, off + c * bM, Nx, Ny, Nz, ng) for c in range(ncomp)]
            results.append(jnp.stack(comps, axis=-1))
    return results


def parallel_for(kernel, cell_field, lev, bf=BF, out_mf=None):
    """Tiled Pallas dispatch. Kernel holds all data as equinox leaves.

    For ncomp>1, processes each component separately — the kernel always
    operates on phi[i,j,k,0] (single component). The tile offsets point
    to the same flat buffer; for comp c we shift by c * plane_size per box.
    With uniform boxes (all same size), plane_size is total_phi / ncomp.
    """
    mf = cell_field.mf[lev]
    phi_flat = mf.contiguous_array()

    # Reduce bf so it evenly divides all valid dimensions across all boxes
    ng = mf.n_grow()
    meta = mf.fab_metadata()
    while bf > 1:
        if all(
            (m[1] - 2 * ng) % bf == 0 and (m[2] - 2 * ng) % bf == 0 and (m[3] - 2 * ng) % bf == 0
            for m in meta
        ):
            break
        bf //= 2
    layout = blockamr.build_tile_layout(mf, bf)
    ncomp = cell_field.ncomp
    total_phi = phi_flat.shape[0]

    k_leaves, k_treedef = jax.tree.flatten(kernel)
    k_leaves = [jnp.asarray(leaf) if not hasattr(leaf, "shape") else leaf for leaf in k_leaves]

    # Per-box cell-valid-start: offset to first valid cell (ng, ng, ng)
    n_boxes = len(meta)
    mb = 1
    while mb < n_boxes:
        mb <<= 1
    cvs = jnp.array(
        [int(m[0]) + ng + m[1] * ng + m[1] * m[2] * ng for m in meta],
        dtype=jnp.int32,
    )
    cvs = jnp.pad(cvs, (0, mb - n_boxes), constant_values=int(cvs[0]))

    target_mf = out_mf if out_mf is not None else mf

    if ncomp == 1:
        out_flat = _run_pallas(
            k_treedef,
            layout.n_tiles,
            layout.n_tiles_padded,
            total_phi,
            bf,
            phi_flat,
            layout.tiles,
            cvs,
            *k_leaves,
        )
        # Extract valid cells only — ghost cells must not be overwritten
        results = _extract_valid_boxes(out_flat, meta, ng, ncomp=1)
        target_mf.copy_arrays(results)
    else:
        # For ncomp>1: run the kernel once per component.
        # Shift tile offsets so phi[i,j,k,0] reads component c's data.
        # Fortran order: comp c starts at c*M within each box (M=Nx*Ny*Nz).
        n_boxes = len(meta)
        n_t = layout.n_tiles_padded
        offset_idx = jnp.arange(n_t) * 5

        # Per-tile component stride (M = Nx*Ny*Nz per box)
        box_Ms = [m[1] * m[2] * m[3] for m in meta]
        M0 = box_Ms[0]
        uniform = all(bM == M0 for bM in box_Ms)
        if uniform:
            per_tile_M = jnp.full(n_t, M0, dtype=jnp.int32)
        else:
            comp_strides = jnp.array(box_Ms, dtype=jnp.int32)
            mb = 1
            while mb < n_boxes:
                mb <<= 1
            padded_strides = jnp.pad(
                comp_strides, (0, mb - n_boxes), constant_values=int(comp_strides[0])
            )
            box_ids = layout.tiles[4::5][:n_t]
            per_tile_M = padded_strides[box_ids]

        # Run kernel per component, extract valid cells
        comp_valid = []  # comp_valid[c][bi] = (vNx, vNy, vNz) array
        for c in range(ncomp):
            shifted_tiles = layout.tiles.at[offset_idx].add(per_tile_M * c)
            shifted_cvs = (
                cvs
                + jnp.array(
                    [m[1] * m[2] * m[3] * c for m in meta] + [0] * (mb - n_boxes), dtype=jnp.int32
                )[:mb]
            )
            out_c = _run_pallas(
                k_treedef,
                layout.n_tiles,
                layout.n_tiles_padded,
                total_phi,
                bf,
                phi_flat,
                shifted_tiles,
                shifted_cvs,
                *k_leaves,
            )
            # Extract valid cells for comp c from each box
            per_box = []
            for bi, m in enumerate(meta):
                off, Nx, Ny, Nz = int(m[0]), m[1], m[2], m[3]
                bM = Nx * Ny * Nz
                per_box.append(_gather_valid(out_c, off + c * bM, Nx, Ny, Nz, ng))
            comp_valid.append(per_box)

        # Assemble per-box (vNx, vNy, vNz, ncomp) and write back
        results = []
        for bi in range(n_boxes):
            results.append(jnp.stack([comp_valid[c][bi] for c in range(ncomp)], axis=-1))
        target_mf.copy_arrays(results)


class JaxBackend:
    """Explicit forward-Euler backend on JAX/Pallas tiled GPU dispatch."""

    def euler_step(self, equation, cell_field, lev, t, dt):
        ddt_coeff = equation.temporal_ops[0].coeff
        # The no-IBM path stays the fused kernel, call-for-call ("absent ⇒
        # bitwise the plain operator", api §1). An active band never lands
        # here: solve() routes it through the driver's source_level (which
        # owns the pin) + euler_update (R4).
        _forward_euler_level(equation, cell_field, lev, t, dt, ddt_coeff)

    def source(self, terms, cell_field, lev, t, ibm=None):
        """Accumulated source MultiFab: ``Σ coeff·op(phi)``, band rows applied
        when ``ibm`` is given. The R4 seam — a fresh MultiFab per call (jax has
        no scratch cache), ngrow matching the field (``euler_update`` sweeps
        the valid box by global index, so the ghost band is inert).
        """
        mesh = cell_field.mesh
        mf = cell_field.mf[lev]
        dh = tuple(float(d) for d in mesh.geom(lev).cell_size())
        ng = mf.n_grow()

        ctx = TiledContext(dh=dh, ng=ng, lev=lev)
        spatial_kernels = tuple(op.build_kernel_3d(ctx, t) for op in terms)
        kernel = CombinedSource(spatial_kernels)

        # Create temp MultiFab for output
        out_mf = blockamr.MultiFab(
            mesh.box_array(lev), mesh.dm(lev), cell_field.ncomp, ng, memory="default"
        )
        out_mf.set_val(0.0)

        parallel_for(kernel, cell_field, lev, out_mf=out_mf)

        if ibm is not None:
            # The band overwrite, after the interior sweep of every term and
            # never fused into it (row-format rule R4) — the same call the cpp
            # backend makes, on the same rows: the backend difference is the
            # launch, not the arithmetic.
            ibm.apply(out_mf, lev, t)
        return out_mf

    def evaluate(self, terms, cell_field, lev, t, ibm=None):
        mf = cell_field.mf[lev]
        ng = mf.n_grow()
        out_mf = self.source(terms, cell_field, lev, t, ibm=ibm)

        # Extract per-box valid results
        meta = mf.fab_metadata()
        lev_results = []
        for arr, m in zip(out_mf.arrays(), meta):
            Nx, Ny, Nz = m[1], m[2], m[3]
            vNx, vNy, vNz = Nx - 2 * ng, Ny - 2 * ng, Nz - 2 * ng
            lev_results.append(arr[ng : ng + vNx, ng : ng + vNy, ng : ng + vNz, : cell_field.ncomp])
        return lev_results
