# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import jax.numpy as jnp

import neon.blockamr as blockamr
from ..schemes.laplacian_schemes import CentralDiffLaplacian
from ..dsl.eqterm import EqTerm


class Laplacian(EqTerm):
    """Laplacian operator: div(gamma * grad(phi)).

    gamma can be:
      - callable(x, y, z, t) -> array: evaluated on the grid per box
      - float/int: constant gamma
    """

    kind = "spatial"
    _scheme_operator = "laplacian"
    scheme_key = "laplacian"

    def __init__(self, gamma, field, coeff=1.0, scheme=None):
        super().__init__(
            field, coeff=coeff, coefficient=gamma, scheme=scheme or CentralDiffLaplacian()
        )
        self.gamma = gamma
        self._scheme_explicit = scheme is not None

    def build_kernel_3d(self, ctx, t):
        """Build a 3D spatial kernel from TiledContext.

        Returns an eqx.Module with __call__(box_id, i, j, k, phi) → scalar.
        Stateless — no phi stored on the kernel.

        Parameters
        ----------
        ctx : TiledContext
            Tiled dispatch context (dh, ng, lev).
        t : float
            Current time.
        """
        if isinstance(self.gamma, (int, float)):
            return self.scheme.build_spatial_kernel(dh=ctx.dh, coeff=self.coeff * self.gamma)

        if not callable(self.gamma):
            raise TypeError(f"gamma must be callable or number, got {type(self.gamma)}")

        # Check if gamma is effectively constant
        g1 = float(self.gamma(jnp.array([0.1]), jnp.array([0.2]), jnp.array([0.3]), t)[0])
        g2 = float(self.gamma(jnp.array([0.7]), jnp.array([0.4]), jnp.array([0.15]), t)[0])
        if abs(g1 - g2) < 1e-14 * (abs(g1) + 1.0):
            return self.scheme.build_spatial_kernel(dh=ctx.dh, coeff=self.coeff * g1)

        raise NotImplementedError("Variable gamma with tiled dispatch not yet implemented")

    def build_kernel(self, bucket, t):
        """Build a cell-level kernel for a bucket of boxes."""
        ncomp = self.field.ncomp
        lev = bucket.lev
        mf = self.field.mf[lev]
        geom = self.field.mesh.geom(lev)
        ng = bucket.ng

        if isinstance(self.gamma, (int, float)):
            return self.scheme.build_kernel(
                bucket.dh_arr,
                coeff=self.coeff * self.gamma,
                ncomp=ncomp,
            )

        if not callable(self.gamma):
            raise TypeError(f"gamma must be callable or number, got {type(self.gamma)}")

        # Evaluate gamma at two asymmetric points to check if constant
        g1 = float(self.gamma(jnp.array([0.1]), jnp.array([0.2]), jnp.array([0.3]), t)[0])
        g2 = float(self.gamma(jnp.array([0.7]), jnp.array([0.4]), jnp.array([0.15]), t)[0])
        if abs(g1 - g2) < 1e-14 * (abs(g1) + 1.0):
            return self.scheme.build_kernel(
                bucket.dh_arr,
                coeff=self.coeff * g1,
                ncomp=ncomp,
            )

        # Variable gamma — evaluate on grown-box grids
        gamma_buf, gamma_offsets = _build_gamma_buffer(
            self.gamma,
            mf,
            geom,
            bucket,
            t,
        )
        return self.scheme.build_kernel(
            bucket.dh_arr,
            coeff=self.coeff,
            ncomp=ncomp,
            gamma_buf=gamma_buf,
            gamma_offsets=gamma_offsets,
            Nx=bucket.Nx_arr,
            Ny=bucket.Ny_arr,
            Nz=bucket.Nz_arr,
            ng=ng,
        )


def _build_gamma_buffer(gamma_func, mf, geom, bucket, t):
    """Evaluate gamma_func on grown-box grids and build a flat buffer.

    Returns (gamma_buf, gamma_offsets) where gamma_buf is a 1D contiguous
    array and gamma_offsets[i] is the start of box i's gamma data.
    """
    dx = geom.cell_size()
    prob_lo = geom.prob_lo()
    ng = bucket.ng
    meta = mf.fab_metadata()

    # Evaluate gamma per box (MFIterator order = meta order)
    gamma_per_box = {}
    box_idx = 0
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        lo = bx.small_end()
        m = meta[box_idx]
        bNx, bNy, bNz = m[1], m[2], m[3]

        xs = jnp.array([prob_lo[0] + (lo[0] - ng + i + 0.5) * dx[0] for i in range(bNx)])
        ys = jnp.array([prob_lo[1] + (lo[1] - ng + j + 0.5) * dx[1] for j in range(bNy)])
        zs = jnp.array([prob_lo[2] + (lo[2] - ng + k + 0.5) * dx[2] for k in range(bNz)])
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
        gamma_3d = gamma_func(X, Y, Z, t)
        # Flatten in Fortran order: i fastest, then j, then k
        gamma_flat = gamma_3d.transpose(2, 1, 0).reshape(-1)
        gamma_per_box[box_idx] = gamma_flat
        box_idx += 1

    # Build contiguous buffer for this bucket's boxes
    parts = []
    offsets = []
    cur = 0
    for mf_idx in bucket.box_indices:
        g = gamma_per_box[mf_idx]
        parts.append(g)
        offsets.append(cur)
        cur += len(g)
    # Pad offsets to max_boxes
    dummy = offsets[0] if offsets else 0
    for _ in range(bucket.max_boxes - len(bucket.box_indices)):
        offsets.append(dummy)

    gamma_buf = jnp.concatenate(parts)
    gamma_offsets = jnp.array(offsets[: bucket.max_boxes], dtype=jnp.int32)
    return gamma_buf, gamma_offsets
