# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for eqx.Module cell-level kernel pattern (build_kernel)."""

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import neon.blockamr as blockamr
from neon.blockamr.field import CellField
from neon.blockamr.flattened_boxes import BucketContext, flattened_boxes_from_mf
from neon.blockamr.mesh import Mesh
from neon.blockamr.operators.div import Div, build_face_fluxes
from neon.blockamr.operators.laplacian import Laplacian
from neon.blockamr.schemes.div_schemes import Upwind, VanLeer
from neon.blockamr.schemes.laplacian_schemes import CentralDiffLaplacian
from neon.blockamr.cell_kernels import CellLaplacianKernel


def _make_mesh(n_cell=32, max_size=32):
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    return mesh, box, dm, geom


def _init_sin3d_cell(phi, geom):
    dx = geom.cell_size()
    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        nx, ny, nz = arr.shape[:3]
        for i in range(nx):
            x = (lo[0] + i + 0.5) * dx[0]
            for j in range(ny):
                y = (lo[1] + j + 0.5) * dx[1]
                for k in range(nz):
                    z = (lo[2] + k + 0.5) * dx[2]
                    arr[i, j, k, 0] = (
                        math.sin(2 * math.pi * x)
                        * math.sin(2 * math.pi * y)
                        * math.sin(2 * math.pi * z)
                    )
        phi.mf[0].copy_from(mfi, arr)
    phi.fill_patch(0, 0.0)


def _make_fluxes(box, dm, geom, ngrow=1):
    def vel(x, y, z, t):
        return np.ones_like(x), np.zeros_like(x), np.zeros_like(x)

    return build_face_fluxes(vel, box, dm, geom, ngrow=ngrow, t=0.0)


def _make_bucket(cell_field, lev=0):
    """Create a BucketContext from a CellField for testing."""
    mf = cell_field.mf[lev]
    fb = flattened_boxes_from_mf(mf)
    meta = mf.fab_metadata()
    Nx, Ny, Nz = meta[0][1], meta[0][2], meta[0][3]
    ng = mf.n_grow()
    dh = tuple(float(d) for d in cell_field.mesh.geom(lev).cell_size())
    n_boxes = len(meta)
    box_indices = tuple(range(n_boxes))
    from neon.blockamr.flattened_boxes import _cell_tier

    n_cells = (Nx - 2 * ng) * (Ny - 2 * ng) * (Nz - 2 * ng)
    return BucketContext(
        box_offsets=fb.offsets,
        cell_buf=fb.contiguous_array,
        Nx_arr=jnp.array([Nx] * n_boxes, dtype=jnp.int32),
        Ny_arr=jnp.array([Ny] * n_boxes, dtype=jnp.int32),
        Nz_arr=jnp.array([Nz] * n_boxes, dtype=jnp.int32),
        n_cells_arr=jnp.array([n_cells] * n_boxes, dtype=jnp.int32),
        dh_arr=jnp.array([list(dh)] * n_boxes, dtype=jnp.float64),
        ng=ng,
        n_cells_padded=_cell_tier(n_cells),
        max_boxes=n_boxes,
        n_valid=n_boxes,
        box_indices=box_indices,
        lev=lev,
    )


# ---------------------------------------------------------------------------
# Operator build_kernel(bucket, t) tests
# ---------------------------------------------------------------------------


class TestDivBuildKernel:
    def test_returns_callable(self):
        """Div.build_kernel(bucket, t) returns a callable eqx.Module."""
        mesh, box, dm, geom = _make_mesh()
        phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
        _init_sin3d_cell(phi, geom)
        ff = _make_fluxes(box, dm, geom)
        div_op = Div(ff, phi)
        bucket = _make_bucket(phi)
        kernel = div_op.build_kernel(bucket, t=0.0)
        assert callable(kernel)
        assert isinstance(kernel, eqx.Module)


class TestLaplacianBuildKernel:
    def test_returns_callable(self):
        mesh, *_ = _make_mesh()
        phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
        _init_sin3d_cell(phi, mesh.geom(0))
        lap_op = Laplacian(lambda x, y, z, t: np.ones_like(x), phi)
        bucket = _make_bucket(phi)
        kernel = lap_op.build_kernel(bucket, t=0.0)
        assert callable(kernel)
        assert isinstance(kernel, CellLaplacianKernel)


# ---------------------------------------------------------------------------
# Scheme-level build_kernel tests
# ---------------------------------------------------------------------------


class TestUpwindSchemeBuildKernel:
    def test_returns_eqx_module(self):
        scheme = Upwind()
        face_bufs = (jnp.zeros(10), jnp.zeros(10), jnp.zeros(10))
        kernel = scheme.build_kernel(
            face_bufs,
            face_offsets=jnp.zeros(1, dtype=jnp.int32),
            Nx=6,
            Ny=6,
            Nz=6,
            ng=1,
            dh=(1.0, 1.0, 1.0),
            coeff=1.0,
        )
        assert isinstance(kernel, eqx.Module)
        assert callable(kernel)


class TestVanLeerSchemeBuildKernel:
    def test_returns_eqx_module(self):
        scheme = VanLeer()
        face_bufs = (jnp.zeros(10), jnp.zeros(10), jnp.zeros(10))
        kernel = scheme.build_kernel(
            face_bufs,
            face_offsets=jnp.zeros(1, dtype=jnp.int32),
            Nx=8,
            Ny=8,
            Nz=8,
            ng=2,
            dh=(1.0, 1.0, 1.0),
            coeff=1.0,
        )
        assert isinstance(kernel, eqx.Module)
        assert callable(kernel)


class TestLaplacianSchemeBuildKernel:
    def test_returns_callable(self):
        scheme = CentralDiffLaplacian()
        kernel = scheme.build_kernel(dh=(0.03125, 0.03125, 0.03125), coeff=1.0)
        assert callable(kernel)
        assert isinstance(kernel, CellLaplacianKernel)


# ---------------------------------------------------------------------------
# Equation.__sub__ mutation test
# ---------------------------------------------------------------------------


class TestEquationSubDoesNotMutate:
    def test_sub_preserves_original_coeff(self):
        from neon.blockamr.dsl.equation import Equation

        mesh, box, dm, geom = _make_mesh()
        phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
        ff = _make_fluxes(box, dm, geom)
        div_op = Div(ff, phi)
        original_coeff = div_op.coeff

        expr = Equation()
        expr = expr - div_op

        assert div_op.coeff == original_coeff
