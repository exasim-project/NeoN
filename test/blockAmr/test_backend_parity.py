# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""jax vs cpp explicit-backend parity (plan 03 §Tests).

One ``solve()`` forward-Euler step ``exp.ddt(phi) + exp.div(ff, phi)`` is run
on two identically-seeded CellFields — one with ``solution={"backend":"jax"}``,
one with ``{"backend":"cpp"}`` — for every div scheme × ncomp × mesh. The
resulting fields must agree to ``atol=1e-12, rtol=1e-9``.

Tolerance rationale: ``jax_enable_x64=True`` (``src/blockAmr/python/blockamr/__init__.py``)
makes *both* backends float64, so the only difference is float64 summation
order (composable accumulate-then-axpy vs the fused ``FusedEulerKernel``). The
step size ``dt`` is a negative power of two so that ``dt_over_coeff`` is exactly
representable in the ``jnp.float32`` cast the jax fused kernel applies to it —
isolating the composable-kernel numerics from that pre-existing jax f32 scaling
quirk. See loop/blockamr-dsl-03/evidence-slice3.md.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import blockamr
from blockamr.dsl import exp, solve
from blockamr.field import CellField, FaceField
from blockamr.fillpatch import FillPatchCellConservative
from blockamr.mesh import AmrMesh, Mesh
from blockamr.operators.div import Div, update_face_fluxes
from blockamr.schemes.div_schemes import QUICK, Linear, Upwind, VanLeer

# dt = 2**-2 → dt_over_coeff exactly representable as float32 (see module docstring).
DT = 0.25
ATOL = 1e-12
RTOL = 1e-9

_SCHEMES = {"Upwind": Upwind, "Linear": Linear, "VanLeer": VanLeer, "QUICK": QUICK}


def _vel(x, y, z, t):
    """Non-uniform, mixed-sign face velocity (exercises both upwind branches)."""
    u = 1.0 + 0.5 * jnp.sin(2 * jnp.pi * x)
    v = jnp.cos(2 * jnp.pi * y) - 0.3
    w = 0.5 * jnp.ones_like(z)
    return u, v, w


def _build_single_level(ncomp, ngrow):
    n_cell, max_size = 16, 8  # 8 boxes
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    phi_jax = CellField(mesh, ncomp=ncomp, ngrow=ngrow, name="phi_jax")
    phi_cpp = CellField(mesh, ncomp=ncomp, ngrow=ngrow, name="phi_cpp")
    ff = FaceField(mesh, ncomp=1, ngrow=ngrow, name="U")
    return mesh, phi_jax, phi_cpp, ff


def _make_tag_center(mesh, width=0.2):
    def _tag(lev, tags, time, ngrow):
        dx = mesh.geom(lev).cell_size()
        prob_lo = mesh.geom(lev).prob_lo()
        for tbi in blockamr.TagBoxIterator(tags):
            bx = tbi.valid_box()
            lo = bx.small_end()
            hi = bx.big_end()
            nx = hi[0] - lo[0] + 1
            ny = hi[1] - lo[1] + 1
            nz = hi[2] - lo[2] + 1
            xs = (np.arange(nx) + lo[0] + 0.5) * dx[0] + prob_lo[0]
            ys = (np.arange(ny) + lo[1] + 0.5) * dx[1] + prob_lo[1]
            mask_2d = (np.abs(xs - 0.5)[:, None] < width) & (np.abs(ys - 0.5)[None, :] < width)
            mask = np.broadcast_to(mask_2d[:, :, None], (nx, ny, nz)).astype(np.int32).copy()
            tbi.set_tags(mask)

    return _tag


def _build_amr_2level(ncomp, ngrow):
    N, Nz, max_size = 32, 8, 16
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, Nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, Nz / N])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    info = blockamr.AmrInfo()
    info.max_level = 1
    info.set_ref_ratio(0, 2)
    info.set_max_grid_size(0, max_size)
    info.set_blocking_factor(0, 8)
    mesh = AmrMesh(geom, info)
    phi_jax = CellField(
        mesh, ncomp=ncomp, ngrow=ngrow, name="phi_jax", fill_patch=FillPatchCellConservative()
    )
    phi_cpp = CellField(
        mesh, ncomp=ncomp, ngrow=ngrow, name="phi_cpp", fill_patch=FillPatchCellConservative()
    )
    ff = FaceField(mesh, ncomp=1, ngrow=ngrow, name="U")
    mesh.init_from_scratch(0.0)
    mesh.regrid(0.0, tag=_make_tag_center(mesh))
    assert mesh.n_levels() == 2, "AMR fixture must produce 2 levels"
    return mesh, phi_jax, phi_cpp, ff


def _seed_identical(fields, mesh, seed):
    """Fill every field's valid cells with the SAME reproducible random data.

    Only one AMReX ``MFIterator`` is active at a time — nested/concurrent
    MFIters abort. Box data is generated once, then written into each field.
    """
    rng = np.random.default_rng(seed)
    for lev in range(mesh.n_levels()):
        data = [
            rng.standard_normal(fields[0].mf[lev].copy_to_host(mfi).shape)
            for mfi in blockamr.MFIterator(fields[0].mf[lev])
        ]
        for f in fields:
            for i, mfi in enumerate(blockamr.MFIterator(f.mf[lev])):
                f.mf[lev].copy_from(mfi, data[i])
    for lev in range(mesh.n_levels()):
        for f in fields:
            f.fill_patch(lev, 0.0)


def _compare_valid(phi_a, phi_b, mesh):
    for lev in range(mesh.n_levels()):
        a_boxes = [phi_a.mf[lev].copy_to_host(mfi) for mfi in blockamr.MFIterator(phi_a.mf[lev])]
        b_boxes = [phi_b.mf[lev].copy_to_host(mfi) for mfi in blockamr.MFIterator(phi_b.mf[lev])]
        for a, b in zip(a_boxes, b_boxes):
            np.testing.assert_allclose(b, a, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("mesh_kind", ["single", "amr2"])
@pytest.mark.parametrize("ncomp", [1, 3])
@pytest.mark.parametrize("scheme_name", ["Upwind", "Linear", "VanLeer", "QUICK"])
def test_div_euler_step_jax_cpp_parity(blockamr_session, scheme_name, ncomp, mesh_kind):
    scheme = _SCHEMES[scheme_name]()
    ngrow = scheme.stencil_width

    builder = _build_single_level if mesh_kind == "single" else _build_amr_2level
    mesh, phi_jax, phi_cpp, ff = builder(ncomp, ngrow)

    _seed_identical([phi_jax, phi_cpp], mesh, seed=20260712)
    for lev in range(mesh.n_levels()):
        update_face_fluxes(ff[lev], _vel, mesh.geom(lev), 0.0)

    solve(
        exp.ddt(phi_jax) + Div(ff, phi_jax, scheme=scheme),
        t=0.0,
        dt=DT,
        solution={"backend": "jax"},
    )
    solve(
        exp.ddt(phi_cpp) + Div(ff, phi_cpp, scheme=scheme),
        t=0.0,
        dt=DT,
        solution={"backend": "cpp"},
    )

    _compare_valid(phi_jax, phi_cpp, mesh)
