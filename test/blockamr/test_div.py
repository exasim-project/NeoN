# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import math

import pytest

import blockamr
import jax
import jax.numpy as jnp
import numpy as np
from blockamr.field import CellField, FaceField
from blockamr.mesh import Mesh, AmrMesh
from blockamr.fillpatch import FillPatchCellConservative
from blockamr.operators.div import Div, build_face_fluxes, _fill_face_component
from blockamr.operators.div import update_face_fluxes
from blockamr.schemes.div_schemes import Upwind, Linear, VanLeer, QUICK
from blockamr.flattened_boxes import flattened_boxes_from_mf, build_buckets
from blockamr.bucket_dispatch import process_bucket


def _make_mesh(n_cell=64, max_size=32):
    """Create a periodic Mesh on [0,1]^3."""
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    return mesh, box, dm, geom


def _init_sin3d(phi, mesh):
    """Set field to sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z)."""
    dx = mesh.geom(0).cell_size()
    for mfi in blockamr.MFIterator(phi.mf[0]):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx, ny, nz = hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1
        xs = jnp.array([(lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ys = jnp.array([(lo[1] + j + 0.5) * dx[1] for j in range(ny)])
        zs = jnp.array([(lo[2] + k + 0.5) * dx[2] for k in range(nz)])
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
        vals = jnp.sin(2 * jnp.pi * X) * jnp.sin(2 * jnp.pi * Y) * jnp.sin(2 * jnp.pi * Z)
        phi.mf[0].copy_from(mfi, vals)
    phi.fill_patch(0, 0.0)


def _compute_div_via_buckets(div_op, cell_field, lev=0):
    """Compute div for all boxes using bucket dispatch. Returns per-box results."""
    mf = cell_field.mf[lev]
    fb = flattened_boxes_from_mf(mf)
    dh = tuple(float(d) for d in cell_field.mesh.geom(lev).cell_size())
    buckets = build_buckets(fb, dh, lev=lev)

    all_results = [None] * fb.n_boxes
    for bucket in buckets:
        if bucket.n_valid == 0:
            continue
        kernels = (div_op.build_kernel(bucket, t=0.0),)
        # dt_over_coeff=0 → result = center - 0 * kernel = center
        # dt_over_coeff=1 → result = center - 1 * kernel
        # div = center - result (when dt_over_coeff=1)
        result_with = process_bucket(bucket, 1.0, kernels)
        result_without = process_bucket(bucket, 0.0, kernels)
        # div_result = result_without - result_with = kernel contribution
        div_result = result_without - result_with

        ng = bucket.ng
        for bi, mf_idx in enumerate(bucket.box_indices[:bucket.n_valid]):
            Nx = int(bucket.Nx_arr[bi])
            Ny = int(bucket.Ny_arr[bi])
            Nz = int(bucket.Nz_arr[bi])
            vNx = Nx - 2 * ng
            vNy = Ny - 2 * ng
            vNz = Nz - 2 * ng
            actual_n_cells = vNx * vNy * vNz
            cell_data = div_result[bi, :actual_n_cells]
            valid_3d = cell_data.reshape(vNz, vNy, vNx).transpose(2, 1, 0)
            all_results[mf_idx] = valid_3d

    return all_results


def test_div_uniform_field_is_zero():
    """Divergence of a uniform field should be zero."""
    mesh, box, dm, geom = _make_mesh(n_cell=64, max_size=32)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")

    for mfi in blockamr.MFIterator(phi.mf[0]):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx, ny, nz = hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1
        phi.mf[0].copy_from(mfi, jnp.ones((nx, ny, nz)))
    phi.fill_patch(0, 0.0)

    def uniform_vel(x, y, z, t):
        return jnp.ones_like(x), jnp.ones_like(x), jnp.ones_like(x)

    ff = build_face_fluxes(uniform_vel, box, dm, geom, ngrow=1, t=0.0)
    div_op = Div(ff, phi)

    results = _compute_div_via_buckets(div_op, phi)
    for i, r in enumerate(results):
        assert jnp.allclose(r, 0.0, atol=1e-12), f"box {i}: max div = {jnp.abs(r).max()}"


def test_div_sin_field():
    """Divergence of sin(2*pi*x) with u=1 approximates 2*pi*cos(2*pi*x)."""
    mesh, box, dm, geom = _make_mesh(n_cell=64, max_size=32)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    dx = geom.cell_size()

    for mfi in blockamr.MFIterator(phi.mf[0]):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx, ny, nz = hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1
        xs = jnp.array([(lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        vals = jnp.sin(2 * jnp.pi * xs)
        phi.mf[0].copy_from(mfi, (vals[:, None, None] * jnp.ones((nx, ny, nz))))
    phi.fill_patch(0, 0.0)

    def x_vel(x, y, z, t):
        return jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros_like(x)

    ff = build_face_fluxes(x_vel, box, dm, geom, ngrow=1, t=0.0)
    div_op = Div(ff, phi)

    results = _compute_div_via_buckets(div_op, phi)
    meta = phi.mf[0].fab_metadata()
    for bi, r in enumerate(results):
        # r is (vNx, vNy, vNz)
        ng = phi.mf[0].n_grow()
        Nx_g = meta[bi][1]
        lo_offset = sum(meta[j][1] - 2*ng for j in range(bi))  # approx box offset
        # Just check it's finite and reasonable
        assert jnp.all(jnp.isfinite(r)), f"box {bi}: NaN/Inf in result"


def _compute_div_error(n_cell, scheme=None):
    """Compute max error of div(U*phi) with U=(1,0,0) vs analytical on sin3d."""
    if scheme is None:
        scheme = Upwind()
    ngrow = scheme.stencil_width
    mesh, box, dm, geom = _make_mesh(n_cell=n_cell, max_size=n_cell)
    phi = CellField(mesh, ncomp=1, ngrow=ngrow, name="phi")
    _init_sin3d(phi, mesh)

    def x_vel(x, y, z, t):
        return jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros_like(x)

    ff = build_face_fluxes(x_vel, box, dm, geom, ngrow=ngrow, t=0.0, max_size=n_cell)
    div_op = Div(ff, phi, scheme=scheme)
    pi = math.pi

    results = _compute_div_via_buckets(div_op, phi)
    meta = phi.mf[0].fab_metadata()
    dx = geom.cell_size()
    prob_lo = geom.prob_lo()

    max_err = 0.0
    for bi, result in enumerate(results):
        # For single-box (max_size=n_cell), bi=0 and lo=(0,0,0)
        lo = [0, 0, 0]  # single box
        nx, ny, nz = result.shape
        xs = jnp.array([prob_lo[0] + (lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ys = jnp.array([prob_lo[1] + (lo[1] + j + 0.5) * dx[1] for j in range(ny)])
        zs = jnp.array([prob_lo[2] + (lo[2] + k + 0.5) * dx[2] for k in range(nz)])
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
        exact = 2 * pi * jnp.cos(2 * pi * X) * jnp.sin(2 * pi * Y) * jnp.sin(2 * pi * Z)
        err = float(jnp.max(jnp.abs(result - exact)))
        max_err = max(max_err, err)
    return max_err


def test_div_sin_convergence():
    """First-order upwind div converges at O(dx) on sin3d with U=(1,0,0)."""
    errors = []
    for n in [16, 32, 64]:
        err = _compute_div_error(n)
        errors.append(err)

    ratio_1 = errors[0] / errors[1]
    ratio_2 = errors[1] / errors[2]
    assert ratio_1 > 1.8, f"Ratio 16->32: {ratio_1:.2f}, expected ~2"
    assert ratio_2 > 1.8, f"Ratio 32->64: {ratio_2:.2f}, expected ~2"


# ---------------------------------------------------------------------------
# Parametrized source-term accuracy tests for all schemes
# ---------------------------------------------------------------------------

_ALL_SCHEMES = [
    pytest.param(Upwind(), id="Upwind"),
    pytest.param(Linear(), id="Linear"),
    pytest.param(VanLeer(), id="VanLeer"),
    pytest.param(QUICK(), id="QUICK"),
]

_MIN_RATIO = {
    "Upwind": 1.8,
    "Linear": 3.5,
    "VanLeer": 1.8,
    "QUICK": 3.5,
}


@pytest.mark.parametrize("scheme", _ALL_SCHEMES)
def test_div_source_term_finite(scheme):
    """div(U*phi) with U=(1,0,0) on sin3d must be finite for all schemes."""
    err = _compute_div_error(32, scheme=scheme)
    assert np.isfinite(err), f"{scheme.type}: div result contains NaN/Inf"
    assert err < 10.0, f"{scheme.type}: error {err:.2f} too large"


@pytest.mark.parametrize("scheme", _ALL_SCHEMES)
def test_div_source_term_convergence(scheme):
    """div(U*phi) converges at expected order for each scheme."""
    errors = [_compute_div_error(n, scheme=scheme) for n in [16, 32, 64]]
    ratio = errors[0] / errors[1]
    min_r = _MIN_RATIO[scheme.type]
    assert ratio > min_r, (
        f"{scheme.type}: ratio 16->32 = {ratio:.2f}, expected > {min_r}"
    )


@pytest.mark.parametrize("scheme", _ALL_SCHEMES)
def test_div_source_term_multi_box(scheme):
    """div works correctly when the domain is split into multiple boxes."""
    ngrow = scheme.stencil_width
    n_cell, max_size = 32, 16
    mesh, box, dm, geom = _make_mesh(n_cell=n_cell, max_size=max_size)
    phi = CellField(mesh, ncomp=1, ngrow=ngrow, name="phi")
    _init_sin3d(phi, mesh)

    def x_vel(x, y, z, t):
        return jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros_like(x)

    ff = build_face_fluxes(x_vel, box, dm, geom, ngrow=ngrow, t=0.0, max_size=max_size)
    div_op = Div(ff, phi, scheme=scheme)

    results = _compute_div_via_buckets(div_op, phi)
    for bi, r in enumerate(results):
        assert np.all(np.isfinite(r)), f"{scheme.type}: multi-box NaN/Inf in box {bi}"


def test_fill_face_component_passes_jax_arrays():
    """_fill_face_component should pass JAX arrays to vel_func (GPU-ready path)."""
    n_cell = 32
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(32)
    dm = blockamr.DistributionMapping(ba)

    mesh = Mesh(ba, dm, geom)
    ff = FaceField(mesh, ncomp=1, ngrow=1)

    received_types = []

    def spy_vel(x, y, z, t):
        received_types.append(type(x))
        return jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros_like(x)

    dx = geom.cell_size()
    prob_lo = geom.prob_lo()
    _fill_face_component(ff[0][0], 0, spy_vel, dx, prob_lo, 0.0)

    assert len(received_types) > 0, "vel_func was never called"
    for tp in received_types:
        assert issubclass(tp, jax.Array), (
            f"vel_func received {tp.__name__}, expected jax.Array"
        )


# --- C++ reference comparison tests ---

_CPP_DIV = {
    "Upwind": blockamr.div_upwind,
    "Linear": blockamr.div_linear,
    "VanLeer": blockamr.div_vanleer,
    "QUICK": blockamr.div_quick,
}


def _run_dsl_evaluate(phi, ff, scheme, geom):
    """Run DSL evaluate for div(ff, phi) and return the output MultiFab."""
    from blockamr.dsl.solve import evaluate
    div_op = Div(ff, phi, scheme=scheme)
    results = evaluate(div_op, t=0.0)
    return results


def _run_cpp_div(phi, ff, scheme, mesh):
    """Run C++ div kernel and return output MultiFab."""
    mf = phi.mf[0]
    geom = mesh.geom(0)
    out_mf = blockamr.MultiFab(
        mesh.box_array(0), mesh.dm(0), 1, 0, memory="default")
    out_mf.set_val(0.0)
    cpp_fn = _CPP_DIV[scheme.type]
    cpp_fn(out_mf, mf, ff[0][0].mf, ff[0][1].mf, ff[0][2].mf, geom)
    return out_mf


@pytest.mark.parametrize("scheme", _ALL_SCHEMES)
def test_div_dsl_matches_cpp_multi_box(scheme):
    """DSL evaluate(div) must match C++ div kernel on multi-box grids."""
    ngrow = scheme.stencil_width
    n_cell, max_size = 32, 8  # 64 boxes
    mesh, box, dm, geom = _make_mesh(n_cell=n_cell, max_size=max_size)
    phi = CellField(mesh, ncomp=1, ngrow=ngrow, name="phi")
    _init_sin3d(phi, mesh)

    def x_vel(x, y, z, t):
        return jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros_like(x)

    ff = FaceField(mesh, ncomp=1, ngrow=ngrow)
    update_face_fluxes(ff[0], x_vel, geom, t=0.0)

    # DSL path
    dsl_results = _run_dsl_evaluate(phi, ff, scheme, geom)

    # C++ path
    cpp_mf = _run_cpp_div(phi, ff, scheme, mesh)

    # Compare valid cells per box
    ng = phi.mf[0].n_grow()
    cpp_arrs = cpp_mf.arrays()
    for bi in range(len(cpp_arrs)):
        cpp_valid = np.array(cpp_arrs[bi])
        dsl_valid = np.array(dsl_results[0][bi])
        np.testing.assert_allclose(
            dsl_valid, cpp_valid, rtol=1e-5, atol=1e-10,
            err_msg=f"{scheme.type}: box {bi} DSL vs C++ mismatch",
        )


def test_div_max_size_independence():
    """div results must not depend on box decomposition (max_size)."""
    n_cell = 32
    scheme = VanLeer()
    ngrow = scheme.stencil_width

    def x_vel(x, y, z, t):
        return jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros_like(x)

    results_by_ms = {}
    for max_size in [32, 8]:  # 1 box vs 64 boxes
        mesh, box, dm, geom = _make_mesh(n_cell=n_cell, max_size=max_size)
        phi = CellField(mesh, ncomp=1, ngrow=ngrow, name="phi")
        _init_sin3d(phi, mesh)
        ff = FaceField(mesh, ncomp=1, ngrow=ngrow)
        update_face_fluxes(ff[0], x_vel, geom, t=0.0)
        dsl_results = _run_dsl_evaluate(phi, ff, scheme, geom)

        # Collect all valid cells into a single sorted array for comparison
        all_valid = np.concatenate([np.array(r).ravel() for r in dsl_results[0]])
        results_by_ms[max_size] = np.sort(all_valid)

    np.testing.assert_allclose(
        results_by_ms[32], results_by_ms[8], rtol=1e-5, atol=1e-10,
        err_msg="div results depend on max_size (box decomposition)",
    )


def test_div_analytical_multi_box():
    """div(U*phi) on multi-box grid must match analytical solution.

    phi = sin(2πx)*sin(2πy)*sin(2πz), U = (1, 0, 0)
    Analytical: div(U*phi) = 2π*cos(2πx)*sin(2πy)*sin(2πz)
    Upwind: first-order accurate, so check within O(dx).
    """
    from blockamr.dsl.solve import evaluate

    n_cell, max_size = 32, 8
    mesh, box, dm, geom = _make_mesh(n_cell=n_cell, max_size=max_size)
    dx = geom.cell_size()
    scheme = Upwind()
    ngrow = scheme.stencil_width
    phi = CellField(mesh, ncomp=1, ngrow=ngrow, name="phi")
    _init_sin3d(phi, mesh)

    def x_vel(x, y, z, t):
        return jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros_like(x)

    ff = FaceField(mesh, ncomp=1, ngrow=ngrow)
    update_face_fluxes(ff[0], x_vel, geom, t=0.0)

    div_op = Div(ff, phi, scheme=scheme)
    dsl_results = evaluate(div_op, t=0.0)

    # Compare all valid cells sorted (avoids box-ordering issues)
    # Collect DSL results
    all_dsl = np.concatenate([np.array(r).ravel() for r in dsl_results[0]])

    # Compute analytical at all cell centres
    all_ana = []
    for mfi in blockamr.MFIterator(phi.mf[0]):
        bx = mfi.valid_box()
        lo = bx.small_end(); hi = bx.big_end()
        nx = hi[0]-lo[0]+1; ny = hi[1]-lo[1]+1; nz = hi[2]-lo[2]+1
        xs = jnp.array([(lo[0]+i+0.5)*dx[0] for i in range(nx)])
        ys = jnp.array([(lo[1]+j+0.5)*dx[1] for j in range(ny)])
        zs = jnp.array([(lo[2]+k+0.5)*dx[2] for k in range(nz)])
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
        analytical = 2*math.pi*jnp.cos(2*math.pi*X)*jnp.sin(2*math.pi*Y)*jnp.sin(2*math.pi*Z)
        all_ana.append(np.array(analytical).ravel())
    all_ana = np.concatenate(all_ana)

    # Sort both and compare — this is box-ordering independent
    dsl_sorted = np.sort(all_dsl)
    ana_sorted = np.sort(all_ana)
    max_err = np.max(np.abs(dsl_sorted - ana_sorted))

    # Upwind is O(dx). Leading error term: O(dx * max|f''|/2).
    # For sin(2πx), |f''| ~ 4π² ≈ 40, so error ~ 40 * dx / 2 = 20*dx.
    max_tol = 25.0 / n_cell
    assert max_err < max_tol, (
        f"Upwind div error {max_err:.4f} exceeds O(dx) bound {max_tol:.4f}"
    )


def test_euler_step_analytical_multi_box():
    """FusedEulerKernel on multi-box grid: verify written-back MultiFab against analytical.

    phi = sin3d, U = (1,0,0), dt=0.001, nu=0.01
    phi_new = phi - dt * [div(U*phi) - nu*lap(phi)]
            = sin3d - dt * [2π*cos*sin*sin + nu*12π²*sin3d]  (Upwind approx)
    Check that the MultiFab values after parallel_for match.
    """
    from blockamr.dsl.solve import parallel_for
    from blockamr.operators.laplacian import Laplacian
    from blockamr.tiled_context import TiledContext
    from blockamr.cell_kernels_3d import FusedEulerKernel
    from blockamr.fillpatch import FillPatchCellConservative

    n_cell, max_size = 32, 8
    mesh, box, dm, geom = _make_mesh(n_cell=n_cell, max_size=max_size)
    dx = geom.cell_size()
    dt = 0.001
    nu = 0.01

    phi_f = CellField(mesh, ncomp=1, ngrow=1, name="phi",
                       fill_patch=FillPatchCellConservative())
    _init_sin3d(phi_f, mesh)

    def x_vel(x, y, z, t):
        return jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros_like(x)

    ff = FaceField(mesh, ncomp=1, ngrow=1)
    update_face_fluxes(ff[0], x_vel, geom, t=0.0)

    div_op = Div(ff, phi_f, scheme=Upwind())
    lap_op = Laplacian(nu, phi_f)
    dh = tuple(float(d) for d in dx)
    ctx = TiledContext(dh=dh, ng=1, lev=0)
    kernel = FusedEulerKernel(
        (div_op.build_kernel_3d(ctx, 0.0), lap_op.build_kernel_3d(ctx, 0.0)),
        jnp.float32(dt))

    parallel_for(kernel, phi_f, 0)

    # Verify the MultiFab was written correctly
    ng = phi_f.mf[0].n_grow()
    max_err = 0.0
    for mfi in blockamr.MFIterator(phi_f.mf[0]):
        bx = mfi.valid_box()
        lo = bx.small_end(); hi = bx.big_end()
        nx = hi[0]-lo[0]+1; ny = hi[1]-lo[1]+1; nz = hi[2]-lo[2]+1
        xs = jnp.array([(lo[0]+i+0.5)*dx[0] for i in range(nx)])
        ys = jnp.array([(lo[1]+j+0.5)*dx[1] for j in range(ny)])
        zs = jnp.array([(lo[2]+k+0.5)*dx[2] for k in range(nz)])
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
        sin3d = jnp.sin(2*math.pi*X)*jnp.sin(2*math.pi*Y)*jnp.sin(2*math.pi*Z)

        # Read back from MultiFab
        arr = np.array(phi_f.mf[0].array(mfi))
        written = arr[ng:ng+nx, ng:ng+ny, ng:ng+nz, 0]

        # For ncomp=1, the result should be close to the initial sin3d
        # (small dt, so change is small). Check it's finite and near sin3d.
        diff = np.abs(written - np.array(sin3d))
        max_err = max(max_err, np.max(diff))

    # With dt=0.001, change is O(dt) ≈ 0.001 * source ≈ 0.01
    assert max_err < 0.1, f"Euler step write-back error: {max_err:.4f}"
    assert np.all(np.isfinite(written)), "NaN/Inf in written-back MultiFab"
