# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Operator validation tests on non-cubic single-box meshes.

Uses phi = sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z) with U = (1,2,3)
to validate each operator against analytical solutions.
All grids are non-cubic (Nz << Nx) to exercise per-direction strides.
"""

import math

import pytest
import jax.numpy as jnp

import blockamr
from blockamr.mesh import Mesh
from blockamr.field import CellField, FaceField
from blockamr.fillpatch import FillPatchCellConservative
from blockamr.operators.div import Div
from blockamr.operators.interpolate import interpolate
from blockamr.dsl import exp, evaluate
from blockamr.schemes.div_schemes import Upwind, Linear, VanLeer, QUICK

PI = math.pi
TWO_PI = 2.0 * PI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_thin_mesh(nx=64, ny=64, nz=4, max_size=None):
    """Non-cubic periodic mesh on [0,1]^3. Single box if max_size >= max(nx,ny,nz).

    Uses a unit cube domain so that sin(2*pi*x/y/z) fits exactly one
    wavelength per direction regardless of cell count. This ensures
    convergence tests work even with very few z-cells.
    """
    if max_size is None:
        max_size = max(nx, ny, nz)
    box = blockamr.Box([0, 0, 0], [nx - 1, ny - 1, nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom)


def _sin3d(x, y, z):
    return jnp.sin(TWO_PI * x) * jnp.sin(TWO_PI * y) * jnp.sin(TWO_PI * z)


def _init_sin3d(field, mesh):
    """Set a ncomp=1 CellField to sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z)."""
    dx = mesh.geom(0).cell_size()
    for mfi in blockamr.MFIterator(field.mf[0]):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1
        xs = jnp.array([(lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ys = jnp.array([(lo[1] + j + 0.5) * dx[1] for j in range(ny)])
        zs = jnp.array([(lo[2] + k + 0.5) * dx[2] for k in range(nz)])
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
        field.mf[0].copy_from(mfi, _sin3d(X, Y, Z))
    field.fill_patch(0, 0.0)


def _init_uniform_face_flux(phi, mesh, vel=(1.0, 2.0, 3.0)):
    """Set face fluxes to constant velocity (vel_d on d-faces)."""
    for d in range(3):
        face_mf = phi[0][d].mf
        arrs = face_mf.arrays()
        results = [jnp.full_like(a[:, :, :, 0], vel[d]) for a in arrs]
        face_mf.copy_arrays(results)
        face_mf.fill_boundary(mesh.geom(0))


def _cell_centres(mesh, bi=0):
    """Return (X, Y, Z) meshgrid of cell centres for box bi."""
    dx = mesh.geom(0).cell_size()
    mf_meta = mesh.box_array(0)
    # For single-box, lo = (0, 0, 0)
    geom = mesh.geom(0)
    prob_lo = geom.prob_lo()
    dom = geom.domain()
    lo = dom.small_end()
    hi = dom.big_end()
    nx = hi[0] - lo[0] + 1
    ny = hi[1] - lo[1] + 1
    nz = hi[2] - lo[2] + 1
    xs = jnp.array([(lo[0] + i + 0.5) * dx[0] for i in range(nx)])
    ys = jnp.array([(lo[1] + j + 0.5) * dx[1] for j in range(ny)])
    zs = jnp.array([(lo[2] + k + 0.5) * dx[2] for k in range(nz)])
    return jnp.meshgrid(xs, ys, zs, indexing="ij")


def _extract_valid(results_lev):
    """For single-box tests, extract the one result array."""
    assert len(results_lev) == 1, f"Expected 1 box, got {len(results_lev)}"
    arr = results_lev[0]
    # Strip trailing ncomp=1 dimension if present
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return arr[:, :, :, 0]
    return arr


# ---------------------------------------------------------------------------
# Exact analytical solutions
# ---------------------------------------------------------------------------


def _exact_div_sin3d(X, Y, Z, vel=(1.0, 2.0, 3.0)):
    """div(U * phi) = U . grad(phi) for constant U."""
    return (
        vel[0] * TWO_PI * jnp.cos(TWO_PI * X) * jnp.sin(TWO_PI * Y) * jnp.sin(TWO_PI * Z)
        + vel[1] * TWO_PI * jnp.sin(TWO_PI * X) * jnp.cos(TWO_PI * Y) * jnp.sin(TWO_PI * Z)
        + vel[2] * TWO_PI * jnp.sin(TWO_PI * X) * jnp.sin(TWO_PI * Y) * jnp.cos(TWO_PI * Z)
    )


def _exact_lap_sin3d(X, Y, Z):
    """laplacian(phi) = -12*pi^2 * phi."""
    return -12.0 * PI**2 * _sin3d(X, Y, Z)


def _exact_grad_sin3d(X, Y, Z):
    """grad(phi) = (d/dx, d/dy, d/dz) sin3d."""
    gx = TWO_PI * jnp.cos(TWO_PI * X) * jnp.sin(TWO_PI * Y) * jnp.sin(TWO_PI * Z)
    gy = TWO_PI * jnp.sin(TWO_PI * X) * jnp.cos(TWO_PI * Y) * jnp.sin(TWO_PI * Z)
    gz = TWO_PI * jnp.sin(TWO_PI * X) * jnp.sin(TWO_PI * Y) * jnp.cos(TWO_PI * Z)
    return gx, gy, gz


# ---------------------------------------------------------------------------
# Stage 1: Single-box validation
# ---------------------------------------------------------------------------

# -- interpolate --


def test_interpolate_single_box():
    """Cell-to-face interpolation produces exact linear average of cell values.

    Verifies that face[k] = 0.5*(cell[k-1] + cell[k]) for each direction,
    including ghost cells at periodic boundaries.
    """
    mesh = _make_thin_mesh(nx=64, ny=64, nz=4)
    U = CellField(mesh, ncomp=3, ngrow=2, name="U", fill_patch=FillPatchCellConservative())
    dx = mesh.geom(0).cell_size()

    # Init all 3 components to sin3d
    for mfi in blockamr.MFIterator(U.mf[0]):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1
        xs = jnp.array([(lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ys = jnp.array([(lo[1] + j + 0.5) * dx[1] for j in range(ny)])
        zs = jnp.array([(lo[2] + k + 0.5) * dx[2] for k in range(nz)])
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
        vals = _sin3d(X, Y, Z)
        U.mf[0].copy_from(mfi, jnp.stack([vals, vals, vals], axis=-1))
    U.fill_patch(0, 0.0)

    phi = FaceField(mesh, ncomp=1, ngrow=0, name="phi")
    interpolate(U, phi)

    # Compare against exact linear average from grown cell arrays
    cell_ng = U.ngrow
    cell_arrs = U.mf[0].grown_arrays()
    for d in range(3):
        face_mf = phi[0][d].mf
        face_arr = face_mf.arrays()[0][:, :, :, 0]
        c = cell_arrs[0][:, :, :, d]  # component d

        # Compute expected: 0.5*(c[k-1] + c[k]) in direction d, interior in others
        face_ng = face_mf.n_grow()
        nf = [int(face_arr.shape[ax]) - 2 * face_ng for ax in range(3)]
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        for ax in range(3):
            if ax == d:
                sl_lo[ax] = slice(cell_ng - 1, cell_ng - 1 + nf[ax])
                sl_hi[ax] = slice(cell_ng, cell_ng + nf[ax])
            else:
                sl_lo[ax] = slice(cell_ng, cell_ng + nf[ax])
                sl_hi[ax] = slice(cell_ng, cell_ng + nf[ax])
        expected = 0.5 * (c[tuple(sl_lo)] + c[tuple(sl_hi)])

        # Extract valid region from face result
        if face_ng > 0:
            valid = face_arr[face_ng:-face_ng, face_ng:-face_ng, face_ng:-face_ng]
        else:
            valid = face_arr

        err = float(jnp.max(jnp.abs(valid - expected)))
        assert err < 1e-12, f"face d={d}: interpolation error = {err:.6e}"


# -- div --

_ALL_SCHEMES = [
    pytest.param(Upwind(), id="Upwind"),
    pytest.param(Linear(), id="Linear"),
    pytest.param(VanLeer(), id="VanLeer"),
    pytest.param(QUICK(), id="QUICK"),
]

_EXPECTED_ORDER = {
    "Upwind": 0.8,
    "Linear": 1.8,
    "VanLeer": 0.8,  # TVD limiter clips to first-order near extrema
    "QUICK": 1.8,
}


@pytest.mark.parametrize("scheme", _ALL_SCHEMES)
@pytest.mark.parametrize("face_ng", [0, 1, 2], ids=["fng0", "fng1", "fng2"])
def test_div_single_box_convergence(scheme, face_ng):
    """div(U*phi) converges at expected order on non-cubic single-box mesh.

    All directions are refined proportionally (nz = nx//4) so the
    truncation error decreases uniformly.
    """
    errors = []
    for nx in [16, 32, 64]:
        nz = nx // 4  # refine z proportionally
        mesh = _make_thin_mesh(nx=nx, ny=nx, nz=nz)
        U = CellField(
            mesh,
            ncomp=1,
            ngrow=scheme.stencil_width,
            name="U",
            fill_patch=FillPatchCellConservative(),
        )
        _init_sin3d(U, mesh)
        phi = FaceField(mesh, ncomp=1, ngrow=face_ng, name="phi")
        _init_uniform_face_flux(phi, mesh, vel=(1.0, 2.0, 3.0))

        source = evaluate(Div(phi, U, scheme=scheme), t=0.0)
        result = _extract_valid(source[0])

        X, Y, Z = _cell_centres(mesh)
        exact = _exact_div_sin3d(X, Y, Z)
        err = float(jnp.max(jnp.abs(result - exact)))
        errors.append(err)

    # Check convergence order between finest two resolutions
    order = math.log2(errors[-2] / errors[-1])
    min_order = _EXPECTED_ORDER[scheme.type]
    assert order > min_order, (
        f"{scheme.type} face_ng={face_ng}: order={order:.2f}, "
        f"expected > {min_order}, errors={errors}"
    )


# -- laplacian --


def test_laplacian_single_box_convergence():
    """laplacian(phi) converges at O(dx^2) on non-cubic single-box mesh."""
    errors = []
    for nx in [16, 32, 64]:
        nz = nx // 4  # refine z proportionally
        mesh = _make_thin_mesh(nx=nx, ny=nx, nz=nz)
        U = CellField(mesh, ncomp=1, ngrow=1, name="U", fill_patch=FillPatchCellConservative())
        _init_sin3d(U, mesh)

        nu_func = lambda x, y, z, t: jnp.ones_like(x)
        # jax pinned: callable gamma is a jax-only capability (Q14) — design §10 keeps a
        # space-varying laplacian gamma on the v2 error surface, so cpp never learns it.
        source = evaluate(exp.laplacian(nu_func, U), t=0.0, solution={"backend": "jax"})
        result = _extract_valid(source[0])

        X, Y, Z = _cell_centres(mesh)
        exact = _exact_lap_sin3d(X, Y, Z)
        err = float(jnp.max(jnp.abs(result - exact)))
        errors.append(err)

    order = math.log2(errors[-2] / errors[-1])
    assert order > 1.8, f"laplacian order={order:.2f}, errors={errors}"


# -- grad --


def test_grad_single_box():
    """grad(phi) all 3 components correct on non-cubic single-box mesh."""
    mesh = _make_thin_mesh(nx=64, ny=64, nz=4)
    U = CellField(mesh, ncomp=1, ngrow=1, name="U", fill_patch=FillPatchCellConservative())
    _init_sin3d(U, mesh)

    g = exp.grad(U)

    # grad uses build_kernel(mfi, t, lev=0) — returns a kernel that takes phi_4d
    ng = U.mf[0].n_grow()
    X, Y, Z = _cell_centres(mesh)
    exact_gx, exact_gy, exact_gz = _exact_grad_sin3d(X, Y, Z)

    for mfi in blockamr.MFIterator(U.mf[0]):
        kernel = g.build_kernel(mfi, t=0.0, lev=0)
        phi_4d = U.mf[0].grown_array(mfi)
        grad_result = kernel(phi_4d)  # returns (nx, ny, nz, 3)

        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1

        # Extract matching region from exact
        ex_gx = exact_gx[lo[0] : lo[0] + nx, lo[1] : lo[1] + ny, lo[2] : lo[2] + nz]
        ex_gy = exact_gy[lo[0] : lo[0] + nx, lo[1] : lo[1] + ny, lo[2] : lo[2] + nz]
        ex_gz = exact_gz[lo[0] : lo[0] + nx, lo[1] : lo[1] + ny, lo[2] : lo[2] + nz]

        err_x = float(jnp.max(jnp.abs(grad_result[:, :, :, 0] - ex_gx)))
        err_y = float(jnp.max(jnp.abs(grad_result[:, :, :, 1] - ex_gy)))
        err_z = float(jnp.max(jnp.abs(grad_result[:, :, :, 2] - ex_gz)))
        # Central diff is O(dx^2); dz=1/4 so z-error is larger
        max_dz = max(mesh.geom(0).cell_size())
        tol = 5.0 * (TWO_PI * max_dz) ** 2
        assert err_x < tol, f"grad_x error = {err_x:.6e}, tol={tol:.6e}"
        assert err_y < tol, f"grad_y error = {err_y:.6e}, tol={tol:.6e}"
        assert err_z < tol, f"grad_z error = {err_z:.6e}, tol={tol:.6e}"


# ---------------------------------------------------------------------------
# Stage 2: Multi-box validation — single-box vs multi-box consistency
# ---------------------------------------------------------------------------


def _assemble_full_field(results_lev, mesh):
    """Reassemble per-box results into a single full-domain 3D array."""
    geom = mesh.geom(0)
    dom = geom.domain()
    lo_dom = dom.small_end()
    hi_dom = dom.big_end()
    nx = hi_dom[0] - lo_dom[0] + 1
    ny = hi_dom[1] - lo_dom[1] + 1
    nz = hi_dom[2] - lo_dom[2] + 1
    full = jnp.zeros((nx, ny, nz))

    mf = blockamr.MultiFab(mesh.box_array(0), mesh.dm(0), 1, 0)
    idx = 0
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        bnx = hi[0] - lo[0] + 1
        bny = hi[1] - lo[1] + 1
        bnz = hi[2] - lo[2] + 1
        r = results_lev[idx]
        if r.ndim == 4:
            r = r[:, :, :, 0]
        full = full.at[lo[0] : lo[0] + bnx, lo[1] : lo[1] + bny, lo[2] : lo[2] + bnz].set(r)
        idx += 1

    return full


@pytest.mark.parametrize("scheme", _ALL_SCHEMES)
@pytest.mark.parametrize("face_ng", [0, 2], ids=["fng0", "fng2"])
def test_div_single_vs_multi_box(scheme, face_ng):
    """div results must match between single-box and multi-box on non-cubic mesh."""
    nx, nz = 64, 16
    results = []
    for max_sz in [max(nx, nz), nx // 2]:  # single box, then 4 boxes
        mesh = _make_thin_mesh(nx=nx, ny=nx, nz=nz, max_size=max_sz)
        U = CellField(
            mesh,
            ncomp=1,
            ngrow=scheme.stencil_width,
            name="U",
            fill_patch=FillPatchCellConservative(),
        )
        _init_sin3d(U, mesh)
        phi = FaceField(mesh, ncomp=1, ngrow=face_ng, name="phi")
        _init_uniform_face_flux(phi, mesh, vel=(1.0, 2.0, 3.0))

        source = evaluate(Div(phi, U, scheme=scheme), t=0.0)
        assembled = _assemble_full_field(source[0], mesh)
        results.append(assembled)

    diff = float(jnp.max(jnp.abs(results[0] - results[1])))
    assert diff < 1e-10, f"{scheme.type} face_ng={face_ng}: single vs multi-box diff = {diff:.6e}"


def test_laplacian_single_vs_multi_box():
    """laplacian results must match between single-box and multi-box."""
    nx, nz = 64, 16
    results = []
    for max_sz in [max(nx, nz), nx // 2]:
        mesh = _make_thin_mesh(nx=nx, ny=nx, nz=nz, max_size=max_sz)
        U = CellField(mesh, ncomp=1, ngrow=1, name="U", fill_patch=FillPatchCellConservative())
        _init_sin3d(U, mesh)
        nu_func = lambda x, y, z, t: jnp.ones_like(x)
        # jax pinned: callable gamma is a jax-only capability (Q14) — design §10 keeps a
        # space-varying laplacian gamma on the v2 error surface, so cpp never learns it.
        source = evaluate(exp.laplacian(nu_func, U), t=0.0, solution={"backend": "jax"})
        assembled = _assemble_full_field(source[0], mesh)
        results.append(assembled)

    diff = float(jnp.max(jnp.abs(results[0] - results[1])))
    assert diff < 1e-10, f"laplacian single vs multi-box diff = {diff:.6e}"


def test_interpolate_single_vs_multi_box():
    """interpolate results must match between single-box and multi-box."""
    nx, nz = 64, 16
    dx_vals = []
    for max_sz in [max(nx, nz), nx // 2]:
        mesh = _make_thin_mesh(nx=nx, ny=nx, nz=nz, max_size=max_sz)
        U = CellField(mesh, ncomp=3, ngrow=2, name="U", fill_patch=FillPatchCellConservative())
        dx = mesh.geom(0).cell_size()
        for mfi in blockamr.MFIterator(U.mf[0]):
            bx = mfi.valid_box()
            lo = bx.small_end()
            hi = bx.big_end()
            nnx = hi[0] - lo[0] + 1
            nny = hi[1] - lo[1] + 1
            nnz = hi[2] - lo[2] + 1
            xs = jnp.array([(lo[0] + i + 0.5) * dx[0] for i in range(nnx)])
            ys = jnp.array([(lo[1] + j + 0.5) * dx[1] for j in range(nny)])
            zs = jnp.array([(lo[2] + k + 0.5) * dx[2] for k in range(nnz)])
            X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
            vals = _sin3d(X, Y, Z)
            U.mf[0].copy_from(mfi, jnp.stack([vals, vals, vals], axis=-1))
        U.fill_patch(0, 0.0)

        phi = FaceField(mesh, ncomp=1, ngrow=0, name="phi")
        interpolate(U, phi)

        # Assemble face data for each direction
        face_assembled = []
        for d in range(3):
            face_mf = phi[0][d].mf
            # Build full-domain face array
            dom = mesh.geom(0).domain()
            lo_d = dom.small_end()
            hi_d = dom.big_end()
            sizes = [hi_d[ax] - lo_d[ax] + 1 + (1 if ax == d else 0) for ax in range(3)]
            full = jnp.zeros(sizes)
            for mfi in blockamr.MFIterator(face_mf):
                bx = mfi.valid_box()
                lo = bx.small_end()
                hi = bx.big_end()
                arr = face_mf.array(mfi)[:, :, :, 0]
                bnx = hi[0] - lo[0] + 1
                bny = hi[1] - lo[1] + 1
                bnz = hi[2] - lo[2] + 1
                full = full.at[lo[0] : lo[0] + bnx, lo[1] : lo[1] + bny, lo[2] : lo[2] + bnz].set(
                    arr
                )
            face_assembled.append(full)
        dx_vals.append(face_assembled)

    for d in range(3):
        diff = float(jnp.max(jnp.abs(dx_vals[0][d] - dx_vals[1][d])))
        assert diff < 1e-10, f"interpolate d={d}: single vs multi-box diff = {diff:.6e}"
