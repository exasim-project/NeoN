# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for scheme classes: build_kernel returns correct kernel objects,
   and kernels produce correct results via the accessor-based dispatch."""

import jax
import jax.numpy as jnp

from pydantic import TypeAdapter

from blockamr.schemes.div_schemes import QUICK, DivScheme, Linear, Upwind, VanLeer
from blockamr.schemes.laplacian_schemes import CentralDiffLaplacian
from blockamr.schemes.grad_schemes import CentralDiffGrad
from blockamr.schemes.ddt_schemes import DdtScheme, ForwardEuler, RungeKutta2, RungeKutta4
from blockamr.schemes.schemes_dict import SchemesDict
from blockamr.cell_accessor import CellAccessor, FaceAccessor
from blockamr.flattened_boxes import BucketContext


def _make_flat_cell_buf(data_3d, ng):
    """Pack a 3D array (valid region) into a flat buffer with ghost cells.

    data_3d: shape (nx, ny, nz) — valid cell values.
    Returns: flat buffer of shape (Nx*Ny*Nz,), Nx=nx+2*ng, etc.
    Also returns (Nx, Ny, Nz).
    """
    nx, ny, nz = data_3d.shape
    Nx, Ny, Nz = nx + 2 * ng, ny + 2 * ng, nz + 2 * ng
    grown = jnp.zeros((Nx, Ny, Nz))
    grown = grown.at[ng:ng+nx, ng:ng+ny, ng:ng+nz].set(data_3d)
    # Periodic ghost fill for simplicity
    if ng > 0:
        for ax in range(3):
            sl_lo = [slice(None)] * 3
            sl_hi = [slice(None)] * 3
            sl_src_lo = [slice(None)] * 3
            sl_src_hi = [slice(None)] * 3
            n = [nx, ny, nz][ax]
            sl_lo[ax] = slice(0, ng)
            sl_src_lo[ax] = slice(ng + n - ng, ng + n)
            sl_hi[ax] = slice(ng + n, ng + n + ng)
            sl_src_hi[ax] = slice(ng, ng + ng)
            grown = grown.at[tuple(sl_lo)].set(grown[tuple(sl_src_lo)])
            grown = grown.at[tuple(sl_hi)].set(grown[tuple(sl_src_hi)])
    # Flatten in Fortran order (i-fastest) — matches AMReX planar layout for ncomp=1
    flat = grown.transpose(2, 1, 0).reshape(-1)
    return flat, Nx, Ny, Nz


def _make_flat_face_buf(face_3d, ng, direction):
    """Pack a 3D face array into flat buffer.

    face_3d: shape depends on direction (Nx+1,Ny,Nz) for x-faces, etc.
    Returns flat buffer.
    """
    return face_3d.transpose(2, 1, 0).reshape(-1)


def _eval_kernel_on_valid(kernel, cell_buf, Nx, Ny, Nz, ng, n_cells):
    """Evaluate a kernel on all valid cells via vmap."""
    def eval_cell(cell_idx):
        phi = CellAccessor(cell_buf, 0, cell_idx, Nx, Ny, Nz, ng)
        return kernel(phi)
    return jax.vmap(eval_cell)(jnp.arange(n_cells))


def _make_div_kernel(scheme, u_3d, fx_3d, fy_3d, fz_3d, ng, dh):
    """Build a div kernel from 3D arrays."""
    nx = u_3d.shape[0] - 2 * ng
    ny = u_3d.shape[1] - 2 * ng
    nz = u_3d.shape[2] - 2 * ng
    Nx, Ny, Nz = u_3d.shape
    # Flatten cell buffer (already grown)
    cell_buf = u_3d.transpose(2, 1, 0).reshape(-1)
    # Flatten face buffers
    fx_buf = fx_3d.transpose(2, 1, 0).reshape(-1)
    fy_buf = fy_3d.transpose(2, 1, 0).reshape(-1)
    fz_buf = fz_3d.transpose(2, 1, 0).reshape(-1)
    face_bufs = (fx_buf, fy_buf, fz_buf)
    face_offsets = jnp.array([0], dtype=jnp.int32)
    dh_tuple = tuple(float(d) for d in dh)
    kernel = scheme.build_kernel(
        face_bufs=face_bufs, face_offsets=face_offsets,
        Nx=Nx, Ny=Ny, Nz=Nz, ng=ng, dh=dh_tuple,
    )
    return kernel, cell_buf, Nx, Ny, Nz, ng, nx * ny * nz


def test_upwind_uniform_field_3d() -> None:
    """Upwind on uniform field is zero (all flux in = flux out)."""
    ng = 1
    u = jnp.ones((4, 4, 4))  # Nx=4 with ng=1 → 2 valid cells per axis
    fx = jnp.ones((5, 4, 4))
    fy = jnp.ones((4, 5, 4))
    fz = jnp.ones((4, 4, 5))
    dh = jnp.array([1.0, 1.0, 1.0])

    kernel, cell_buf, Nx, Ny, Nz, ng_, n_cells = _make_div_kernel(
        Upwind(), u, fx, fy, fz, ng, dh)
    result = _eval_kernel_on_valid(kernel, cell_buf, Nx, Ny, Nz, ng_, n_cells)
    assert jnp.allclose(result, 0.0)


def test_upwind_positive_flux() -> None:
    """Upwind with positive flux selects left (upwind) cell."""
    ng = 1
    u = jnp.arange(4.0).reshape(4, 1, 1) * jnp.ones((1, 4, 4))
    fx = jnp.ones((5, 4, 4))
    fy = jnp.zeros((4, 5, 4))
    fz = jnp.zeros((4, 4, 5))
    dh = jnp.array([1.0, 1.0, 1.0])

    kernel, cell_buf, Nx, Ny, Nz, ng_, n_cells = _make_div_kernel(
        Upwind(), u, fx, fy, fz, ng, dh)
    result = _eval_kernel_on_valid(kernel, cell_buf, Nx, Ny, Nz, ng_, n_cells)
    assert result.shape == (8,)  # 2*2*2
    assert jnp.allclose(result, 1.0)


def test_linear_averages() -> None:
    """Linear gives (left + right) / 2 face value -> zero for uniform field."""
    ng = 1
    u = jnp.ones((4, 4, 4))
    fx = jnp.ones((5, 4, 4))
    fy = jnp.ones((4, 5, 4))
    fz = jnp.ones((4, 4, 5))
    dh = jnp.array([1.0, 1.0, 1.0])

    kernel, cell_buf, Nx, Ny, Nz, ng_, n_cells = _make_div_kernel(
        Linear(), u, fx, fy, fz, ng, dh)
    result = _eval_kernel_on_valid(kernel, cell_buf, Nx, Ny, Nz, ng_, n_cells)
    assert jnp.allclose(result, 0.0)


def test_vanleer_uniform() -> None:
    """VanLeer on uniform field is zero."""
    ng = 2
    u = jnp.ones((6, 6, 6))
    fx = jnp.ones((7, 6, 6))
    fy = jnp.ones((6, 7, 6))
    fz = jnp.ones((6, 6, 7))
    dh = jnp.array([1.0, 1.0, 1.0])

    kernel, cell_buf, Nx, Ny, Nz, ng_, n_cells = _make_div_kernel(
        VanLeer(), u, fx, fy, fz, ng, dh)
    result = _eval_kernel_on_valid(kernel, cell_buf, Nx, Ny, Nz, ng_, n_cells)
    assert jnp.allclose(result, 0.0)


def test_vanleer_linear_field() -> None:
    """VanLeer on linear field gives exact face value (limiter=1 for r=1)."""
    ng = 2
    x = jnp.arange(6.0).reshape(6, 1, 1) * jnp.ones((1, 6, 6))
    fx = jnp.ones((7, 6, 6))
    fy = jnp.zeros((6, 7, 6))
    fz = jnp.zeros((6, 6, 7))
    dh = jnp.array([1.0, 1.0, 1.0])

    kernel, cell_buf, Nx, Ny, Nz, ng_, n_cells = _make_div_kernel(
        VanLeer(), x, fx, fy, fz, ng, dh)
    result = _eval_kernel_on_valid(kernel, cell_buf, Nx, Ny, Nz, ng_, n_cells)
    assert jnp.allclose(result, 1.0)


def test_quick_linear_field() -> None:
    """QUICK on linear field gives exact face value."""
    ng = 2
    x = jnp.arange(6.0).reshape(6, 1, 1) * jnp.ones((1, 6, 6))
    fx = jnp.ones((7, 6, 6))
    fy = jnp.zeros((6, 7, 6))
    fz = jnp.zeros((6, 6, 7))
    dh = jnp.array([1.0, 1.0, 1.0])

    kernel, cell_buf, Nx, Ny, Nz, ng_, n_cells = _make_div_kernel(
        QUICK(), x, fx, fy, fz, ng, dh)
    result = _eval_kernel_on_valid(kernel, cell_buf, Nx, Ny, Nz, ng_, n_cells)
    assert jnp.allclose(result, 1.0)


def test_div_scheme_discriminator_roundtrip() -> None:
    """Pydantic discriminator resolves DivScheme from dict."""
    adapter = TypeAdapter(DivScheme)
    for tag, cls in [
        ("Upwind", Upwind),
        ("Linear", Linear),
        ("VanLeer", VanLeer),
        ("QUICK", QUICK),
    ]:
        obj = adapter.validate_python({"type": tag})
        assert isinstance(obj, cls)
    obj = adapter.validate_python({"type": "VanLeer"})
    assert isinstance(obj, VanLeer)


def test_operators_have_build_kernel_method():
    """All spatial operators expose build_kernel."""
    from blockamr.operators.div import Div
    from blockamr.operators.grad import Grad
    from blockamr.operators.laplacian import Laplacian
    from blockamr.operators.source import Source

    assert hasattr(Div, "build_kernel")
    assert hasattr(Grad, "build_kernel")
    assert hasattr(Laplacian, "build_kernel")
    assert hasattr(Source, "build_kernel")


# --- Laplacian schemes ---


def test_laplacian_uniform_field_3d():
    """Laplacian of uniform field with uniform gamma is zero."""
    ng = 1
    phi_3d = jnp.ones((4, 4, 4))
    cell_buf = phi_3d.transpose(2, 1, 0).reshape(-1)
    Nx, Ny, Nz = 4, 4, 4
    dh = (1.0, 1.0, 1.0)
    kernel = CentralDiffLaplacian().build_kernel(dh, coeff=1.0)

    n_cells = 2 * 2 * 2
    result = _eval_kernel_on_valid(kernel, cell_buf, Nx, Ny, Nz, ng, n_cells)
    assert jnp.allclose(result, 0.0)


def test_laplacian_quadratic_field():
    """Laplacian of phi=x^2 with gamma=1 gives 2.0."""
    ng = 1
    x = jnp.arange(4.0)
    phi_3d = (x**2).reshape(4, 1, 1) * jnp.ones((1, 4, 4))
    cell_buf = phi_3d.transpose(2, 1, 0).reshape(-1)
    Nx, Ny, Nz = 4, 4, 4
    dh = (1.0, 1.0, 1.0)
    kernel = CentralDiffLaplacian().build_kernel(dh, coeff=1.0)

    n_cells = 2 * 2 * 2
    result = _eval_kernel_on_valid(kernel, cell_buf, Nx, Ny, Nz, ng, n_cells)
    assert jnp.allclose(result, 2.0)


def test_central_diff_laplacian_discriminator():
    """Pydantic roundtrip for CentralDiffLaplacian."""
    adapter = TypeAdapter(CentralDiffLaplacian)
    obj = adapter.validate_python({"type": "CentralDiffLaplacian"})
    assert isinstance(obj, CentralDiffLaplacian)


# --- Grad schemes ---


def test_grad_linear_field_3d():
    """Gradient of phi=x gives (1, 0, 0)."""
    x = jnp.arange(4.0).reshape(4, 1, 1) * jnp.ones((1, 4, 4))
    dh = jnp.array([1.0, 1.0, 1.0])
    result = CentralDiffGrad().compute(x, dh)
    assert result.shape == (2, 2, 2, 3)
    assert jnp.allclose(result[..., 0], 1.0)
    assert jnp.allclose(result[..., 1], 0.0)
    assert jnp.allclose(result[..., 2], 0.0)


def test_central_diff_grad_discriminator():
    """Pydantic roundtrip for CentralDiffGrad."""
    adapter = TypeAdapter(CentralDiffGrad)
    obj = adapter.validate_python({"type": "CentralDiffGrad"})
    assert isinstance(obj, CentralDiffGrad)


# --- Ddt schemes ---


def test_ddt_scheme_discriminator():
    """DdtScheme discriminator resolves all time steppers."""
    adapter = TypeAdapter(DdtScheme)
    for tag, cls in [
        ("ForwardEuler", ForwardEuler),
        ("RungeKutta2", RungeKutta2),
        ("RungeKutta4", RungeKutta4),
    ]:
        obj = adapter.validate_python({"type": tag})
        assert isinstance(obj, cls)


# --- SchemesDict ---


def test_schemes_dict_exact_match():
    """SchemesDict returns exact key match."""
    sd = SchemesDict({"Div": Linear()})
    result = sd.lookup("Div", Upwind())
    assert isinstance(result, Linear)


def test_schemes_dict_default_fallback():
    """SchemesDict falls back to 'default' key."""
    sd = SchemesDict({"default": Linear()})
    result = sd.lookup("Div", Upwind())
    assert isinstance(result, Linear)


def test_schemes_dict_hardcoded_fallback():
    """SchemesDict falls back to hardcoded default when no match."""
    sd = SchemesDict({})
    result = sd.lookup("Div", Upwind())
    assert isinstance(result, Upwind)


def test_schemes_dict_none():
    """SchemesDict with None returns hardcoded default."""
    sd = SchemesDict(None)
    result = sd.lookup("Div", Upwind())
    assert isinstance(result, Upwind)
