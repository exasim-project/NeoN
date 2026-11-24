# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for scheme classes: build_cpp_kernel returns the scheme's C++ kernel,
   and the kernels produce correct results on a single box with prescribed ghosts."""

import numpy as np
from pydantic import TypeAdapter

import blockamr
from blockamr.field import CellField, FaceField
from blockamr.mesh import Mesh
from blockamr.operators.div import Div
from blockamr.operators.grad import Grad
from blockamr.operators.laplacian import Laplacian
from blockamr.schemes.div_schemes import QUICK, DivScheme, Linear, Upwind, VanLeer
from blockamr.schemes.laplacian_schemes import CentralDiffLaplacian
from blockamr.schemes.grad_schemes import CentralDiffGrad
from blockamr.schemes.ddt_schemes import DdtScheme, ForwardEuler, RungeKutta2, RungeKutta4
from blockamr.schemes.schemes_dict import SchemesDict


def _single_box_mesh(n_valid):
    """One non-periodic box of ``n_valid``^3 cells with unit spacing."""
    box = blockamr.Box([0, 0, 0], [n_valid - 1] * 3)
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [float(n_valid)] * 3)
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 0])
    ba = blockamr.BoxArray(box)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom)


def _set_grown(mf, data_3d):
    """Write a grown (ghost-inclusive) 3D array into a one-box, one-component MultiFab."""
    mf.copy_grown_arrays([np.asfortranarray(np.asarray(data_3d, dtype=float)[..., None])])


def _apply(op, cell_field, out_ncomp=1):
    """Run ``op``'s C++ kernel on level 0 WITHOUT a ghost fill; valid-region result."""
    mesh = cell_field.mesh
    src = blockamr.MultiFab(mesh.box_array(0), mesh.dm(0), out_ncomp, 0)
    src.set_val(0.0)
    op.scheme.build_cpp_kernel().add_to(src, op, cell_field, 0, mesh.geom(0), 0.0)
    out = [src.copy_to_host(mfi) for mfi in blockamr.MFIterator(src)][0]
    return out[..., 0] if out_ncomp == 1 else out


def _div_result(scheme, u_3d, fx_3d, fy_3d, fz_3d, ng):
    """div(U*u) from grown cell / face arrays (ghosts prescribed, not filled)."""
    n_valid = u_3d.shape[0] - 2 * ng
    mesh = _single_box_mesh(n_valid)
    u = CellField(mesh, ncomp=1, ngrow=ng, name="u")
    ff = FaceField(mesh, ncomp=1, ngrow=ng, name="U")
    _set_grown(u.mf[0], u_3d)
    for d, f in enumerate((fx_3d, fy_3d, fz_3d)):
        _set_grown(ff[0][d].mf, f)
    return _apply(Div(ff, u, scheme=scheme), u).ravel()


def _cell_result(make_op, phi_3d, ng, out_ncomp=1):
    """Apply a cell-only operator built by ``make_op(field)`` to a grown array."""
    n_valid = phi_3d.shape[0] - 2 * ng
    mesh = _single_box_mesh(n_valid)
    phi = CellField(mesh, ncomp=1, ngrow=ng, name="phi")
    _set_grown(phi.mf[0], phi_3d)
    return _apply(make_op(phi), phi, out_ncomp)


def test_upwind_uniform_field_3d() -> None:
    """Upwind on uniform field is zero (all flux in = flux out)."""
    ng = 1
    u = np.ones((4, 4, 4))  # Nx=4 with ng=1 → 2 valid cells per axis
    fx = np.ones((5, 4, 4))
    fy = np.ones((4, 5, 4))
    fz = np.ones((4, 4, 5))
    result = _div_result(Upwind(), u, fx, fy, fz, ng)
    assert np.allclose(result, 0.0)


def test_upwind_positive_flux() -> None:
    """Upwind with positive flux selects left (upwind) cell."""
    ng = 1
    u = np.arange(4.0).reshape(4, 1, 1) * np.ones((1, 4, 4))
    fx = np.ones((5, 4, 4))
    fy = np.zeros((4, 5, 4))
    fz = np.zeros((4, 4, 5))
    result = _div_result(Upwind(), u, fx, fy, fz, ng)
    assert result.shape == (8,)  # 2*2*2
    assert np.allclose(result, 1.0)


def test_linear_averages() -> None:
    """Linear gives (left + right) / 2 face value -> zero for uniform field."""
    ng = 1
    u = np.ones((4, 4, 4))
    fx = np.ones((5, 4, 4))
    fy = np.ones((4, 5, 4))
    fz = np.ones((4, 4, 5))
    result = _div_result(Linear(), u, fx, fy, fz, ng)
    assert np.allclose(result, 0.0)


def test_vanleer_uniform() -> None:
    """VanLeer on uniform field is zero."""
    ng = 2
    u = np.ones((6, 6, 6))
    fx = np.ones((7, 6, 6))
    fy = np.ones((6, 7, 6))
    fz = np.ones((6, 6, 7))
    result = _div_result(VanLeer(), u, fx, fy, fz, ng)
    assert np.allclose(result, 0.0)


def test_vanleer_linear_field() -> None:
    """VanLeer on linear field gives exact face value (limiter=1 for r=1)."""
    ng = 2
    x = np.arange(6.0).reshape(6, 1, 1) * np.ones((1, 6, 6))
    fx = np.ones((7, 6, 6))
    fy = np.zeros((6, 7, 6))
    fz = np.zeros((6, 6, 7))
    result = _div_result(VanLeer(), x, fx, fy, fz, ng)
    assert np.allclose(result, 1.0)


def test_quick_linear_field() -> None:
    """QUICK on linear field gives exact face value."""
    ng = 2
    x = np.arange(6.0).reshape(6, 1, 1) * np.ones((1, 6, 6))
    fx = np.ones((7, 6, 6))
    fy = np.zeros((6, 7, 6))
    fz = np.zeros((6, 6, 7))
    result = _div_result(QUICK(), x, fx, fy, fz, ng)
    assert np.allclose(result, 1.0)


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


def test_operators_have_build_cpp_kernel_method():
    """Every spatial operator (or its default scheme) exposes build_cpp_kernel."""
    from blockamr.operators.source import Source

    for scheme in (Upwind(), Linear(), VanLeer(), QUICK(), CentralDiffLaplacian(),
                   CentralDiffGrad()):
        assert hasattr(scheme.build_cpp_kernel(), "add_to")
    assert hasattr(Source, "build_cpp_kernel")


# --- Laplacian schemes ---


def test_laplacian_uniform_field_3d():
    """Laplacian of uniform field with uniform gamma is zero."""
    ng = 1
    phi_3d = np.ones((4, 4, 4))
    result = _cell_result(lambda f: Laplacian(1.0, f), phi_3d, ng)
    assert np.allclose(result, 0.0)


def test_laplacian_quadratic_field():
    """Laplacian of phi=x^2 with gamma=1 gives 2.0."""
    ng = 1
    x = np.arange(4.0)
    phi_3d = (x**2).reshape(4, 1, 1) * np.ones((1, 4, 4))
    result = _cell_result(lambda f: Laplacian(1.0, f), phi_3d, ng)
    assert np.allclose(result, 2.0)


def test_central_diff_laplacian_discriminator():
    """Pydantic roundtrip for CentralDiffLaplacian."""
    adapter = TypeAdapter(CentralDiffLaplacian)
    obj = adapter.validate_python({"type": "CentralDiffLaplacian"})
    assert isinstance(obj, CentralDiffLaplacian)


# --- Grad schemes ---


def test_grad_linear_field_3d():
    """Gradient of phi=x gives (1, 0, 0)."""
    x = np.arange(4.0).reshape(4, 1, 1) * np.ones((1, 4, 4))
    result = _cell_result(Grad, x, 1, out_ncomp=3)
    assert result.shape == (2, 2, 2, 3)
    assert np.allclose(result[..., 0], 1.0)
    assert np.allclose(result[..., 1], 0.0)
    assert np.allclose(result[..., 2], 0.0)


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
