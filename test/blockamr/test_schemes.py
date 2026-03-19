# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import jax.numpy as jnp

from pydantic import TypeAdapter

from blockamr.schemes.div_schemes import QUICK, DivScheme, Linear, Upwind, VanLeer
from blockamr.schemes.laplacian_schemes import CentralDiffLaplacian
from blockamr.schemes.grad_schemes import CentralDiffGrad
from blockamr.schemes.ddt_schemes import DdtScheme, ForwardEuler, RungeKutta2, RungeKutta4
from blockamr.schemes.schemes_dict import SchemesDict


def test_upwind_uniform_field_3d() -> None:
    """Upwind compute on uniform field is zero (all flux in = flux out)."""
    u = jnp.ones((4, 4, 4))
    fx = jnp.ones((3, 4, 4))
    fy = jnp.ones((4, 3, 4))
    fz = jnp.ones((4, 4, 3))
    dh = jnp.array([1.0, 1.0, 1.0])
    result = Upwind().compute(u, [fx, fy, fz], dh)
    assert result.shape == (2, 2, 2)
    assert jnp.allclose(result, 0.0)


def test_upwind_positive_flux() -> None:
    """Upwind with positive flux selects left (upwind) cell."""
    # 1D-like: 3 cells along x, 1 interior cell, trivial y/z (size 3 → 1 interior)
    # u = [1, 2, 3] along x (ghost, interior, ghost)
    u = jnp.array([[[1.0, 2.0, 3.0]]]).transpose((0, 1, 2))  # shape (3,1,1) won't work
    # Need shape (N+2g, N+2g, N+2g) with g=1 for upwind → need at least 4 along each axis
    # Simpler: use (4,4,4) with known values
    # Along x: values 0,1,2,3 — interior cells are [1] and [2]
    # positive flux → upwind selects left neighbour
    u = jnp.arange(4.0).reshape(4, 1, 1) * jnp.ones((1, 4, 4))
    fx = jnp.ones((3, 4, 4))   # positive flux along x
    fy = jnp.zeros((4, 3, 4))  # zero flux along y
    fz = jnp.zeros((4, 4, 3))  # zero flux along z
    dh = jnp.array([1.0, 1.0, 1.0])
    result = Upwind().compute(u, [fx, fy, fz], dh)
    # interior shape: (2,2,2)
    # Along x with positive flux: F_l = fl * u_left, F_r = fr * u_centre
    # For interior cell i=1 (value=1): F_l = 1*0 = 0, F_r = 1*1 = 1, div = (F_r-F_l)/1 = 1
    # For interior cell i=2 (value=2): F_l = 1*1 = 1, F_r = 1*2 = 2, div = (F_r-F_l)/1 = 1
    assert result.shape == (2, 2, 2)
    assert jnp.allclose(result, 1.0)


def test_linear_averages() -> None:
    """Linear gives (left + right) / 2 face value → zero for uniform field."""
    u = jnp.ones((4, 4, 4))
    fx = jnp.ones((3, 4, 4))
    fy = jnp.ones((4, 3, 4))
    fz = jnp.ones((4, 4, 3))
    dh = jnp.array([1.0, 1.0, 1.0])
    result = Linear().compute(u, [fx, fy, fz], dh)
    assert result.shape == (2, 2, 2)
    assert jnp.allclose(result, 0.0)


def test_vanleer_uniform() -> None:
    """VanLeer on uniform field is zero."""
    u = jnp.ones((6, 6, 6))
    fx = jnp.ones((3, 6, 6))
    fy = jnp.ones((6, 3, 6))
    fz = jnp.ones((6, 6, 3))
    dh = jnp.array([1.0, 1.0, 1.0])
    result = VanLeer().compute(u, [fx, fy, fz], dh)
    assert result.shape == (2, 2, 2)
    assert jnp.allclose(result, 0.0)


def test_vanleer_linear_field() -> None:
    """VanLeer on linear field gives exact face value (limiter=1 for r=1)."""
    # Linear profile along x: 0,1,2,3,4,5 — needs stencil_width=2 → 6 cells, 2 interior
    x = jnp.arange(6.0).reshape(6, 1, 1) * jnp.ones((1, 6, 6))
    fx = jnp.ones((3, 6, 6))
    fy = jnp.zeros((6, 3, 6))
    fz = jnp.zeros((6, 6, 3))
    dh = jnp.array([1.0, 1.0, 1.0])
    result = VanLeer().compute(x, [fx, fy, fz], dh)
    # Linear field with uniform flux: div(u*phi) with phi linear and u=1
    # face values are exact (2.5, 3.5) → F_r - F_l = 3.5 - 2.5 = 1 for both interior cells
    assert result.shape == (2, 2, 2)
    assert jnp.allclose(result, 1.0)


def test_quick_linear_field() -> None:
    """QUICK on linear field gives exact face value."""
    x = jnp.arange(6.0).reshape(6, 1, 1) * jnp.ones((1, 6, 6))
    fx = jnp.ones((3, 6, 6))
    fy = jnp.zeros((6, 3, 6))
    fz = jnp.zeros((6, 6, 3))
    dh = jnp.array([1.0, 1.0, 1.0])
    result = QUICK().compute(x, [fx, fy, fz], dh)
    assert result.shape == (2, 2, 2)
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
    # VanLeer round-trip
    obj = adapter.validate_python({"type": "VanLeer"})
    assert isinstance(obj, VanLeer)


# --- compute rename ---


def test_operators_have_compute_method():
    """All spatial operators expose compute (not compute_on_patch)."""
    from blockamr.operators.div import Div
    from blockamr.operators.grad import Grad
    from blockamr.operators.laplacian import Laplacian
    from blockamr.operators.source import Source

    assert hasattr(Div, "compute")
    assert hasattr(Grad, "compute")
    assert hasattr(Laplacian, "compute")
    assert hasattr(Source, "compute")
    assert not hasattr(Div, "compute_on_patch")


# --- Laplacian schemes ---


def test_laplacian_uniform_field_3d():
    """Laplacian of uniform field with uniform gamma is zero."""
    phi = jnp.ones((4, 4, 4))
    gamma = jnp.ones((4, 4, 4))
    dh = jnp.array([1.0, 1.0, 1.0])
    result = CentralDiffLaplacian().compute(phi, gamma, dh)
    assert result.shape == (2, 2, 2)
    assert jnp.allclose(result, 0.0)


def test_laplacian_quadratic_field():
    """Laplacian of phi=x^2 with gamma=1 gives 2.0."""
    # 4 cells along x: x = 0, 1, 2, 3 → interior cells 1, 2
    # phi = x^2: 0, 1, 4, 9 → d²phi/dx² = 2.0
    x = jnp.arange(4.0)
    phi = (x**2).reshape(4, 1, 1) * jnp.ones((1, 4, 4))
    gamma = jnp.ones((4, 4, 4))
    dh = jnp.array([1.0, 1.0, 1.0])
    result = CentralDiffLaplacian().compute(phi, gamma, dh)
    assert result.shape == (2, 2, 2)
    assert jnp.allclose(result, 2.0)


def test_central_diff_laplacian_discriminator():
    """Pydantic roundtrip for CentralDiffLaplacian."""
    adapter = TypeAdapter(CentralDiffLaplacian)
    obj = adapter.validate_python({"type": "CentralDiffLaplacian"})
    assert isinstance(obj, CentralDiffLaplacian)


# --- Grad schemes ---


def test_grad_linear_field_3d():
    """Gradient of phi=x gives (1, 0, 0)."""
    # 4 cells along x: x = 0, 1, 2, 3 → interior cells 1, 2
    x = jnp.arange(4.0).reshape(4, 1, 1) * jnp.ones((1, 4, 4))
    dh = jnp.array([1.0, 1.0, 1.0])
    result = CentralDiffGrad().compute(x, dh)
    assert result.shape == (2, 2, 2, 3)
    assert jnp.allclose(result[..., 0], 1.0)  # dphi/dx = 1
    assert jnp.allclose(result[..., 1], 0.0)  # dphi/dy = 0
    assert jnp.allclose(result[..., 2], 0.0)  # dphi/dz = 0


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
