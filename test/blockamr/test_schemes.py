# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import jax.numpy as jnp

from pydantic import TypeAdapter

from neon.blockamr.schemes.div_schemes import QUICK, DivScheme, Linear, Upwind, VanLeer
from neon.blockamr.schemes.laplacian_schemes import CentralDiffLaplacian
from neon.blockamr.schemes.grad_schemes import CentralDiffGrad
from neon.blockamr.schemes.ddt_schemes import DdtScheme, ForwardEuler, RungeKutta2, RungeKutta4
from neon.blockamr.schemes.schemes_dict import SchemesDict


def test_upwind_uniform_field_3d() -> None:
    """Upwind compute on uniform field is zero (all flux in = flux out)."""
    # u: 4 cells (1 ghost each side, 2 interior). Fluxes need 2+1+2*w = 5 faces along own axis.
    u = jnp.ones((4, 4, 4))
    fx = jnp.ones((5, 4, 4))
    fy = jnp.ones((4, 5, 4))
    fz = jnp.ones((4, 4, 5))
    dh = jnp.array([1.0, 1.0, 1.0])
    result = Upwind().compute(u, [fx, fy, fz], dh)
    assert result.shape == (2, 2, 2)
    assert jnp.allclose(result, 0.0)


def test_upwind_positive_flux() -> None:
    """Upwind with positive flux selects left (upwind) cell."""
    # Along x: values 0,1,2,3 — interior cells are [1] and [2]
    # positive flux → upwind selects left neighbour
    u = jnp.arange(4.0).reshape(4, 1, 1) * jnp.ones((1, 4, 4))
    fx = jnp.ones((5, 4, 4))   # positive flux along x
    fy = jnp.zeros((4, 5, 4))  # zero flux along y
    fz = jnp.zeros((4, 4, 5))  # zero flux along z
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
    fx = jnp.ones((5, 4, 4))
    fy = jnp.ones((4, 5, 4))
    fz = jnp.ones((4, 4, 5))
    dh = jnp.array([1.0, 1.0, 1.0])
    result = Linear().compute(u, [fx, fy, fz], dh)
    assert result.shape == (2, 2, 2)
    assert jnp.allclose(result, 0.0)


def test_vanleer_uniform() -> None:
    """VanLeer on uniform field is zero."""
    # VanLeer stencil_width=2: u needs 2+2*2=6, fluxes need 2+1+2*2=7 along own axis
    u = jnp.ones((6, 6, 6))
    fx = jnp.ones((7, 6, 6))
    fy = jnp.ones((6, 7, 6))
    fz = jnp.ones((6, 6, 7))
    dh = jnp.array([1.0, 1.0, 1.0])
    result = VanLeer().compute(u, [fx, fy, fz], dh)
    assert result.shape == (2, 2, 2)
    assert jnp.allclose(result, 0.0)


def test_vanleer_linear_field() -> None:
    """VanLeer on linear field gives exact face value (limiter=1 for r=1)."""
    x = jnp.arange(6.0).reshape(6, 1, 1) * jnp.ones((1, 6, 6))
    fx = jnp.ones((7, 6, 6))
    fy = jnp.zeros((6, 7, 6))
    fz = jnp.zeros((6, 6, 7))
    dh = jnp.array([1.0, 1.0, 1.0])
    result = VanLeer().compute(x, [fx, fy, fz], dh)
    assert result.shape == (2, 2, 2)
    assert jnp.allclose(result, 1.0)


def test_quick_linear_field() -> None:
    """QUICK on linear field gives exact face value."""
    x = jnp.arange(6.0).reshape(6, 1, 1) * jnp.ones((1, 6, 6))
    fx = jnp.ones((7, 6, 6))
    fy = jnp.zeros((6, 7, 6))
    fz = jnp.zeros((6, 6, 7))
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


def test_operators_have_build_kernel_method():
    """All spatial operators expose build_kernel."""
    from neon.blockamr.operators.div import Div
    from neon.blockamr.operators.grad import Grad
    from neon.blockamr.operators.laplacian import Laplacian
    from neon.blockamr.operators.source import Source

    assert hasattr(Div, "build_kernel")
    assert hasattr(Grad, "build_kernel")
    assert hasattr(Laplacian, "build_kernel")
    assert hasattr(Source, "build_kernel")


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
