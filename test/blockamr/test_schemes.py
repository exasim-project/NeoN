# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import jax.numpy as jnp

from pydantic import TypeAdapter

from blockamr.schemes.div_schemes import QUICK, DivScheme, Linear, Upwind, VanLeer
from blockamr.schemes.laplacian_schemes import CentralDiffLaplacian, LaplacianScheme
from blockamr.schemes.grad_schemes import CentralDiffGrad, GradScheme
from blockamr.schemes.ddt_schemes import DdtScheme, ForwardEuler, RungeKutta2, RungeKutta4
from blockamr.schemes.schemes_dict import SchemesDict


def test_upwind_positive_velocity():
    """Upwind stencil with positive velocity selects left (upwind) cell value."""
    scheme = Upwind()
    phi_left = jnp.array([1.0, 2.0, 3.0])
    phi_right = jnp.array([4.0, 5.0, 6.0])
    vel_face = jnp.array([1.0, 1.0, 1.0])
    result = scheme.face_value(phi_left, phi_right, vel_face)
    assert jnp.allclose(result, phi_left)


def test_upwind_negative_velocity():
    """Upwind stencil with negative velocity selects right (downwind) cell value."""
    scheme = Upwind()
    phi_left = jnp.array([1.0, 2.0, 3.0])
    phi_right = jnp.array([4.0, 5.0, 6.0])
    vel_face = jnp.array([-1.0, -1.0, -1.0])
    result = scheme.face_value(phi_left, phi_right, vel_face)
    assert jnp.allclose(result, phi_right)


def test_linear_averages():
    """Linear stencil averages left and right cell values."""
    scheme = Linear()
    phi_left = jnp.array([1.0, 2.0, 3.0])
    phi_right = jnp.array([3.0, 4.0, 5.0])
    vel_face = jnp.array([1.0, -1.0, 0.0])
    result = scheme.face_value(phi_left, phi_right, vel_face)
    assert jnp.allclose(result, jnp.array([2.0, 3.0, 4.0]))


def test_vanleer_uniform_field():
    """VanLeer on uniform field gives that value (no gradient -> upwind)."""
    scheme = VanLeer()
    phi_far_left = jnp.array([1.0, 1.0])
    phi_left = jnp.array([1.0, 1.0])
    phi_right = jnp.array([1.0, 1.0])
    phi_far_right = jnp.array([1.0, 1.0])
    vel_face = jnp.array([1.0, -1.0])
    result = scheme.face_value(phi_far_left, phi_left, phi_right, phi_far_right, vel_face)
    assert jnp.allclose(result, jnp.array([1.0, 1.0]))


def test_vanleer_linear_field():
    """VanLeer on a linear field gives exact face value (limiter = 1 for r=1)."""
    scheme = VanLeer()
    # Linear profile: 1, 2, 3, 4 -> face between 2 and 3 should be 2.5
    phi_far_left = jnp.array([1.0])
    phi_left = jnp.array([2.0])
    phi_right = jnp.array([3.0])
    phi_far_right = jnp.array([4.0])
    vel_face = jnp.array([1.0])
    result = scheme.face_value(phi_far_left, phi_left, phi_right, phi_far_right, vel_face)
    assert jnp.allclose(result, jnp.array([2.5]), atol=1e-12)


def test_div_scheme_discriminator_roundtrip():
    """Pydantic discriminator resolves DivScheme from dict."""
    adapter = TypeAdapter(DivScheme)
    for tag, cls in [("Upwind", Upwind), ("Linear", Linear), ("VanLeer", VanLeer), ("QUICK", QUICK)]:
        obj = adapter.validate_python({"type": tag})
        assert isinstance(obj, cls)


def test_quick_linear_field():
    """QUICK on a linear field gives exact face value."""
    scheme = QUICK()
    # Linear profile: 1, 2, 3, 4 -> face between 2 and 3 should be 2.5
    phi_far_left = jnp.array([1.0])
    phi_left = jnp.array([2.0])
    phi_right = jnp.array([3.0])
    phi_far_right = jnp.array([4.0])
    vel_face = jnp.array([1.0])
    result = scheme.face_value(phi_far_left, phi_left, phi_right, phi_far_right, vel_face)
    assert jnp.allclose(result, jnp.array([2.5]), atol=1e-12)


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


def test_central_diff_laplacian():
    """CentralDiffLaplacian averages left and right gamma values."""
    scheme = CentralDiffLaplacian()
    gamma_left = jnp.array([1.0, 2.0])
    gamma_right = jnp.array([3.0, 4.0])
    result = scheme.face_value(gamma_left, gamma_right)
    assert jnp.allclose(result, jnp.array([2.0, 3.0]))


def test_laplacian_scheme_discriminator():
    """LaplacianScheme discriminator resolves from dict."""
    adapter = TypeAdapter(LaplacianScheme)
    obj = adapter.validate_python({"type": "CentralDiffLaplacian"})
    assert isinstance(obj, CentralDiffLaplacian)


# --- Grad schemes ---


def test_central_diff_grad():
    """CentralDiffGrad computes (phi_right - phi_left) / (2*dx)."""
    scheme = CentralDiffGrad()
    phi_left = jnp.array([1.0, 2.0])
    phi_right = jnp.array([3.0, 6.0])
    dx = 0.5
    result = scheme.face_value(phi_left, phi_right, dx)
    assert jnp.allclose(result, jnp.array([2.0, 4.0]))


def test_grad_scheme_discriminator():
    """GradScheme discriminator resolves from dict."""
    adapter = TypeAdapter(GradScheme)
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
