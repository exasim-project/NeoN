# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Performance tests: tiled Pallas DSL dispatch vs C++.

Tests each scheme through the DSL (evaluate/solve) and compares against
C++ baselines where available. Verifies correctness and performance.
"""

import math
import time

import jax
import jax.numpy as jnp
import pytest

import blockamr
from blockamr.mesh import Mesh
from blockamr.field import CellField, FaceField
from blockamr.operators.div import Div, update_face_fluxes
from blockamr.dsl import exp
from blockamr.dsl.solve import solve, evaluate
from blockamr.schemes.div_schemes import Upwind, Linear, VanLeer, QUICK


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mesh(N, max_size=None):
    if max_size is None:
        max_size = N
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom), geom


def _init_sin3d(phi_mf, geom):
    dx = geom.cell_size()
    for mfi in blockamr.MFIterator(phi_mf):
        bx = mfi.valid_box()
        lo = bx.small_end()
        arr = phi_mf.copy_to_host(mfi)
        nx, ny, nz = arr.shape[:3]
        xs = (jnp.arange(nx) + lo[0] + 0.5) * dx[0]
        ys = (jnp.arange(ny) + lo[1] + 0.5) * dx[1]
        zs = (jnp.arange(nz) + lo[2] + 0.5) * dx[2]
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
        arr[:, :, :, 0] = (
            jnp.sin(2 * math.pi * X) * jnp.sin(2 * math.pi * Y) * jnp.sin(2 * math.pi * Z)
        )
        phi_mf.copy_from(mfi, arr)
    phi_mf.fill_boundary(geom)


def _time_fn(fn, n_warmup=5, n_iter=50):
    """Warmup + time a function, return ms per call."""
    for _ in range(n_warmup):
        fn()
    jax.block_until_ready(None)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    jax.block_until_ready(None)
    return (time.perf_counter() - t0) / n_iter * 1000


def _setup_advdiff(N, ngrow):
    """Create mesh + fields for advection-diffusion test."""
    mesh, geom = _make_mesh(N)
    phi = CellField(mesh, ncomp=1, ngrow=ngrow, name="phi")
    ff = FaceField(mesh, ncomp=1, ngrow=0, name="U")
    _init_sin3d(phi.mf[0], geom)

    def vel(x, y, z, t):
        return (jnp.ones_like(x), jnp.ones_like(x), jnp.ones_like(x))

    update_face_fluxes(ff[0], vel, geom, 0.0)

    return mesh, geom, phi, ff


# ---------------------------------------------------------------------------
# Laplacian (source term)
# ---------------------------------------------------------------------------


def test_evaluate_laplacian_vs_cpp(blockamr_session):
    """DSL evaluate(laplacian) matches C++ and measures performance."""
    N = 128
    mesh, geom = _make_mesh(N)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    _init_sin3d(phi.mf[0], geom)

    lap_expr = exp.laplacian(1.0, phi)
    result = evaluate(lap_expr, t=0.0)

    # C++ reference
    out = CellField(mesh, ncomp=1, ngrow=0, name="out")
    blockamr.laplacian(out.mf[0], phi.mf[0], geom)

    # Correctness
    cpp_arrs = out.mf[0].arrays()
    max_err = 0.0
    for pallas_box, cpp_box in zip(result[0], cpp_arrs):
        p = pallas_box.squeeze()
        c = cpp_box[:, :, :, 0] if cpp_box.ndim == 4 else cpp_box
        err = float(jnp.max(jnp.abs(p - c)))
        max_err = max(max_err, err)

    print(f"\nLaplacian {N}^3: max_err vs C++ = {max_err:.2e}")
    assert max_err < 1e-6, f"Laplacian error too large: {max_err}"

    # Performance
    pallas_ms = _time_fn(lambda: evaluate(lap_expr, t=0.0))
    cpp_ms = _time_fn(lambda: blockamr.laplacian(out.mf[0], phi.mf[0], geom))

    print(f"  Pallas: {pallas_ms:.3f} ms, C++: {cpp_ms:.3f} ms, ratio: {pallas_ms / cpp_ms:.2f}x")


# ---------------------------------------------------------------------------
# Per-scheme evaluate (source term, no time step)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", [Upwind(), Linear(), VanLeer(), QUICK()])
def test_evaluate_div_scheme(blockamr_session, scheme):
    """DSL evaluate(div(scheme)) produces non-zero source for each scheme."""
    ngrow = scheme.stencil_width
    N = 64
    mesh, geom, phi, ff = _setup_advdiff(N, ngrow)

    div_expr = Div(ff, phi, scheme=scheme)
    result = evaluate(div_expr, t=0.0)

    max_val = max(float(jnp.max(jnp.abs(box.squeeze()))) for box in result[0])
    print(f"\n{scheme.type} div {N}^3: max|source| = {max_val:.6f}")
    assert max_val > 1e-6, f"{scheme.type} div result near-zero: {max_val}"

    pallas_ms = _time_fn(lambda: evaluate(div_expr, t=0.0))
    print(f"  Pallas: {pallas_ms:.3f} ms")


# ---------------------------------------------------------------------------
# Per-scheme solve (forward Euler step) vs C++ where available
# ---------------------------------------------------------------------------


def _solve_one_step(phi, ff, geom, scheme, nu, dt):
    """Run one DSL forward Euler step."""
    _init_sin3d(phi.mf[0], geom)
    expr = exp.ddt(phi) + Div(ff, phi, scheme=scheme) - exp.laplacian(nu, phi)
    solve(expr, t=0.0, dt=dt)


# ---------------------------------------------------------------------------
# Per-operator evaluate (source term only, no fusing)
# ---------------------------------------------------------------------------


def test_evaluate_laplacian_perf(blockamr_session):
    """Laplacian source term alone: DSL evaluate vs C++."""
    N = 64
    mesh, geom = _make_mesh(N)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    out = CellField(mesh, ncomp=1, ngrow=0, name="out")
    _init_sin3d(phi.mf[0], geom)

    lap_expr = exp.laplacian(1.0, phi)
    evaluate(lap_expr, t=0.0)  # warmup
    jax.block_until_ready(None)

    pallas_ms = _time_fn(lambda: evaluate(lap_expr, t=0.0))
    cpp_ms = _time_fn(lambda: blockamr.laplacian(out.mf[0], phi.mf[0], geom))

    print(f"\nLaplacian only {N}^3:")
    print(f"  Pallas: {pallas_ms:.3f} ms, C++: {cpp_ms:.3f} ms, ratio: {pallas_ms / cpp_ms:.2f}x")


CPP_DIV_BASELINES = {
    "Upwind": "div_upwind",
    "Linear": "div_linear",
    "VanLeer": "div_vanleer",
    "QUICK": "div_quick",
}


@pytest.mark.parametrize("scheme", [Upwind(), Linear(), VanLeer(), QUICK()])
def test_evaluate_div_perf(blockamr_session, scheme):
    """Div source term alone: DSL evaluate vs C++ per scheme."""
    N = 64
    ngrow = scheme.stencil_width
    mesh, geom, phi, ff = _setup_advdiff(N, ngrow)
    out = CellField(mesh, ncomp=1, ngrow=0, name="out")

    div_expr = Div(ff, phi, scheme=scheme)
    evaluate(div_expr, t=0.0)  # warmup

    cpp_fn = getattr(blockamr, CPP_DIV_BASELINES[scheme.type])
    cpp_fn(out.mf[0], phi.mf[0], ff[0][0].mf, ff[0][1].mf, ff[0][2].mf, geom)  # warmup
    jax.block_until_ready(None)

    pallas_ms = _time_fn(lambda: evaluate(div_expr, t=0.0))
    cpp_ms = _time_fn(
        lambda: cpp_fn(out.mf[0], phi.mf[0], ff[0][0].mf, ff[0][1].mf, ff[0][2].mf, geom)
    )

    print(f"\n{scheme.type} div only {N}^3:")
    print(f"  Pallas: {pallas_ms:.3f} ms, C++: {cpp_ms:.3f} ms, ratio: {pallas_ms / cpp_ms:.2f}x")


# ---------------------------------------------------------------------------
# Per-scheme solve (fused div + lap + Euler) vs C++
# ---------------------------------------------------------------------------

CPP_BASELINES = {
    "Upwind": "euler_step_upwind_lap",
    "Linear": "euler_step_linear_lap",
    "VanLeer": "euler_step_vanleer_lap",
    "QUICK": "euler_step_quick_lap",
}


@pytest.mark.parametrize("scheme", [Upwind(), Linear(), VanLeer(), QUICK()])
def test_solve_scheme_vs_cpp(blockamr_session, scheme):
    """DSL solve with each scheme vs C++ baseline: correctness + performance."""
    N = 64
    nu = 0.001
    dt = 0.25 / N
    ngrow = scheme.stencil_width
    mesh, geom, phi, ff = _setup_advdiff(N, ngrow)
    phi_cpp = CellField(mesh, ncomp=1, ngrow=ngrow, name="phi_cpp")

    def dsl_step():
        _solve_one_step(phi, ff, geom, scheme, nu, dt)

    cpp_fn_name = CPP_BASELINES[scheme.type]
    cpp_fn = getattr(blockamr, cpp_fn_name)

    def cpp_step():
        _init_sin3d(phi_cpp.mf[0], geom)
        cpp_fn(phi_cpp.mf[0], ff[0][0].mf, ff[0][1].mf, ff[0][2].mf, geom, dt, nu, 1)

    # Warmup
    dsl_step()
    cpp_step()
    jax.block_until_ready(None)

    # Performance
    pallas_ms = _time_fn(dsl_step)
    cpp_ms = _time_fn(cpp_step)
    ratio = pallas_ms / cpp_ms

    print(f"\n{scheme.type}+Lap solve {N}^3:")
    print(f"  Pallas: {pallas_ms:.3f} ms, C++: {cpp_ms:.3f} ms, ratio: {ratio:.2f}x")
    assert ratio < 10.0, f"{scheme.type}+Lap {ratio:.1f}x C++ — expected < 10x"

    # Correctness: result should be bounded and non-NaN
    for arr in phi.mf[0].arrays():
        vals = arr[:, :, :, 0]
        assert not bool(jnp.any(jnp.isnan(vals))), f"{scheme.type}: NaN in DSL result"
        max_val = float(jnp.max(jnp.abs(vals)))
        assert max_val < 10.0, f"{scheme.type}: result unbounded (max={max_val})"
