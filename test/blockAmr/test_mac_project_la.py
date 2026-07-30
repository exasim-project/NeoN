# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""``mac_project(backend="la")`` against the MLABecLaplacian/MLMG route it replaced.

The MAC projection runs every timestep, so substituting ``MFFaceCoeffs`` plus
``linear_algebra.laplacian`` for ``MLABecLaplacian`` at alpha=0/beta=1/b=1 has to be
shown to be the SAME discrete operator rather than merely a plausible one. Both routes
stay reachable through ``backend=``, so every comparison below runs them on the same
input in one process.

What is checked, on a periodic single box, a periodic MULTI-box layout, an
outflow-Dirichlet configuration and an all-Neumann one:

* the SIGN carries over untouched. ``laplacian`` writes each face coefficient as
  ``-gamma/dx**2``, so a system of that term alone is ``-div(grad p)`` -- the same sign
  as MLABecLaplacian at alpha=0/beta=1 -- and the rhs stays ``-div(phi)``. Nothing here
  flips a sign, and the two routes agreeing on ``p_mac`` is what says none was needed;
* NO diagonal source is written: alpha=0 has none, so ``Matrix.diagonal_source`` is
  never called (``test_la_boundary_conditions.py`` pins the same statement at
  coefficient level);
* the six-element BC list is ORDERED (xlo, xhi, ylo, yhi, zlo, zhi) as ``la::parseBc``
  reads it, unit-tested against ``pressure_bc``'s per-AXIS pair -- the one mistake in
  this substitution that would otherwise be silent;
* the face-gradient kernel reproduces ``MLMG.get_fluxes`` BITWISE off the same
  ``p_mac``: the same staggering, the same sign;
* the projection still projects -- ``max|div(phi)|`` collapses to the solve tolerance,
  measured exactly as ``test_verification_projection.py::_max_face_divergence``
  measures it, so the number is comparable to the shipped gate's.

MAX_ORDER is the one place the two routes are NOT bitwise. At a DIRICHLET domain face
AMReX defaults to a third-order ghost extrapolation (``MLLinOp`` maxorder 3) while the
linear-algebra layer reflects the boundary cell, which is AMReX's ``max_order=2``; both
are second-order accurate. The agreement rows therefore put the oracle on
``max_order=2``, and ``test_default_amrex_max_order_moves_the_dirichlet_answer`` is the
anti-vacuity check that says doing so is load-bearing rather than cosmetic.
``max_order`` does not enter a periodic or a Neumann side at all, so the other three
configurations are unaffected by it and are held to the same tolerance.
"""

import numpy as np
import pytest

import blockamr
from blockamr.field import FaceField
from blockamr.mesh import Mesh
from blockamr.operators import mac_project as mac

N = 16
_SOL_P = {"rtol": 1e-12, "atol": 1e-14, "maxIter": 400}

# A divergence-free constant per direction, added to the input so the PROJECTED flux is
# O(1) rather than round-off: a pure gradient projects to nothing, which would turn
# every relative comparison of the corrected field into a comparison of two noise floors.
_UNIFORM_FLUX = (100.0, -50.0, 25.0)

_BT = blockamr.LinOpBCType

# The non-trivial mapping: a pressure Dirichlet on ONE side only, paired with the
# velocity Neumann of an outlet, on an otherwise periodic/Neumann domain. This is the
# `cylinder_re20` shape, and the only configuration where the two routes' Dirichlet
# closures differ.
_OUTFLOW_PERIODICITY = [0, 1, 1]
_OUTFLOW_BC = (
    [_BT.Neumann, _BT.Periodic, _BT.Periodic],
    [_BT.Dirichlet, _BT.Periodic, _BT.Periodic],
)

# (name, periodicity, pressure_bc, max_size). `pressure_bc` None is the
# periodic/all-Neumann default both routes build for themselves when the field has none;
# `max_size` below N gives the MULTI-box layout.
_CASES = [
    ("periodic", [1, 1, 1], None, N),
    ("periodic_multibox", [1, 1, 1], None, N // 2),
    ("outflow_dirichlet", _OUTFLOW_PERIODICITY, _OUTFLOW_BC, N),
    ("all_neumann", [0, 0, 0], ([_BT.Neumann] * 3, [_BT.Neumann] * 3), N),
]

# Two multigrid-preconditioned solves of one operator, so not bitwise. Measured worst
# case across every row below is ~1e-11 relative, i.e. ~100x of headroom.
_AGREE_TOL = 1e-9


def _make_mesh(periodic, max_size):
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, periodic)
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom)


def _global_face_flux(geom, bc, seed):
    """``-grad_f(s) + uniform`` for random ``s``, on the GLOBAL grid, built in numpy.

    Independent of the code under test, and solvability-CONSISTENT under every BC in
    ``_CASES``: a per-face random field is not, because on a periodic axis the low and
    high domain faces are one physical face and must carry one value, and on a Neumann
    wall the net normal flux must vanish. Without that the singular operator cannot
    solve its own rhs and the two routes would be compared on two differently-modified
    problems. A face gradient satisfies both by construction.
    """
    dx = geom.cell_size()
    g = np.zeros((N + 2, N + 2, N + 2))
    g[1:-1, 1:-1, 1:-1] = np.random.default_rng(seed).standard_normal((N, N, N))

    # One ghost layer per side, filled as both routes fill it: periodic wraparound,
    # Dirichlet reflect-odd, Neumann reflect-even.
    for d in range(3):
        for side in (0, 1):
            kind = bc[2 * d + side]
            low = side == 0
            dst = [slice(1, N + 1)] * 3
            dst[d] = 0 if low else N + 1
            src = [slice(1, N + 1)] * 3
            if kind == "periodic":
                src[d] = N if low else 1
            else:
                src[d] = 1 if low else N
            g[tuple(dst)] = (-1.0 if kind == "dirichlet" else 1.0) * g[tuple(src)]

    faces = []
    for d in range(3):
        sl_hi = [slice(1, N + 1) for _ in range(3)]
        sl_lo = list(sl_hi)
        sl_hi[d] = slice(1, N + 2)
        sl_lo[d] = slice(0, N + 1)
        faces.append(-(g[tuple(sl_hi)] - g[tuple(sl_lo)]) / dx[d] + _UNIFORM_FLUX[d])
    return faces


def _face_field(mesh, p_bc, bc, seed=3):
    """A ``FaceField`` carrying ``_global_face_flux``, scattered per box."""
    geom = mesh.geom(0)
    faces = _global_face_flux(geom, bc, seed)
    phi = FaceField(mesh, ncomp=1, ngrow=1, name="phi")
    for d in range(3):
        mf = phi[0][d].mf
        mf.set_val(0.0)
        for mfi in blockamr.MFIterator(mf):
            bx = mfi.valid_box()
            lo, hi = bx.small_end(), bx.big_end()
            arr = mf.copy_to_host(mfi)
            arr[:, :, :, 0] = faces[d][tuple(slice(lo[ax], hi[ax] + 1) for ax in range(3))]
            mf.copy_from(mfi, arr)
        mf.fill_boundary(geom)
    phi.pressure_bc = p_bc
    return phi


def _max_face_divergence(phi, mesh):
    """max|div(phi)| from the MAC face flux.

    Deliberately the same measure as ``test_verification_projection.py``'s, so the
    numbers here are comparable to the shipped divergence-free gate's.
    """
    dx = mesh.geom(0).cell_size()
    max_abs = 0.0
    face_arrs = [phi[0][d].mf.arrays() for d in range(3)]
    for bi in range(len(face_arrs[0])):
        div_val = None
        for d in range(3):
            f = face_arrs[d][bi][:, :, :, 0]
            ng = phi[0][d].mf.n_grow()
            nc = [int(f.shape[ax]) - 2 * ng - (1 if ax == d else 0) for ax in range(3)]
            sl_hi = [slice(ng, ng + nc[ax]) for ax in range(3)]
            sl_lo = [slice(ng, ng + nc[ax]) for ax in range(3)]
            sl_hi[d] = slice(ng + 1, ng + 1 + nc[d])
            sl_lo[d] = slice(ng, ng + nc[d])
            contrib = (f[tuple(sl_hi)] - f[tuple(sl_lo)]) / dx[d]
            div_val = contrib if div_val is None else div_val + contrib
        max_abs = max(max_abs, float(np.max(np.abs(div_val))))
    return max_abs


def _valid(mf):
    """Per-box valid-region (i, j, k) arrays, ghosts and component dropped."""
    ng = mf.n_grow()
    out = []
    for arr in mf.arrays():
        v = np.asarray(arr[:, :, :, 0])
        out.append(v[ng:-ng, ng:-ng, ng:-ng] if ng else v)
    return out


def _flat_faces(phi):
    return np.concatenate([v.ravel() for d in range(3) for v in _valid(phi[0][d].mf)])


def _peak(x):
    return float(np.max(np.abs(np.asarray(x))))


def _mean_free(mf):
    """The valid ``p_mac``, mean removed.

    A periodic or all-Neumann pressure Poisson operator is SINGULAR, so its solution is
    only defined up to a constant: MLMG returns whatever representative its cycle lands
    on, the Krylov route returns the mean-zero one (``project_nullspace``). The
    projection reads only the GRADIENT, so the constant never reaches the answer -- and
    comparing the raw fields would flag a difference the physics cannot see.
    """
    v = np.concatenate([x.ravel() for x in _valid(mf)])
    return v - v.mean()


def _project(mesh, p_bc, backend, max_order=None):
    """Project a fresh copy of the identical input through one backend."""
    bc = mac._la_bc_from_pressure_bc(p_bc, mesh.geom(0))
    phi = _face_field(mesh, p_bc, bc)
    scale = _peak(_flat_faces(phi))
    div_before = _max_face_divergence(phi, mesh)
    if max_order is not None:
        # Reaches the CACHED operator, so the production route is untouched. MLLinOp
        # reads maxorder when it fills ghosts, not at setup, so setting it after
        # set_level_bc/set_b_coeffs is equivalent to setting it before.
        mac._ensure_mac_cache(phi, 0).lp.set_max_order(max_order)
    mac.mac_project(phi, _SOL_P, backend=backend)
    p_mf = phi._mac_la_cache.p_mf if backend == "la" else phi._mac_cache.phi_mf
    return phi, p_mf, scale, div_before


@pytest.mark.parametrize("case, periodic, p_bc, max_size", _CASES)
def test_the_two_backends_agree_on_p_mac_and_on_the_corrected_flux(
    blockamr_session, case, periodic, p_bc, max_size
):
    """One input, two routes, one answer -- to the solve tolerance.

    The load-bearing row. ``p_mac`` is compared mean-free (the singular configurations
    have no preferred constant) and the corrected face flux directly. A sign error in
    the rhs, a diagonal source that should not be there or a permuted BC list all
    land here as a gross disagreement rather than as a slow drift.
    """
    mesh = _make_mesh(periodic, max_size)
    phi_ref, p_ref, scale, div_before = _project(mesh, p_bc, "mlmg", max_order=2)
    phi_la, p_la, scale_la, div_before_la = _project(mesh, p_bc, "la")

    assert scale_la == scale, f"{case}: the two runs did not get the same input"
    assert div_before_la == div_before

    p_diff = _peak(_mean_free(p_la) - _mean_free(p_ref)) / _peak(_mean_free(p_ref))
    assert p_diff < _AGREE_TOL, f"{case}: |p_mac(la) - p_mac(mlmg)| = {p_diff:.3e} (relative)"

    flux_diff = _peak(_flat_faces(phi_la) - _flat_faces(phi_ref)) / scale
    assert flux_diff < _AGREE_TOL, (
        f"{case}: |phi(la) - phi(mlmg)| = {flux_diff:.3e} of the input flux peak {scale:.3e}"
    )


@pytest.mark.parametrize("case, periodic, p_bc, max_size", _CASES)
@pytest.mark.parametrize("backend", ["la", "mlmg"])
def test_the_projection_leaves_the_face_flux_divergence_free(
    blockamr_session, case, periodic, p_bc, max_size, backend
):
    """``max|div(phi)|`` after the projection is at the solve tolerance, either route.

    This is the property the shipped gate measures, and the one the substitution must
    not lose: the MAC pattern makes it exact by construction because the correction is
    the adjoint of the divergence it solved against, so the residual left here is the
    solve's, not the discretisation's.
    """
    mesh = _make_mesh(periodic, max_size)
    phi, _, _, div_before = _project(mesh, p_bc, backend)

    div_after = _max_face_divergence(phi, mesh)
    assert div_after / div_before < _AGREE_TOL, (
        f"{case}/{backend}: max|div phi| went {div_before:.3e} -> {div_after:.3e}"
    )


@pytest.mark.parametrize("case, periodic, p_bc, max_size", _CASES)
def test_the_face_gradient_kernel_reproduces_get_fluxes_bitwise(
    blockamr_session, case, periodic, p_bc, max_size
):
    """Off the SAME ``p_mac``, the kernel is ``MLMG.get_fluxes`` to the last bit.

    Fed one pressure field, the two gradients can only differ by staggering, by sign or
    by which ghost the domain face reads -- so bitwise equality is the right claim and a
    tolerance would hide exactly the mistakes worth catching. A DIRICHLET side is
    excluded and only there: that is the third-order ghost extrapolation of the module
    docstring, not a staggering difference, and the interior faces of the same direction
    are still held to bitwise.
    """
    mesh = _make_mesh(periodic, max_size)
    bc = mac._la_bc_from_pressure_bc(p_bc, mesh.geom(0))
    phi, p_mf, _, _ = _project(mesh, p_bc, "mlmg")
    cache = phi._mac_cache

    want = [
        [np.asarray(a[:, :, :, 0]) for a in f.arrays()]
        for f in (cache.flux_x, cache.flux_y, cache.flux_z)
    ]
    got = mac._face_gradient_flux(p_mf, mesh.geom(0), bc)

    for d in range(3):
        keep = [slice(None)] * 3
        keep[d] = slice(
            1 if bc[2 * d] == "dirichlet" else 0,
            -1 if bc[2 * d + 1] == "dirichlet" else None,
        )
        for bi, (w, g) in enumerate(zip(want[d], got[d])):
            np.testing.assert_array_equal(
                np.asarray(g)[tuple(keep)],
                w[tuple(keep)],
                err_msg=f"{case}: direction {d}, box {bi}",
            )


def test_default_amrex_max_order_moves_the_dirichlet_answer(blockamr_session):
    """The anti-vacuity check for pinning the oracle at ``max_order=2``.

    Every agreement row above forces the MLMG oracle onto AMReX's second-order
    Dirichlet closure, which is the linear-algebra layer's. If that made no difference
    the rows would be passing for a reason unrelated to the closure, and a real
    regression in the layer's boundary handling could hide behind it. At AMReX's DEFAULT
    ``max_order=3`` the same comparison must therefore FAIL, and by much more than the
    agreement bar. Measured ~1.4e-1 relative on ``p_mac``.
    """
    mesh = _make_mesh(_OUTFLOW_PERIODICITY, N)

    _, p_default, _, _ = _project(mesh, _OUTFLOW_BC, "mlmg")
    _, p_la, _, _ = _project(mesh, _OUTFLOW_BC, "la")

    diff = _peak(_mean_free(p_la) - _mean_free(p_default)) / _peak(_mean_free(p_default))
    assert diff > 1e-3, (
        f"max_order=3 and the layer's reflection agree to {diff:.3e}: forcing max_order=2 "
        "in the rows above is not load-bearing, so re-examine what they prove"
    )


def test_pressure_bc_maps_onto_the_six_sides_in_parseBc_order(blockamr_session):
    """``(lo_bc, hi_bc)`` per AXIS -> (xlo, xhi, ylo, yhi, zlo, zhi).

    ``pressure_bc`` is indexed by axis and the layer's list by SIDE, so the mapping is a
    transpose and the ordering is the one thing here that fails silently: every wrong
    permutation still produces six legal strings, and ``la::parseBc`` only objects when a
    side disagrees with the geometry's periodicity. Every side is given a distinguishable
    kind so no permutation survives.
    """
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    mixed = blockamr.Geometry(box, rb, 0, [0, 0, 1])

    assert mac._la_bc_from_pressure_bc(
        ([_BT.Dirichlet, _BT.Neumann, _BT.Periodic], [_BT.Neumann, _BT.Dirichlet, _BT.Periodic]),
        mixed,
    ) == ["dirichlet", "neumann", "neumann", "dirichlet", "periodic", "periodic"]

    # No pressure_bc: periodic where the geometry is, Neumann elsewhere -- the default
    # the MLMG route builds for itself, so the two routes still see one problem.
    assert mac._la_bc_from_pressure_bc(None, mixed) == [
        "neumann",
        "neumann",
        "neumann",
        "neumann",
        "periodic",
        "periodic",
    ]
    assert (
        mac._la_bc_from_pressure_bc(None, blockamr.Geometry(box, rb, 0, [1, 1, 1]))
        == ["periodic"] * 6
    )

    # A LinOpBCType the layer has no coefficient convention for is refused, not mapped
    # onto the nearest one it does have.
    with pytest.raises(ValueError, match="does not model"):
        mac._la_bc_from_pressure_bc(([_BT.Robin] * 3, [_BT.Neumann] * 3), mixed)


def test_an_unknown_backend_is_refused(blockamr_session):
    """A typo must not silently skip the projection and leave phi divergent."""
    mesh = _make_mesh([1, 1, 1], N)
    phi = _face_field(mesh, None, ["periodic"] * 6)
    with pytest.raises(ValueError, match="Unknown mac_project backend"):
        mac.mac_project(phi, _SOL_P, backend="mlgm")
