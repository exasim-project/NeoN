# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""``face_gradient.cell_gradient``: a cell-centred ``grad(p)`` from FACE differences.

The piece that lets a cell-centred pressure solve feed ``correct(U, -dt*exp.grad(p))``
with no DSL change: it produces exactly what ``dsl/solve.py`` stores on
``p_field.grad`` today, per-box ``(nx, ny, nz, 3)`` cell-centred arrays.

What is checked:

* EXACTNESS on a linear field, in the domain interior, under every BC kind and across
  INTERNAL box seams. A gradient that cannot reproduce a constant is wrong for a
  reason no convergence study localises, and running it under all four BC
  configurations is also the statement that the boundary closure touches the
  outermost cell layer and nothing else;
* SECOND ORDER on a smooth periodic field, where no cell is a boundary cell so the
  measured rate is the interpolation's alone;
* BITWISE agreement, off a real ``MLABecLaplacian``/MLMG solve, with
  ``get_fluxes``-averaged-to-cells. This is what pins the convention -- staggering,
  sign, and which ghost each domain face reads -- to the implementation the shipped
  projection already trusts. The MLMG oracle is forced onto ``max_order=2``, the
  linear-algebra layer's Dirichlet closure (see ``test_mac_project_la.py``);
* the output is consumed by ``PressureGradient`` and ``correct`` UNCHANGED.

The boundary closure is the MATRIX's, not the field's: ``_face_gradient_flux`` fills
the ghost layer with periodic wraparound, Neumann ghost = interior (wall face gradient
exactly zero) and Dirichlet ghost = -interior (wall face gradient ``-2p/dx``, the
one-sided difference to a zero value ON the face). That is what keeps the face
gradient the exact adjoint of the face divergence the pressure was solved against.
"""

import numpy as np
import pytest

import blockamr
from blockamr.dsl import exp
from blockamr.dsl.exp import PressureGradient
from blockamr.field import CellField
from blockamr.mesh import Mesh
from blockamr.operators.correct import correct
from blockamr.operators.face_gradient import cell_gradient

N = 16
_RTOL_SOLVE = 1e-12
_ATOL_SOLVE = 1e-14

_BT = blockamr.LinOpBCType

# The `cylinder_re20` shape: a pressure Dirichlet on one side only.
_OUTFLOW_PERIODICITY = [0, 1, 1]
_OUTFLOW_BC = (
    [_BT.Neumann, _BT.Periodic, _BT.Periodic],
    [_BT.Dirichlet, _BT.Periodic, _BT.Periodic],
)

# (name, periodicity, (lo_bc, hi_bc), max_size). `max_size` below N is the MULTI-box
# layout, whose internal seams are filled by FillBoundary rather than by a BC.
_CASES = [
    ("periodic", [1, 1, 1], ([_BT.Periodic] * 3, [_BT.Periodic] * 3), N),
    ("periodic_multibox", [1, 1, 1], ([_BT.Periodic] * 3, [_BT.Periodic] * 3), N // 2),
    ("all_neumann", [0, 0, 0], ([_BT.Neumann] * 3, [_BT.Neumann] * 3), N),
    ("all_neumann_multibox", [0, 0, 0], ([_BT.Neumann] * 3, [_BT.Neumann] * 3), N // 2),
    ("outflow_dirichlet", _OUTFLOW_PERIODICITY, _OUTFLOW_BC, N),
]

_LA_NAMES = {_BT.Periodic: "periodic", _BT.Dirichlet: "dirichlet", _BT.Neumann: "neumann"}


def _bc_list(p_bc):
    """``(lo_bc, hi_bc)`` per AXIS -> the layer's per-SIDE (xlo, xhi, ... ) strings."""
    return [_LA_NAMES[p_bc[half][d]] for d in range(3) for half in (0, 1)]


def _make_mesh(n, periodic, max_size):
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, periodic)
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom), geom


def _fill(mf, geom, func):
    """Write ``func(x, y, z)`` into the VALID region of a cell-centred MultiFab."""
    dx = geom.cell_size()
    prob_lo = geom.prob_lo()
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        lo, hi = bx.small_end(), bx.big_end()
        coords = [
            prob_lo[ax] + (lo[ax] + np.arange(hi[ax] - lo[ax] + 1) + 0.5) * dx[ax]
            for ax in range(3)
        ]
        x, y, z = np.meshgrid(*coords, indexing="ij")
        arr = mf.copy_to_host(mfi)
        arr[:, :, :, 0] = func(x, y, z)
        mf.copy_from(mfi, arr)


def _gather(mf, per_box, n, ncomp=3):
    """Per-box valid-region arrays -> one global ``(n, n, n, ncomp)`` numpy array.

    The multi-box rows would otherwise only ever be checked box-locally, and it is the
    box SEAM that a ghost-fill mistake shows up on.
    """
    out = np.full((n, n, n, ncomp), np.nan)
    for bi, mfi in enumerate(blockamr.MFIterator(mf)):
        lo, hi = mfi.valid_box().small_end(), mfi.valid_box().big_end()
        sl = tuple(slice(lo[ax], hi[ax] + 1) for ax in range(3))
        out[sl] = np.asarray(per_box[bi]).reshape(out[sl].shape)
    assert not np.isnan(out).any(), "the per-box arrays did not tile the domain"
    return out


def _cell_coords(n, geom):
    dx = geom.cell_size()
    prob_lo = geom.prob_lo()
    coords = [prob_lo[ax] + (np.arange(n) + 0.5) * dx[ax] for ax in range(3)]
    return np.meshgrid(*coords, indexing="ij")


def _pressure_mf(mesh, geom, n, func):
    mf = blockamr.MultiFab(mesh.box_array(0), mesh.dm(0), 1, 1)
    mf.set_val(0.0)
    _fill(mf, geom, func)
    return mf


@pytest.mark.parametrize("case, periodic, p_bc, max_size", _CASES)
def test_the_cell_gradient_is_exact_on_a_linear_field(
    blockamr_session, case, periodic, p_bc, max_size
):
    """``p = a x + b y + c z`` recovers ``(a, b, c)`` to round-off in the interior.

    The outermost cell layer is excluded because a linear field does not SATISFY any
    of these boundary conditions -- its ghost is not the reflection the matrix
    applies -- so the closure is being deliberately left out of this row rather than
    accommodated. Everything inside it, including every internal box seam of the
    multibox rows, is held to round-off.
    """
    a, b, c = 2.0, -3.0, 0.5
    mesh, geom = _make_mesh(N, periodic, max_size)
    p_mf = _pressure_mf(mesh, geom, N, lambda x, y, z: a * x + b * y + c * z)

    got = _gather(p_mf, cell_gradient(p_mf, geom, _bc_list(p_bc)), N)

    interior = got[1:-1, 1:-1, 1:-1, :]
    err = np.max(np.abs(interior - np.array([a, b, c])))
    assert err < 1e-12, f"{case}: max|grad - (a,b,c)| = {err:.3e} in the interior"


def _l2_gradient_error(n):
    """L2 error of the cell gradient of ``sin(kx)sin(ky)sin(kz)`` against the analytic one.

    Periodic, so every cell is an interior cell and the rate measured is the
    face-to-cell interpolation's own rather than a boundary closure's.
    """
    mesh, geom = _make_mesh(n, [1, 1, 1], n)
    k = 2.0 * np.pi

    p_mf = _pressure_mf(
        mesh, geom, n, lambda x, y, z: np.sin(k * x) * np.sin(k * y) * np.sin(k * z)
    )
    got = _gather(p_mf, cell_gradient(p_mf, geom, ["periodic"] * 6), n)

    x, y, z = _cell_coords(n, geom)
    want = np.stack(
        [
            k * np.cos(k * x) * np.sin(k * y) * np.sin(k * z),
            k * np.sin(k * x) * np.cos(k * y) * np.sin(k * z),
            k * np.sin(k * x) * np.sin(k * y) * np.cos(k * z),
        ],
        axis=-1,
    )
    return float(np.sqrt(np.mean((got - want) ** 2)))


def test_the_cell_gradient_is_second_order_on_a_smooth_field(blockamr_session):
    """The refinement RATES across N in {16, 32, 64} -- one error value proves nothing."""
    ns = [16, 32, 64]
    errors = [_l2_gradient_error(n) for n in ns]
    orders = [np.log2(coarse / fine) for coarse, fine in zip(errors[:-1], errors[1:])]

    detail = ", ".join(f"{c}->{f}: {o:.4f}" for c, f, o in zip(ns[:-1], ns[1:], orders))
    print("\nL2 errors " + ", ".join(f"N={n}: {e:.6e}" for n, e in zip(ns, errors)))
    print(f"observed orders {detail}")
    assert min(orders) > 1.9, f"observed orders {detail}; L2 errors {errors}"


def _solved_pressure_and_fluxes(mesh, p_bc):
    """A real ``MLABecLaplacian``/MLMG solve, plus its ``get_fluxes`` output.

    alpha=0 / beta=1 / b=1, i.e. ``-div(grad p)`` -- the operator
    ``linear_algebra.laplacian`` assembles, and the one the MAC projection solves.
    ``max_order=2`` is AMReX's second-order Dirichlet closure, which is the
    linear-algebra layer's; at AMReX's default 3 the two ghost fills differ by
    construction (``test_mac_project_la.py`` documents and pins that).
    """
    geom, ba, dm = mesh.geom(0), mesh.box_array(0), mesh.dm(0)

    lp = blockamr.MLABecLaplacian(geom, ba, dm, blockamr.LPInfo())
    lp.set_domain_bc(list(p_bc[0]), list(p_bc[1]))
    lp.set_level_bc(0, None)
    lp.set_scalars(0.0, 1.0)
    lp.set_max_order(2)

    b_mfs = []
    for d in range(3):
        ba_face = blockamr.BoxArray(ba)
        ba_face.surrounding_nodes(d)
        b_mf = blockamr.MultiFab(ba_face, dm, 1, 0)
        b_mf.set_val(1.0)
        ba_face.enclosed_cells(d)
        b_mfs.append(b_mf)
    lp.set_b_coeffs(0, b_mfs[0], b_mfs[1], b_mfs[2])

    mlmg = blockamr.MLMG(lp)
    mlmg.set_verbose(0)
    mlmg.set_bottom_verbose(0)
    mlmg.set_max_iter(400)

    p_mf = blockamr.MultiFab(ba, dm, 1, 1)
    p_mf.set_val(0.0)
    rhs_mf = blockamr.MultiFab(ba, dm, 1, 0)
    # Mean-zero on cell centres by symmetry, so the singular (periodic / all-Neumann)
    # configurations get a CONSISTENT right-hand side rather than one MLMG has to
    # modify behind the comparison.
    k = 2.0 * np.pi
    _fill(rhs_mf, geom, lambda x, y, z: np.sin(k * x) * np.sin(k * y) * np.sin(k * z))

    mlmg.solve(p_mf, rhs_mf, _RTOL_SOLVE, _ATOL_SOLVE)

    flux_mfs = []
    for d in range(3):
        ba_face = blockamr.BoxArray(ba)
        ba_face.surrounding_nodes(d)
        flux_mfs.append(blockamr.MultiFab(ba_face, dm, 1, 0))
        ba_face.enclosed_cells(d)
    mlmg.get_fluxes(flux_mfs[0], flux_mfs[1], flux_mfs[2])

    # Keep `lp` and the coefficient MultiFabs alive: MLMG holds the operator, and the
    # operator holds the b-coefficients, by reference.
    return p_mf, flux_mfs, (lp, mlmg, b_mfs)


@pytest.mark.parametrize("case, periodic, p_bc, max_size", _CASES)
def test_the_cell_gradient_matches_get_fluxes_averaged_to_cells(
    blockamr_session, case, periodic, p_bc, max_size
):
    """Off one solved ``p``, the helper IS ``-avg(get_fluxes)``, to the last bit.

    ``get_fluxes`` returns ``-grad_f(p)``, so the cell gradient it implies is minus
    the mean of the two bounding faces. Fed the same pressure the only ways the two
    can differ are staggering, sign, or which ghost a domain face reads -- so bitwise
    is the right bar and a tolerance would hide exactly the mistakes worth catching.
    """
    mesh, geom = _make_mesh(N, periodic, max_size)
    p_mf, flux_mfs, _keepalive = _solved_pressure_and_fluxes(mesh, p_bc)

    want_per_box = []
    flux_arrays = [flux_mfs[d].arrays() for d in range(3)]
    for bi in range(len(flux_arrays[0])):
        components = []
        for d in range(3):
            f = np.asarray(flux_arrays[d][bi][:, :, :, 0])
            sl_hi = [slice(None)] * 3
            sl_lo = [slice(None)] * 3
            sl_hi[d] = slice(1, None)
            sl_lo[d] = slice(0, -1)
            components.append(-0.5 * (f[tuple(sl_hi)] + f[tuple(sl_lo)]))
        want_per_box.append(np.stack(components, axis=-1))

    got_per_box = cell_gradient(p_mf, geom, _bc_list(p_bc))

    # Anti-vacuity: a solve that silently left p at zero would make every comparison
    # below 0 == 0 and the row would pass having checked nothing.
    peak = max(float(np.max(np.abs(w))) for w in want_per_box)
    assert peak > 1e-3, f"{case}: get_fluxes is ~zero ({peak:.3e}), so the solve did nothing"

    for bi, (want, got) in enumerate(zip(want_per_box, got_per_box)):
        np.testing.assert_array_equal(np.asarray(got), want, err_msg=f"{case}: box {bi}")


def test_the_output_is_accepted_by_PressureGradient_unchanged(blockamr_session):
    """``p.grad`` filled from the helper flows through ``exp.grad`` and ``correct``.

    The shape-and-convention claim, made against the real consumers rather than
    against an assertion about shapes: ``exp.grad`` must return a
    ``PressureGradient`` (i.e. read the STORED gradient, not fall back to the
    central-difference ``Grad``), and ``correct`` must apply it to a ncomp=3 velocity.
    A multibox layout, because ``.grad[lev]`` is a per-box LIST.
    """
    mesh, geom = _make_mesh(N, [1, 1, 1], N // 2)
    k = 2.0 * np.pi
    dt = 0.1

    p = CellField(mesh, ncomp=1, ngrow=1, name="p")
    velocity = CellField(mesh, ncomp=3, ngrow=1, name="U")
    _fill(p.mf[0], geom, lambda x, y, z: np.sin(k * x) * np.sin(k * y) * np.sin(k * z))

    p.grad = [cell_gradient(p.mf[0], geom, ["periodic"] * 6)]

    term = exp.grad(p)
    assert isinstance(term, PressureGradient), (
        "exp.grad fell back to the central-difference Grad, so the stored gradient "
        "is not being read at all"
    )

    correct(velocity, -dt * exp.grad(p))

    # `velocity` started at zero, so it now holds exactly the applied correction.
    ng = velocity.mf[0].n_grow()
    got_per_box = [np.asarray(a)[ng:-ng, ng:-ng, ng:-ng, :] for a in velocity.mf[0].arrays()]
    got = _gather(velocity.mf[0], got_per_box, N)
    want = -dt * _gather(p.mf[0], p.grad[0], N)

    np.testing.assert_allclose(got, want, rtol=0.0, atol=0.0)
    assert np.max(np.abs(got)) > 1e-3, "the correction was zero, so nothing was checked"
