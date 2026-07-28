# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""``ghostCell``'s boundary schemes — laplacian, div, grad (B9-B12) and the
explicit source (B41).

The one place an operator and the ``ghostCell`` method meet
(``plans/IBM/design.md`` §6). Each scheme reads the mesh's classification and
wall geometry, the method's own image points, and the field's ``ibm_bc`` —
nothing else, and nothing from the backend.

The shared core: :func:`wall_closure`
-------------------------------------

The wall enters all three operators through **one** object: the closure of
``alpha phi_w + beta dphi/dn = gamma`` against the field at the method's image
point. :func:`wall_closure` returns it as two affine functions of that value —
the surface value ``phi_w`` and the surface gradient ``dphi/dn`` — and
:meth:`WallClosure.at` reads it a third way, as the field at any signed
distance ``d`` from the surface along ``n̂``. Every one of the three schemes
below is an assembly over those reads; none of them does wall arithmetic of
its own (the H1 evidence, design §6).

The rows
--------

``laplacian`` is the interior cross difference written as a flux balance::

    L(phi)|_P = coeff * sum_d ( G_d^+ - G_d^- ) / dx_d

An arm whose face neighbour is **fluid** keeps the interior formula exactly —
``G_d^+ = (phi_{P+d} - phi_P)/dx_d`` — which is what makes a constant
annihilate to the last bit. An arm whose face neighbour is **non-fluid** is
taken at the surface instead, ``G_d = n̂_d * dphi/dn|_w`` from the closure's
gradient half.

``div`` and ``grad`` are face balances::

    div(u phi)|_P = coeff * sum_d ( f_d^+ phi_f^+ - f_d^- phi_f^- ) / dx_d
    grad(phi)|_P  = coeff * ( phi_f^+ - phi_f^- ) / dx_0        (component 0)

with ``phi_f`` the interior scheme's own width-1 face value. A face whose
neighbour is non-fluid keeps that same formula and substitutes, for the
neighbour it cannot read, the field the wall condition extrapolates to that
cell centre: ``phi_G = closure.at(d_G)`` at ``d_G = s_P + step*dx_d*n̂_d``, the
signed distance of the neighbour's centre from the surface along ``P``'s
normal.

Substituting the *cell* value rather than the *face* value is the choice that
keeps the interior formula's telescoping intact, and it is what makes a linear
field exact under **upwind**: the upwind face value is the upwind cell's value,
so replacing it by the surface value ``phi_w`` (the literal reading of design
§9's "the face takes the value extrapolated from the wall condition") would
halve the gradient at the wall — measured ``B/2`` instead of ``B`` on
``test_every_div_scheme_is_exact_on_a_linear_field_at_a_plane_wall``. For a
central (``linear``) face value the two readings coincide.

``source`` is the degenerate case of the same two rules, and the only one whose
row touches no wall at all::

    coeff * S|_P

A pointwise term reads no neighbour, so it has no band of its own and — by the
composition rule below — its row at *every* cell of the equation's band is its
plain interior formula. It is emitted all the same, and must be: the band is
overwritten by the first term's rows and added to by the rest, so a source term
that emitted nothing would have its interior sweep erased on exactly the cells
the wall owns. Hence ``nnz = 0`` (it reads no cell, not even its own — ``S`` is
a coefficient, not the unknown) with the value wholly in ``c``.

A non-fluid cell is a row with ``nnz = 0, c = 0``: the operator has no value
there, and a leaked one would be read by nothing and plotted by everything.

The degrade rule (D1)
---------------------

The rows cover ``band(width)`` for the width the *equation* runs on, and every
one of them is a **width-1** row. Inside the band a wider stencil reaches
through the solid, where there is nothing valid to read; outside it the
interior kernel keeps the full wide scheme, untouched (design §6). A width-2
div scheme (``vanLeer``, ``quick``) therefore degrades to first-order upwind in
``band(w)`` and only there; a width-1 one keeps its own face formula.

``width`` is the *equation's* band width — the widest of its terms — not the
term's own (design §6, "the composition rule"). At a cell outside its own
term's band the row is the plain interior formula, so the value there is the
operator's value and the wall never enters it.
"""

from dataclasses import dataclass

import numpy as np

from ...ibm.band_rows import BandRows
from ...ibm.bc import gamma_rows
from ...ibm.classify import _fluid_at_index, _patches, box_grids
from ...ibm.ghost_cell import GhostCell
from . import register

#: ``self + 6 face neighbours + 8 image donors`` — the width-1 row's slots.
STRIDE = 15

#: Slot layout of a row: 0 is the target, 1..6 the face neighbours (``2*d`` for
#: ``+d``, ``2*d + 1`` for ``-d``), 7..14 the image point's trilinear donors.
_DONOR0 = 7

#: Live slot count of a row that has no wall arm: the target and its six face
#: neighbours. A row *with* a wall arm reads the donors too, and uses them all.
_INTERIOR_NNZ = _DONOR0


@dataclass(frozen=True)
class WallClosure:
    """``(phi_w, dphi/dn|_w)`` at the surface, as affine functions of the field.

    Both are ``linear * phi(image) + constant``, with the wall datum wholly in
    ``constant`` (row-contract §3), so a time-dependent datum rebuilds the
    constants and reuses the linear parts. ``value_*`` is what a face-value
    operator needs, ``grad_*`` what a flux operator needs; the two are the same
    closure read twice, and :meth:`at` is the third read.
    """

    value_linear: np.ndarray  # f64 (n,)
    value_constant: np.ndarray  # f64 (n, ncomp)
    grad_linear: np.ndarray  # f64 (n,)
    grad_constant: np.ndarray  # f64 (n, ncomp)

    def at(self, distance):
        """The field at signed ``distance`` from the surface, along ``n̂``.

        ``phi(d) = phi_w + d dphi/dn|_w`` is the closure's own profile, so this
        is the same object read a third way and not a new approximation:
        ``distance > 0`` is in the fluid and ``distance < 0`` inside the body,
        which is the ghost value ``div`` and ``grad`` substitute for a
        neighbour they cannot read.

        ``distance`` is ``(n,)``; the result is ``(linear (n,),
        constant (n, ncomp))``.
        """
        return (
            self.value_linear + distance * self.grad_linear,
            self.value_constant + distance[:, np.newaxis] * self.grad_constant,
        )


def wall_closure(alpha, beta, gamma, distance):
    """Close ``alpha phi_w + beta dphi/dn|_w = gamma`` against one field value.

    The value is taken at ``distance`` from the surface **along the normal**,
    and the profile between the two is linear::

        phi(d) = phi_w + d dphi/dn|_w

    so, eliminating one unknown at a time,

    ==================  ===============================================
    ``dphi/dn|_w``      ``(gamma - alpha phi_i) / (beta - alpha d)``
    ``phi_w``           ``(beta phi_i - d gamma) / (beta - alpha d)``
    ==================  ===============================================

    Dirichlet (``1, 0, value``) gives ``phi_w = value`` and the one-sided
    difference ``(phi_i - value)/d``; Neumann (``0, 1, g``) gives
    ``dphi/dn = g`` and ``phi_w = phi_i - d g``. Both drop out of the general
    form exactly, so there is no branch on the BC type anywhere.

    ``alpha``, ``beta``, ``distance`` are ``(n,)``; ``gamma`` is
    ``(n, ncomp)``.
    """
    den = (beta - alpha * distance)[:, np.newaxis]
    return WallClosure(
        value_linear=(beta[:, np.newaxis] / den)[:, 0],
        value_constant=-distance[:, np.newaxis] * gamma / den,
        grad_linear=(-alpha[:, np.newaxis] / den)[:, 0],
        grad_constant=gamma / den,
    )


# ---------------------------------------------------------------------------
# the band context every scheme assembles over
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BandContext:
    """Everything the three row builders share, band-length and pure numpy.

    Built once per ``rows`` call from the mesh's classification, the method's
    image points and the field's ``ibm_bc``; no MultiFab and no field value is
    in it, which is what lets the arithmetic below be read (and unit-tested)
    without the compiled extension.
    """

    band: object  # blockamr.ibm.band.Band
    grid: object  # blockamr.ibm.classify.BoxGrid
    body_list: list
    target: np.ndarray  # int32 (n, 3)
    dx: np.ndarray  # f64 (3,)
    normal: np.ndarray  # f64 (n, 3), the owning body's outward normal
    sdf: np.ndarray  # f64 (n,), signed distance of the target
    closure: WallClosure
    donor: np.ndarray  # int32 (n, 8, 3)
    weight: np.ndarray  # f64 (n, 8)
    at_wall: np.ndarray  # bool (n,), depth == 1: the rows with a wall arm
    fluid: np.ndarray  # bool (n,), depth >= 1

    @property
    def nrows(self):
        return int(self.target.shape[0])


def _context(term, ibm, lev, ncomp, t, width):
    """The band context of one term on one level, at the equation's ``width``.

    ``t`` is the evaluation time the rows are being built at — a stage time on
    the ``solve()`` path — and reaches the context only through the wall datum,
    which may be a schedule (:func:`~blockamr.ibm.bc.gamma_rows`).
    """
    band = ibm.band(lev, width)
    names, body_list = _patches(ibm.bodies)
    geometries = ibm.geometry(lev)
    data = ibm.data(GhostCell, lev)
    grid = box_grids(ibm.mesh, lev)[0]
    ibm_bc = term.field.ibm_bc
    donor, weight = _band_image(band, data)
    return _BandContext(
        band=band,
        grid=grid,
        body_list=body_list,
        target=np.ascontiguousarray(band.cell, dtype=np.int32),
        dx=np.asarray(grid.dx, dtype=float),
        normal=_band_field(geometries, width, "normal", (0, 3)),
        sdf=_band_field(geometries, width, "sdf", (0,)),
        closure=_band_closure(
            band,
            data,
            [ibm_bc[name].robin() for name in names],
            ncomp,
            _band_field(geometries, width, "wall_point", (0, 3)),
            t,
        ),
        donor=donor,
        weight=weight,
        at_wall=band.depth == 1,
        fluid=band.depth >= 1,
    )


def _band_image(band, data):
    """The image stencil of every band row: ``(donor (n, 8, 3), weight (n, 8))``.

    A row that is not at the wall has no image point; it indexes entry 0, which
    it never reads (its donor slots are dead and carry ``a = 0``), so the only
    case needing care is a band with no wall-layer cell at all.
    """
    row = _data_row(band, data)
    if data.nrows == 0:
        return (
            np.zeros((band.nrows, data.donor.shape[1], 3), dtype=data.donor.dtype),
            np.zeros((band.nrows, data.weight.shape[1])),
        )
    return data.donor[row], data.weight[row]


def _blank(ctx, ncomp, stride):
    """The empty row arrays every builder starts from.

    Every slot points at the target: a slot the builder does not fill is either
    dead (``k >= nnz``) or carries ``a = 0``, and in both cases it must be a
    fluid cell inside the box (Invariant F, row-contract §8).
    """
    stencil = np.repeat(ctx.target[:, np.newaxis, :], stride, axis=1)
    a = np.zeros((ctx.nrows, stride))
    c = np.zeros((ctx.nrows, ncomp))
    nnz = np.zeros(ctx.nrows, dtype=np.int32)
    # depth <= 0 is non-fluid: nnz = 0, c = 0. A fluid row reads the target and
    # its six face neighbours; only one with a wall arm reads the image donors.
    nnz[ctx.fluid] = _INTERIOR_NNZ
    nnz[ctx.at_wall] = stride
    return stencil, a, c, nnz


def _close_donors(ctx, stencil):
    """Point the donor slots at the image stencil — for the wall rows only.

    A deeper fluid row has no wall arm, so its donor slots are dead
    (``nnz = 7``) and must stay on the target: :func:`_data_row` maps such a
    row to entry 0 of the method's data, whose donors belong to another cell
    and possibly another box.
    """
    live = ctx.at_wall[:, np.newaxis] & (ctx.weight != 0.0)
    stencil[:, _DONOR0:, :] = np.where(
        live[..., np.newaxis], ctx.donor, ctx.target[:, np.newaxis, :]
    )


def _neighbour(ctx, d, step):
    """``(index, is_fluid)`` of the ``step``-th face neighbour along ``d``."""
    index = ctx.target.astype(np.int64).copy()
    index[:, d] += step
    return index, _fluid_at_index(index, ctx.grid, ctx.body_list) & ctx.fluid


def _slot(d, step):
    return 1 + 2 * d + (0 if step == 1 else 1)


# ---------------------------------------------------------------------------
# laplacian
# ---------------------------------------------------------------------------


@register
class GhostCellLaplacian:
    """``laplacian x ghostCell``: the cross difference with its wall arms
    closed through the surface."""

    operator = "laplacian"
    method = "ghostCell"
    stride = STRIDE

    def __init__(self, interior_scheme):
        #: Kept for the record; the band width is the equation's, and the rows
        #: are width 1 inside it (the degrade rule).
        self.interior = interior_scheme

    def rows(self, term, ibm, lev, ncomp, t, width):
        """The affine rows of one term on one level."""
        ctx = _context(term, ibm, lev, ncomp, t, width)
        return _closed_flux_rows(ctx, _coefficient(term, t), ncomp, self.stride)

    def build_cpp_kernel(self):
        """The compiled peer of :meth:`rows` — ``laplacian x ghostCell`` (B32).

        Mirrors how an interior scheme's ``build_cpp_kernel()`` returns the
        wrapper for its accumulate binding. The compiled pair reaches the same
        rows this method builds, bitwise, and it reaches them at the cell
        instead of through a band table.

        Nothing calls this yet: :class:`~blockamr.ibm.driver.BandEvaluation`
        still goes through :meth:`rows`, and flipping the driver over is B36.
        """
        from ...cpp_kernels import CppWallKernel

        return CppWallKernel("wall_laplacian_ghost_cell")


def _closed_flux_rows(ctx, coeff, ncomp, stride):
    """``coeff * sum_d (G_d^+ - G_d^-) / dx_d`` — pure numpy, band-length."""
    stencil, a, c, nnz = _blank(ctx, ncomp, stride)
    for d in range(3):
        for step in (1, -1):
            slot = _slot(d, step)
            index, nb_fluid = _neighbour(ctx, d, step)

            # interior arm: the interior scheme's own formula, exactly
            stencil[nb_fluid, slot, :] = index[nb_fluid]
            a[nb_fluid, slot] += 1.0 / ctx.dx[d] ** 2
            a[nb_fluid, 0] -= 1.0 / ctx.dx[d] ** 2

            # wall arm: the flux through the surface, in this arm's direction
            arm = ctx.fluid & ~nb_fluid
            scale = step * ctx.normal[:, d] / ctx.dx[d]
            a[arm, _DONOR0:] += (scale * ctx.closure.grad_linear)[arm, np.newaxis] * ctx.weight[arm]
            c[arm] += scale[arm, np.newaxis] * ctx.closure.grad_constant[arm]

    _close_donors(ctx, stencil)
    a *= coeff
    c *= coeff
    return _rows(ctx, stencil, a, nnz, c, stride)


# ---------------------------------------------------------------------------
# div and grad — the same face balance, summed and differenced
# ---------------------------------------------------------------------------


@register
class GhostCellDiv:
    """``div x ghostCell``: width-1 face values, with a non-fluid neighbour
    replaced by the field the wall condition extrapolates to it."""

    operator = "div"
    method = "ghostCell"
    stride = STRIDE

    def __init__(self, interior_scheme):
        self.interior = interior_scheme

    def rows(self, term, ibm, lev, ncomp, t, width):
        ctx = _context(term, ibm, lev, ncomp, t, width)
        flux = _band_face_flux(term.coefficient, lev, ctx.band)
        return _face_balance_rows(
            ctx,
            axes=(0, 1, 2),
            flux=flux,
            weight_self=_face_weights(self.interior, flux),
            coeff=float(term.coeff),
            ncomp=ncomp,
            stride=self.stride,
        )

    def build_cpp_kernel(self):
        """The compiled peer of :meth:`rows` — ``div x ghostCell`` (B33).

        Nothing calls this yet, exactly as for
        :meth:`GhostCellLaplacian.build_cpp_kernel`:
        :class:`~blockamr.ibm.driver.BandEvaluation` still uploads a
        ``BandTable``, and flipping it over is B36. The compiled pair reaches
        the same rows :meth:`rows` builds, bitwise, and it reaches them at the
        cell instead of through a band table.

        ``wall_div_ghost_cell`` takes four arguments past the canonical twelve
        — ``flux_x, flux_y, flux_z, face_value`` — because a ``div`` row is a
        *face* balance. **B36 owns the face-value mapping** at the call site:
        ``self.interior.type`` selects ``DivFaceValue.Central`` for ``Linear``
        and ``DivFaceValue.Upwind`` for ``Upwind``, ``VanLeer`` and ``QUICK``
        — the last two being the D1 degrade, since a width-2 stencil reaches
        through the solid inside the band.
        """
        from ...cpp_kernels import CppWallKernel

        return CppWallKernel("wall_div_ghost_cell")


@register
class GhostCellGrad:
    """``grad x ghostCell``: the same face closure, differenced instead of
    summed."""

    operator = "grad"
    method = "ghostCell"
    stride = STRIDE

    def __init__(self, interior_scheme):
        self.interior = interior_scheme

    def rows(self, term, ibm, lev, ncomp, t, width):
        _check_grad_ncomp(term, ncomp)
        ctx = _context(term, ibm, lev, ncomp, t, width)
        ones = np.ones((ctx.nrows, 3, 2))
        return _face_balance_rows(
            ctx,
            # The gradient's component ``n`` is the difference along axis
            # ``n``, and one row serves every component (row-contract §2), so
            # only ``n = 0`` is expressible — see :func:`_check_grad_ncomp`.
            axes=(0,),
            flux=ones,
            weight_self=0.5 * ones,
            coeff=float(term.coeff),
            ncomp=ncomp,
            stride=self.stride,
        )


def _check_grad_ncomp(term, ncomp):
    """``grad`` is a row only for a one-component field.

    The row format applies **one** ``a`` to every component
    (``out(target, n) = sum_k a[k] phi(stencil[k], n)``, row-contract §2), and
    the gradient's component ``n`` is a difference along axis ``n`` — a
    different stencil per component. For ``ncomp = 1`` the two agree (the only
    component is the axis-0 derivative, which is what the interior kernel
    writes into it); beyond that they cannot, so this refuses rather than
    returning a plausible field.
    """
    if ncomp != 1:
        raise NotImplementedError(
            f"grad x ghostCell needs a one-component field, but '{term.field.name}' has "
            f"ncomp = {ncomp}: the band row applies one coefficient list to every "
            "component, while the gradient's component n is the difference along axis n. "
            "Expressing that needs a per-component row, which the v1 row format "
            "(plans/IBM/row-contract.md §2) does not have."
        )


def _face_weights(interior_scheme, flux):
    """The interior scheme's width-1 face weight on the **target's** value.

    ``(n, 3, 2)``: index ``0`` along the last axis is the low face of the cell,
    ``1`` the high one. The neighbour's weight is ``1 - this``, so a face value
    is a convex combination and a constant field is reproduced exactly.

    ``Linear`` keeps its central average; every other scheme — first-order
    ``Upwind``, and the width-2 ``vanLeer``/``QUICK`` that degrade to it inside
    the band (D1) — takes the value of the cell the flux comes from. The flux
    is stored in the ``+d`` orientation on both faces, so ``f >= 0`` means the
    low face's upstream cell is the *neighbour* and the high face's is the
    *target*.
    """
    if getattr(interior_scheme, "type", None) == "Linear":
        return np.full(flux.shape, 0.5)
    positive = flux >= 0.0
    weight = np.where(positive, 0.0, 1.0)  # the low face
    weight[..., 1] = np.where(positive[..., 1], 1.0, 0.0)  # the high face
    return weight


def _face_balance_rows(ctx, axes, flux, weight_self, coeff, ncomp, stride):
    """``coeff * sum_{d in axes} (f_d^+ phi_f^+ - f_d^- phi_f^-) / dx_d``.

    ``flux`` and ``weight_self`` are ``(n, 3, 2)`` with the low face at index
    ``0``. A face whose neighbour is non-fluid keeps this formula and reads the
    wall closure at the neighbour's own signed distance instead of the
    neighbour.
    """
    stencil, a, c, nnz = _blank(ctx, ncomp, stride)
    for d in axes:
        for step in (1, -1):
            face = 1 if step == 1 else 0
            slot = _slot(d, step)
            index, nb_fluid = _neighbour(ctx, d, step)

            # coeff * (+f phi_f) at the high face, coeff * (-f phi_f) at the low
            scale = coeff * step * flux[:, d, face] / ctx.dx[d]
            self_part = scale * weight_self[:, d, face]
            nb_part = scale * (1.0 - weight_self[:, d, face])
            a[ctx.fluid, 0] += self_part[ctx.fluid]

            # interior face: the interior scheme's own formula, exactly
            stencil[nb_fluid, slot, :] = index[nb_fluid]
            a[nb_fluid, slot] += nb_part[nb_fluid]

            # wall face: the neighbour's value comes from the wall condition,
            # extrapolated along n̂ to the neighbour's own cell centre
            arm = ctx.fluid & ~nb_fluid
            linear, constant = ctx.closure.at(ctx.sdf + step * ctx.dx[d] * ctx.normal[:, d])
            a[arm, _DONOR0:] += (nb_part * linear)[arm, np.newaxis] * ctx.weight[arm]
            c[arm] += nb_part[arm, np.newaxis] * constant[arm]

    _close_donors(ctx, stencil)
    return _rows(ctx, stencil, a, nnz, c, stride)


def _band_face_flux(face_field, lev, band):
    """The face flux at each band cell's low and high face, ``(n, 3, 2)``.

    The one per-evaluate device-to-host read the design names (§8): a ``div``
    row depends on the flux field, so it is rebuilt every evaluate until B19
    moves the assembly to the device.
    """
    import blockamr

    flux = np.zeros((band.nrows, 3, 2))
    if band.nrows == 0:
        return flux
    for d in range(3):
        mf = face_field[lev][d].mf
        for bi, mfi in enumerate(blockamr.MFIterator(mf)):
            span = slice(int(band.box_offset[bi]), int(band.box_offset[bi + 1]))
            if span.start == span.stop:
                continue
            arr = np.asarray(mf.copy_to_host(mfi))
            lo = np.asarray([int(v) for v in mfi.valid_box().small_end()])
            low = band.cell[span].astype(np.int64) - lo
            high = low.copy()
            high[:, d] += 1
            flux[span, d, 0] = arr[low[:, 0], low[:, 1], low[:, 2], 0]
            flux[span, d, 1] = arr[high[:, 0], high[:, 1], high[:, 2], 0]
    return flux


# ---------------------------------------------------------------------------
# the explicit (Su) source — a row with no wall in it
# ---------------------------------------------------------------------------


@register
class GhostCellSource:
    """``source x ghostCell``: the term's plain interior value, over the band.

    The only boundary scheme here that never builds a :func:`_context`: it needs
    no wall closure, no image point and no ``ibm_bc``, because the source field
    is a coefficient and carries no boundary condition of its own. What it needs
    is the band (to know which cells to write) and the classification's depth
    (to write nothing inside the body).
    """

    operator = "source"
    method = "ghostCell"
    #: One dead slot, pointed at the row's own target — the ``nnz = 0`` shape
    #: :func:`~blockamr.ibm.band_rows.pin_rows` already uses.
    stride = 1

    def __init__(self, interior_scheme):
        self.interior = interior_scheme

    def rows(self, term, ibm, lev, ncomp, t, width):
        band = ibm.band(lev, width)
        target = np.ascontiguousarray(band.cell, dtype=np.int32)
        nrows = band.nrows
        c = float(term.coeff) * _band_cell_values(term.field, lev, band, ncomp)
        c[band.depth <= 0] = 0.0
        return BandRows(
            target=target,
            stencil=np.ascontiguousarray(target[:, np.newaxis, :]),
            a=np.zeros((nrows, 1)),
            nnz=np.zeros(nrows, dtype=np.int32),
            c=np.ascontiguousarray(c, dtype=np.float64),
            patch=np.ascontiguousarray(band.patch, dtype=np.int32),
            box_offset=band.box_offset,
            stride=self.stride,
        )


def _band_cell_values(field, lev, band, ncomp):
    """The source field's value at each band cell, ``(n, ncomp)``.

    The sibling of :func:`_band_face_flux`, one axis simpler: a source row
    depends on the source field, so it is a per-evaluate device-to-host read
    until the assembly moves to the device (design §8).
    """
    import blockamr

    if field.ncomp != ncomp:
        raise NotImplementedError(
            f"the source field '{field.name}' has ncomp = {field.ncomp} but the equation's "
            f"field has {ncomp}; a band row carries one constant per component of the "
            "solved field, so the two must agree."
        )
    values = np.zeros((band.nrows, ncomp))
    if band.nrows == 0:
        return values
    mf = field.mf[lev]
    for bi, mfi in enumerate(blockamr.MFIterator(mf)):
        span = slice(int(band.box_offset[bi]), int(band.box_offset[bi + 1]))
        if span.start == span.stop:
            continue
        arr = np.asarray(mf.copy_to_host(mfi))
        lo = np.asarray([int(v) for v in mfi.valid_box().small_end()])
        index = band.cell[span].astype(np.int64) - lo
        values[span] = arr[index[:, 0], index[:, 1], index[:, 2], :ncomp]
    return values


# ---------------------------------------------------------------------------
# shared band-length lookups
# ---------------------------------------------------------------------------


def _band_closure(band, data, robin, ncomp, wall_point, t):
    """:func:`wall_closure` at every band row, padded where it is not used.

    Only a ``depth == 1`` row can have a wall arm, and only those rows have an
    image point. Deeper fluid rows (a degraded wide scheme, or a narrow term
    inside a wider equation's band) are plain interior rows and never read
    this; the padding is the identity closure, so reading it would be visible
    rather than silently plausible.

    The datum ``gamma`` is read **per row**, at that row's own wall foot point
    and at the evaluation time ``t``, so a callable datum is a schedule the
    rows follow (Q22). This is v1's host-side form of what design §8 moves into
    the kernel later; it costs nothing extra because v1 rebuilds these rows on
    every evaluate anyway.
    """
    nrows = band.nrows
    alpha = np.zeros(nrows)
    beta = np.ones(nrows)
    gamma = np.zeros((nrows, ncomp))
    distance = np.ones(nrows)

    at_wall = band.depth == 1
    if at_wall.any():
        own = band.patch[at_wall].astype(np.int64)
        alpha[at_wall] = np.array([r[0] for r in robin])[own]
        beta[at_wall] = np.array([r[1] for r in robin])[own]
        gamma[at_wall] = _wall_gamma(robin, own, wall_point[at_wall], t, ncomp)
        distance[at_wall] = data.distance
    return wall_closure(alpha, beta, gamma, distance)


def _wall_gamma(robin, own, points, t, ncomp):
    """Every wall row's datum, evaluated per patch at that patch's own rows.

    Per patch and not per row because a callable datum is handed its whole
    patch at once (``f(x, y, z, t)`` on ``(n,)`` arrays), which is both the
    repo's coefficient convention and the reason the evaluation is cheap.
    """
    out = np.zeros((points.shape[0], ncomp))
    for patch, (_alpha, _beta, datum) in enumerate(robin):
        sel = own == patch
        if sel.any():
            out[sel] = gamma_rows(datum, points[sel], t, ncomp)
    return out


def _data_row(band, data):
    """Index into :class:`~blockamr.ibm.ghost_cell.GhostCellData` per band row.

    ``GhostCellData`` lists the ``depth == 1`` cells of each box in the order
    the band lists them, so the map is a running count over that subset — and
    rows that have no image point index slot 0, which they never read.
    """
    at_wall = band.depth == 1
    count = int(at_wall.sum())
    if count != data.nrows:
        raise ValueError(
            f"the band has {count} wall-layer fluid cells but ghostCell preprocessed "
            f"{data.nrows}: the two were built from different grid generations."
        )
    row = np.zeros(band.nrows, dtype=np.int64)
    row[at_wall] = np.arange(count)
    return row


def _band_field(geometries, width, name, empty_shape):
    """One per-cell geometry array of every band row, flattened in band order."""
    blocks = [getattr(g, name)[g.depth <= width] for g in geometries]
    return np.concatenate(blocks) if blocks else np.zeros(empty_shape)


def _coefficient(term, t):
    """``coeff * coefficient`` — the same scalar the interior kernel is built with."""
    gamma = term.coefficient
    if isinstance(gamma, (int, float)):
        return float(term.coeff) * float(gamma)
    if not callable(gamma):
        raise TypeError(f"laplacian gamma must be callable or a number, got {type(gamma)}")
    probe = [
        float(np.asarray(gamma(np.array([x]), np.array([y]), np.array([z]), t)).ravel()[0])
        for x, y, z in ((0.1, 0.2, 0.3), (0.7, 0.4, 0.15))
    ]
    if abs(probe[0] - probe[1]) > 1e-14 * (abs(probe[0]) + 1.0):
        raise NotImplementedError(
            "the ghostCell laplacian boundary scheme needs a constant gamma; the wall row "
            "would otherwise have to carry a per-cell diffusivity, which the row format "
            "does not express yet."
        )
    return float(term.coeff) * probe[0]


def _rows(ctx, stencil, a, nnz, c, stride):
    return BandRows(
        target=ctx.target,
        stencil=np.ascontiguousarray(stencil, dtype=np.int32),
        a=np.ascontiguousarray(a, dtype=np.float64),
        nnz=nnz,
        c=np.ascontiguousarray(c, dtype=np.float64),
        patch=np.ascontiguousarray(ctx.band.patch, dtype=np.int32),
        box_offset=ctx.band.box_offset,
        stride=stride,
    )
