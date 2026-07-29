# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""``ghostCell``'s four boundary schemes — laplacian, div, grad and the
explicit source.

The one place an operator and the ``ghostCell`` method meet
(``plans/IBM/design.md`` §6). Under v2 each of the four **names a compiled
kernel** and computes nothing: the 598 lines of numpy row assembly that used to
live here went with the band, and what a pair still owns in Python is three
small things —

* ``build_cpp_kernel()`` — the entry point, the exact peer of an interior
  scheme's;
* ``wall_coeff(term, t)`` — the scalar the frame multiplies the sink value by;
* ``wall_extras(term, lev)`` — whatever that pair appends past the canonical
  twelve (``div``'s three face fluxes and its face-value selector; the source
  field; nothing for ``laplacian`` and ``grad``).

Two refusals stay Python-side because they need a *name* the kernel does not
have: ``grad``'s ``ncomp > 1`` (:func:`_check_grad_ncomp`, which names the
field) and the laplacian's space-varying gamma (:func:`_coefficient`). Both are
reconciled with the compiled guard that says the same thing — see
``GhostCellGrad.wall_coeff``.

What each pair computes, and where
----------------------------------

======================  ==================================================
``wall_laplacian_...``  the interior cross difference, with each wall-facing
                        arm replaced by a flux through the surface
``wall_div_...``        the width-1 face balance, with a ``SOLID`` neighbour's
                        value substituted by ``closure.at(d_G)`` at that
                        neighbour's own centre
``wall_grad_...``       the same substitution on one axis, differenced
``wall_source_...``     the term's plain interior value — no wall in it at all
======================  ==================================================

:class:`WallClosure` and :func:`wall_closure` are **kept**, and not because a
pair calls them: nothing in this module does any more. They are the numpy
statement of ``robin.H``'s ``closure(alpha, beta, gamma, d)``, and
``test_ibm_robin_closure.py`` is written against them as its oracle. Deleting
them would delete that file's independent side.
"""

from dataclasses import dataclass

import numpy as np

from . import register


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


@register
class GhostCellLaplacian:
    """``laplacian x ghostCell``: the cross difference with its wall arms
    closed through the surface."""

    operator = "laplacian"
    method = "ghostCell"

    def __init__(self, interior_scheme):
        #: Kept for the record; this pair reads no interior face rule.
        self.interior = interior_scheme

    def build_cpp_kernel(self):
        """The compiled peer of :meth:`rows` — ``laplacian x ghostCell`` (B32).

        Mirrors how an interior scheme's ``build_cpp_kernel()`` returns the
        wrapper for its accumulate binding. The compiled pair reaches the same
        rows this method builds, bitwise, and it reaches them at the cell
        instead of through a band table.

        This is what :class:`~blockamr.ibm.driver.WallEvaluation` calls on a
        production evaluate (B36); :meth:`rows` is kept as the oracle the
        row-parity suite compares the pair against.
        """
        from ...cpp_kernels import CppWallKernel

        return CppWallKernel("wall_laplacian_ghost_cell")

    def wall_coeff(self, term, t):
        """The scalar the pair is launched with: ``coeff * gamma``.

        The canonical twelve carry no diffusivity, so the constant gamma is
        folded in here — the same fold :func:`_coefficient` does for the rows,
        with the same refusal of a gamma that is not constant.
        """
        return _coefficient(term, t)

    def wall_extras(self, term, lev):
        """``laplacian x ghostCell`` takes exactly the canonical twelve."""
        return {}


# ---------------------------------------------------------------------------
# div and grad — the same face balance, summed and differenced
# ---------------------------------------------------------------------------


@register
class GhostCellDiv:
    """``div x ghostCell``: width-1 face values, with a non-fluid neighbour
    replaced by the field the wall condition extrapolates to it."""

    operator = "div"
    method = "ghostCell"

    def __init__(self, interior_scheme):
        self.interior = interior_scheme

    def build_cpp_kernel(self):
        """The compiled peer of :meth:`rows` — ``div x ghostCell`` (B33).

        Called by :class:`~blockamr.ibm.driver.WallEvaluation` on a production
        evaluate (B36); the compiled pair reaches the same rows :meth:`rows`
        builds, bitwise, and it reaches them at the cell instead of through a
        band table.

        ``wall_div_ghost_cell`` takes four arguments past the canonical twelve
        — ``flux_x, flux_y, flux_z, face_value`` — because a ``div`` row is a
        *face* balance. They are :meth:`wall_extras`.
        """
        from ...cpp_kernels import CppWallKernel

        return CppWallKernel("wall_div_ghost_cell")

    def wall_coeff(self, term, t):
        """The term's own scalar: a ``div`` row carries no other coefficient
        (the flux is a field the pair reads)."""
        return float(term.coeff)

    def wall_extras(self, term, lev):
        """The four arguments past the twelve — the face fluxes and the
        face-value selector (design §4.4, shipped B33).

        The mapping is this class's, made here and nowhere else:
        ``self.interior.type`` selects ``DivFaceValue.Central`` for ``Linear``
        and ``DivFaceValue.Upwind`` for ``Upwind``, ``VanLeer`` and ``QUICK``
        — the last two being the D1 degrade, since a width-2 stencil reaches
        through the solid inside the band. It is the compiled twin of
        :func:`_face_weights`, which the rows use.
        """
        import blockamr

        faces = term.coefficient[lev]
        central = getattr(self.interior, "type", None) == "Linear"
        return {
            "flux_x": faces[0].mf,
            "flux_y": faces[1].mf,
            "flux_z": faces[2].mf,
            "face_value": (
                blockamr.DivFaceValue.Central if central else blockamr.DivFaceValue.Upwind
            ),
        }


@register
class GhostCellGrad:
    """``grad x ghostCell``: the same face closure, differenced instead of
    summed."""

    operator = "grad"
    method = "ghostCell"

    def __init__(self, interior_scheme):
        self.interior = interior_scheme

    def build_cpp_kernel(self):
        """The compiled peer of :meth:`rows` — ``grad x ghostCell`` (B34).

        Called by :class:`~blockamr.ibm.driver.WallEvaluation` on a production
        evaluate (B36).

        ``wall_grad_ghost_cell`` takes **exactly** the canonical twelve — a
        ``grad`` row is a one-axis face balance at ``flux = 1`` and
        ``weight_self = 0.5``, so unlike ``div`` it needs no face field and no
        face-value selector, and the interior scheme is not read at all.
        """
        from ...cpp_kernels import CppWallKernel

        return CppWallKernel("wall_grad_ghost_cell")

    def wall_coeff(self, term, t):
        """The term's own scalar — and the one place the ``ncomp > 1`` refusal
        is **reconciled** (B36, api §9).

        Both surfaces refuse; they refuse with different types because they are
        different surfaces. A direct call to the binding raises the compiled
        ``RuntimeError`` naming the entry point, which is what a C++ guard can
        say. A call through the *Python* driver raises v1's
        ``NotImplementedError`` naming the field — the sentence api §9's error
        table promises — because the driver knows the field's name and the
        kernel does not. ``NotImplementedError`` is a ``RuntimeError``, so a
        caller that catches the compiled type catches this one too, and the
        Python surface never changed type across the port.
        """
        _check_grad_ncomp(term, term.field.ncomp)
        return float(term.coeff)

    def wall_extras(self, term, lev):
        """``grad x ghostCell`` takes exactly the canonical twelve."""
        return {}


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


# ---------------------------------------------------------------------------
# the explicit (Su) source — a row with no wall in it
# ---------------------------------------------------------------------------


@register
class GhostCellSource:
    """``source x ghostCell``: the term's plain interior value at a wall cell.

    The only pair here that closes no wall at all: it reads no wall closure, no
    image point and no ``ibm_bc``, because an explicit (Su) source field is a
    *coefficient* and carries no boundary condition of its own. What it reads is
    the source field, at the cell — and the marker, to know which cells are its.

    It is emitted all the same, and must be: the first term of an equation
    overwrites the wall cells and the rest add, so a source term that emitted
    nothing would have its interior sweep erased on exactly the cells the wall
    owns.
    """

    operator = "source"
    method = "ghostCell"

    def __init__(self, interior_scheme):
        #: Kept for the record; a pointwise term has no face rule to mirror.
        self.interior = interior_scheme

    def build_cpp_kernel(self):
        """The compiled peer — ``source x ghostCell``, the fourth pair.

        With it every registered pair is compiled, which is what let the row
        path, ``band_table.cpp`` and ``wall_table.cpp`` go.

        ``wall_source_ghost_cell`` takes one argument past the canonical twelve
        — ``source`` — because the value it writes *is* a field.
        """
        from ...cpp_kernels import CppWallKernel

        return CppWallKernel("wall_source_ghost_cell")

    def wall_coeff(self, term, t):
        """The term's own scalar. The kernel multiplies the source value by it,
        where v1 folded it into the row's constant — the same product, one
        multiplication on either side.

        No ``ncomp`` guard here, unlike ``grad``'s: an explicit source's operand
        is **not** the solved field, so this surface cannot compare the two.
        The refusal is the compiled one (``Maker::validate``), which has both
        counts and names its entry point (api §9).
        """
        return float(term.coeff)

    def wall_extras(self, term, lev):
        """The one argument past the twelve: the source field itself."""
        return {"source": term.field.mf[lev]}


# ---------------------------------------------------------------------------
# the one shared helper left
# ---------------------------------------------------------------------------


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
