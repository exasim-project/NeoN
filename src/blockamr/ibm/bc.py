# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Surface BC models for the immersed boundary — the ``ibm_bc`` payload.

All three are the triple ``(alpha, beta, gamma)`` in the one surface condition
(design §1.3)::

    alpha * phi_w + beta * dphi/dn|_w = gamma

so a single row formula serves them all. ``robin()`` is the whole interface the
row builders use; ``gamma`` may be a scalar or a per-component sequence, which
:func:`broadcast_gamma` takes to ``(ncomp,)``, a :class:`Harmonic` — the one
time-dependent datum that reaches the **device** as a compiled expression — or a
**callable of the evaluation time**, which :func:`gamma_rows` evaluates
host-side at the wall rows' own foot points.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Harmonic:
    """``gamma(t) = a0 + ac cos(omega t) + as sin(omega t)`` — the device datum.

    The Python spelling of ``GammaExpr``'s ``Harmonic`` form
    (``schemes/boundary/robin_data.H``): a wall datum that varies in **time**
    and is evaluated *inside* the wall kernel, once per (cell, component), from
    four numbers uploaded with the Robin table. That is design §8's "a
    time-dependent wall value is a ``gamma`` read inside the kernel", and Q4's
    "``gamma(t)`` is a compiled expression, not a callback".

    The cos/sin basis is the compiled one, and it is not a preference: A4
    (Stokes' second problem) is ``U0 cos(omega t)`` — ``Harmonic(cos_amplitude=U0,
    omega=omega)`` — and A6 (Womersley) is ``-(G/omega) sin(omega t)`` —
    ``Harmonic(sin_amplitude=-G/omega, omega=omega)``. An amplitude/phase form
    would spell A6 as ``cos(omega t - pi/2)``, which is not bitwise
    ``sin(omega t)``, and A4/A6's own ``_fit_harmonic`` measures in this basis.

    It is **also** callable with the repo's coefficient signature
    ``f(x, y, z, t)``, so the same object drives the host-side paths that have
    no device kernel — v1's row builder (:func:`gamma_rows`) and the jax backend
    — and the two spellings cannot drift apart into two different waveforms.

    A datum that varies across the patch is still out of scope: one
    ``GammaExpr`` serves a whole patch (review.md §4 Q25, OP-1).
    """

    mean: float = 0.0  #: ``a0``
    cos_amplitude: float = 0.0  #: ``ac``
    sin_amplitude: float = 0.0  #: ``as``
    omega: float = 0.0  #: angular frequency

    def params(self):
        """``(a0, ac, as, omega)`` — the four numbers ``GammaExpr`` holds."""
        return (
            float(self.mean),
            float(self.cos_amplitude),
            float(self.sin_amplitude),
            float(self.omega),
        )

    def at(self, t):
        """The datum at time ``t``, as a scalar — the host peer of
        ``GammaExpr::operator()``, spelled in the same order so the two agree to
        the last bit on any conforming libm."""
        a0, ac, a_s, omega = self.params()
        return a0 + ac * np.cos(omega * t) + a_s * np.sin(omega * t)

    def __call__(self, x, y, z, t):
        """The repo's coefficient signature, for the host-side paths."""
        return np.full(np.shape(x), self.at(t))


def broadcast_gamma(value, ncomp):
    """Broadcast a scalar or per-component BC datum to shape ``(ncomp,)``."""
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.repeat(arr, ncomp)
    if arr.size != ncomp:
        raise ValueError(f"IBM BC datum has {arr.size} components but the field has ncomp={ncomp}")
    return arr


def gamma_rows(value, points, t, ncomp):
    """One patch's wall datum at every one of its wall rows, ``(n, ncomp)``.

    A constant datum is :func:`broadcast_gamma` repeated over the rows — the
    same bits in every row, which is what keeps a constant-datum result bitwise
    what it was before callables existed.

    A **callable** datum is the time-dependent surface value A4/A6 need
    (review.md §4 Q22): it is spelled with the repo's standard coefficient
    signature ``f(x, y, z, t)`` — the one ``exp.source``'s ``coeff_func`` and
    ``update_face_fluxes``'s velocity function already use — and is evaluated
    host-side, once per row build, at ``points``: the wall **foot points** of
    the patch's ``depth == 1`` rows, not their cell centres. Because v1 rebuilds
    the rows on every evaluate (design §8), that is the stage time of the stage
    being evaluated, with no cache to invalidate.

    The result must be ``(n,)`` — one datum per row, applied to every component
    — or ``(n, ncomp)``. Scalars and ``(ncomp,)`` sequences stay the
    *non-callable* spelling, so ``n == ncomp`` is never ambiguous.

    What this is **not**: a purely spatial datum (A3's rotating wall, spelled
    ``f(x, y, z)``) is out of scope here and is not accepted — Q25 OP-1. Nor is
    it v2's form: v2 compiles ``gamma(t)`` into the kernel and takes no Python
    callable (Q4, design §4.4/§8, carried by B30).

    Called once per ``(term, level, stage)`` row build — a two-term equation on
    two levels invokes the datum four times per stage — so keep it cheap and
    stateless. The non-callable branch returns a **read-only** broadcast view;
    the caller copies it into the row arrays.
    """
    n = int(points.shape[0])
    if not callable(value):
        return np.broadcast_to(broadcast_gamma(value, ncomp), (n, ncomp))
    out = np.asarray(value(points[:, 0], points[:, 1], points[:, 2], t), dtype=float)
    if out.shape == (n,):
        return np.repeat(out[:, np.newaxis], ncomp, axis=1)
    if out.shape != (n, ncomp):
        raise ValueError(
            f"a callable IBM BC datum is f(x, y, z, t) evaluated at the patch's {n} wall "
            f"points; it must return shape ({n},) or ({n}, {ncomp}), got {out.shape}."
        )
    return out


def robin_data(names, ibm_bc, ncomp, wall_points, t):
    """The per-patch ``(alpha, beta, gamma(t))`` tables a wall pair reads (B36).

    ``names`` is :func:`~blockamr.ibm.classify._patches`' patch order, i.e. the
    enumeration ``IbmGeometry.patch`` carries; the row ``p`` of every table
    below is the body ``names[p]``, and a table built on any other order applies
    the wrong condition to the wrong body silently.

    ``gamma`` reaches the device as a **compiled expression** and never as a
    Python callable (Q4, design §4.4). The two spellings v1 accepts are
    reconciled here, at the one call site design §4.4 names:

    * a **constant** datum — scalar or per-component — is the ``Constant`` tag,
      bitwise the number the user wrote;
    * a :class:`Harmonic` datum is the ``Harmonic`` tag: its four numbers cross
      as they are, and ``gamma(t)`` is evaluated **device-side, in the kernel**,
      per (cell, component). Nothing about it is collapsed here, so the sweep is
      handed the schedule and not a sample of it — design §8, Q4;
    * a **callable** datum (``f(x, y, z, t)``, B42) is evaluated host-side at
      that patch's wall foot points, at the stage time ``t``, exactly as v1's
      ``_band_closure`` evaluates it, and lands as ``Constant`` for this sweep.
      It is the **fallback**, kept because it expresses waveforms
      :class:`Harmonic` does not; because the tables are rebuilt per ``apply``,
      a schedule is still followed per stage, one sample at a time.

    A datum that varies **across** a patch (A3's rotating wall) is refused
    rather than averaged: one ``GammaExpr`` serves the whole patch, so a spatial
    datum needs the ``Form`` tag Q25's OP-1 left out of scope.
    """
    import blockamr

    npatch = len(names)
    alpha = np.zeros(npatch)
    beta = np.zeros(npatch)
    # The two form tags are the compiled ones (`blockamr.GAMMA_*`, exported by
    # `wall_frame.cpp`) and never a second spelling of 0 and 1 here: a wrong tag
    # is a silently wrong waveform, not an error.
    form = np.full((npatch, ncomp), blockamr.GAMMA_CONSTANT, dtype=np.int32)
    param = np.zeros((npatch, ncomp, 4), dtype=np.float64)
    for patch, name in enumerate(names):
        a, b, datum = ibm_bc[name].robin()
        alpha[patch] = a
        beta[patch] = b
        if isinstance(datum, Harmonic):
            # The device form: four numbers, no evaluation here at all.
            form[patch, :] = blockamr.GAMMA_HARMONIC
            param[patch, :, :] = np.asarray(datum.params(), dtype=np.float64)
            continue
        param[patch, :, 0] = _patch_gamma(datum, name, patch, ncomp, wall_points, t)
    return blockamr.RobinData(alpha, beta, form, param)


def _patch_gamma(datum, name, patch, ncomp, wall_points, t):
    """One patch's ``gamma`` as ``(ncomp,)`` constants — see :func:`robin_data`."""
    if not callable(datum):
        return broadcast_gamma(datum, ncomp)
    points = wall_points(patch)
    if points.shape[0] == 0:
        return np.zeros(ncomp)
    values = gamma_rows(datum, points, t, ncomp)
    first = np.asarray(values[0], dtype=float)
    if not np.array_equal(values, np.broadcast_to(first, values.shape)):
        raise NotImplementedError(
            f"the wall datum on patch '{name}' varies across the patch, and a compiled "
            "gamma is one expression per (patch, component): a spatially varying surface "
            "value needs a Form tag that does not exist yet (plans/IBM/review.md §4 Q25 "
            "OP-1). A datum of the evaluation time alone is supported."
        )
    return first


@dataclass
class FixedValue:
    """Dirichlet: ``phi_w = value`` — the triple ``(1, 0, value)``.

    ``value`` is a scalar, a per-component sequence, a :class:`Harmonic`
    (evaluated device-side, in the kernel), or a callable ``f(x, y, z, t)``
    evaluated host-side per wall row (:func:`gamma_rows`).
    """

    value: float | Callable

    def robin(self):
        return (1.0, 0.0, self.value)


@dataclass
class FixedGradient:
    """Neumann: ``dphi/dn|_w = gradient`` — the triple ``(0, 1, gradient)``.

    ``gradient`` takes the same spellings as :class:`FixedValue`'s ``value``,
    including a :class:`Harmonic` and a callable, because ``robin()`` hands it
    through untouched.
    """

    gradient: float | Callable

    def robin(self):
        return (0.0, 1.0, self.gradient)


@dataclass
class Mixed:
    """OpenFOAM-style blend of the two, weighted by ``fraction``:
    ``(fraction, 1 - fraction, fraction*value + (1 - fraction)*gradient)``.

    ``fraction=1`` is bitwise :class:`FixedValue` and ``fraction=0`` is bitwise
    :class:`FixedGradient` (the dead term multiplies by an exact zero).
    """

    value: float
    gradient: float
    fraction: float

    def robin(self):
        if callable(self.value) or callable(self.gradient):
            raise NotImplementedError(
                "a callable (time-dependent) IBM BC datum is supported on FixedValue and "
                "FixedGradient, whose robin() hands gamma through to the row builder "
                "untouched; Mixed folds value and gradient into gamma here, eagerly, so a "
                "callable there needs a composed gamma that nothing asks for yet "
                "(review.md §4 Q22, scope note)."
            )
        f = float(self.fraction)
        alpha = f
        beta = 1.0 - f
        gamma = alpha * np.asarray(self.value, dtype=float) + beta * np.asarray(
            self.gradient, dtype=float
        )
        return (alpha, beta, gamma if gamma.ndim else float(gamma))
