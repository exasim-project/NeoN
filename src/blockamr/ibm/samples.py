# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The per-patch wall diagnostic — ``wall_samples`` and ``wall_gradient`` (T18).

``plans/IBM/verification.md`` §9 makes A1 + A4 the accuracy gate and then
records that **the gate's own third metric is unavailable**: A4's skin friction
at 45° lead, the only purely-wall quantity A4 has, raises
``ImportError: wall_gradient``. So does A1's wall flux and A5's ``t^-1/2`` wall
shear. This module is what those imports find.

What it reports, and what it deliberately does not
--------------------------------------------------

A ``ghostCell`` wall row *is* a closure: ``alpha phi_w + beta dphi/dn = gamma``
solved against the field at that cell's image point. Both halves of it are
numbers the method already has, so reporting them re-derives nothing:

===================  ======================================================
``point``            the body intercept ``x - sdf * n̂`` of each wall cell
``normal``           the unit normal there, into the fluid
``patch``            the owning body's id
``value``            ``phi_w`` — the field ON the surface
``normal_gradient``  ``dphi/dn|_w`` — the wall-normal derivative
===================  ======================================================

**There is no ``area`` and no full ``grad`` tensor here, and their absence is
the honest part.** ``test_ibm_validation_steady.py``'s ``_wall_samples_contract``
asks for both, and says why: a per-row *wetted area* is a cut-cell aperture in
all but name (T19's flux rows), and the viscous traction ``sigma.n`` needs
``d u_i/d x_j`` in full, not ``d phi/d n`` — on a rotating cylinder the two
differ by the ``u_theta/r`` term and the resulting torque is wrong by ~40 %.
Supplying either as an approximation would turn A1's flux and A3's torque from
*declared missing* into *silently wrong*, which is the failure mode this whole
design exists to avoid. They are left absent, so the rows that need them keep
raising ``AttributeError`` inside their own ``T18_FORCES`` xfail and stay
visibly owed.

That makes ``wall_gradient`` — A4's and A5's metric — the part that lands. Its
"area-averaged" is a **plain mean over the patch's wall rows**, for the same
reason: there are no areas to weight with. On A4 and A5 the immersed surface is
a *plane*, every wall row sits on it and the two averages coincide exactly; on a
curved patch they do not, and :func:`wall_gradient` says so in its docstring
rather than in a comment.

Where the numbers come from
---------------------------

Host-side, out of the method's own preprocessing — the image point's trilinear
donors and weights (:class:`~blockamr.ibm.ghost_cell.GhostCellData`), the wall
geometry (:class:`~blockamr.ibm.geometry.IbmGeometry`) and
:func:`~blockamr.schemes.boundary.ghost_cell.wall_closure`, which is the numpy
statement of the closure the compiled pairs inline. It is a **diagnostic**: it
stages the field through the host and is never on an ``evaluate`` path. A
compiled peer is the obvious next step and is not needed to answer the gate's
question.
"""

from dataclasses import dataclass

import numpy as np

from ..schemes.boundary.ghost_cell import wall_closure
from .bc import gamma_rows
from .classify import _patches
from .ghost_cell import GhostCell


@dataclass(frozen=True)
class WallSamples:
    """One patch's wall rows, in the order the level's boxes list them."""

    point: np.ndarray  # f64 (n, 3)
    normal: np.ndarray  # f64 (n, 3)
    patch: np.ndarray  # int (n,)
    value: np.ndarray  # f64 (n, ncomp)   phi_w
    normal_gradient: np.ndarray  # f64 (n, ncomp)   dphi/dn|_w

    def __len__(self):
        return int(self.point.shape[0])


def wall_samples(field, solution=None, t=0.0):
    """``{patch: WallSamples}`` for every immersed patch of ``field``'s mesh.

    ``solution`` is accepted and validated for symmetry with ``evaluate`` — the
    diagnostic is defined for the method that owns the reconstruction, and
    ``ghostCell`` is the only one that has one.
    """
    method = (solution or {}).get("ibm", GhostCell.name)
    if method not in (GhostCell.name, GhostCell):
        raise NotImplementedError(
            f"wall_samples is defined for the 'ghostCell' method, which reconstructs a wall "
            f"value and a wall gradient at every WALL cell; '{method}' does not, so there is "
            "nothing to sample. Pass solution={'ibm': 'ghostCell'}."
        )
    mesh = field.mesh
    names, _bodies = _patches(mesh.bodies)
    if not names:
        raise ValueError(
            "wall_samples was asked for the wall data of a mesh with no bodies; set "
            "mesh.bodies = {'<patch>': Cylinder(...)} first."
        )

    blocks = [_level_samples(field, lev, names, t) for lev in range(mesh.n_levels())]
    out = {}
    for p, name in enumerate(names):
        parts = [b for b in blocks if b is not None]
        point = _cat([b["point"][b["patch"] == p] for b in parts], 3)
        normal = _cat([b["normal"][b["patch"] == p] for b in parts], 3)
        value = _cat([b["value"][b["patch"] == p] for b in parts], field.ncomp)
        grad = _cat([b["grad"][b["patch"] == p] for b in parts], field.ncomp)
        out[name] = WallSamples(
            point=point,
            normal=normal,
            patch=np.full(point.shape[0], p, dtype=np.int32),
            value=value,
            normal_gradient=grad,
        )
    return out


def wall_gradient(field, patch, solution=None, t=0.0):
    """``dphi/dn|_w`` on ``patch``, averaged over its wall rows — ``(ncomp,)``.

    The mean is **unweighted**, because a per-row wetted area does not exist
    (see the module docstring). On a plane immersed surface — A4's oscillating
    wall and A5's impulsively started one — every row carries the same area and
    the unweighted mean IS the area-weighted one; on a curved patch it is not,
    and a torque or a total flux must not be built on it.
    """
    samples = wall_samples(field, solution=solution, t=t)
    if patch not in samples:
        raise ValueError(
            f"no immersed patch {patch!r} on this mesh; the patches are {sorted(samples)}."
        )
    grad = samples[patch].normal_gradient
    if grad.shape[0] == 0:
        raise ValueError(
            f"patch {patch!r} has no wall cell on this mesh, so its wall gradient is not "
            "defined; the body does not cut this grid."
        )
    return grad.mean(axis=0)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _level_samples(field, lev, names, t):
    """The wall rows of one level, as flat arrays, or ``None`` when it has none."""
    ibm = field.mesh.ibm
    geometries = ibm.geometry(lev)
    data = ibm.data(GhostCell, lev)
    if data.nrows == 0:
        return None

    at_wall = [g.depth == 1 for g in geometries]
    point = _cat([g.wall_point[s] for g, s in zip(geometries, at_wall)], 3)
    normal = _cat([g.normal[s] for g, s in zip(geometries, at_wall)], 3)
    sdf = _cat([g.sdf[s] for g, s in zip(geometries, at_wall)], None)
    patch = np.concatenate([g.patch[s] for g, s in zip(geometries, at_wall)]).astype(np.int64)
    assert point.shape[0] == data.nrows, (
        f"the level has {point.shape[0]} wall-layer cells but ghostCell preprocessed "
        f"{data.nrows}: the two were built from different grid generations."
    )

    phi_image = _image_value(field, lev, data)
    ncomp = field.ncomp
    robin = [field.ibm_bc[name].robin() for name in names]
    alpha = np.array([r[0] for r in robin], dtype=float)[patch]
    beta = np.array([r[1] for r in robin], dtype=float)[patch]
    gamma = np.zeros((point.shape[0], ncomp))
    for p, (_a, _b, datum) in enumerate(robin):
        sel = patch == p
        if sel.any():
            gamma[sel] = gamma_rows(datum, point[sel], t, ncomp)

    closure = wall_closure(alpha, beta, gamma, data.distance)
    return {
        "point": point,
        "normal": normal,
        "patch": patch,
        "sdf": sdf,
        "value": closure.value_linear[:, np.newaxis] * phi_image + closure.value_constant,
        "grad": closure.grad_linear[:, np.newaxis] * phi_image + closure.grad_constant,
    }


def _image_value(field, lev, data):
    """``phi`` at every wall row's image point — the trilinear sum, ``(n, ncomp)``.

    The donors are **global** indices, so the level is assembled once into a
    dense host array and indexed there. The boxes are staged **grown**: a wall
    cell on a box seam — or on the domain edge under a non-periodic face —
    interpolates from a ghost, which is precisely what the field's ``ngrow >= 1``
    exists to hold and what ``fill_patch`` has already filled. A donor outside
    even that is a diagnostic that cannot be answered rather than one that is
    answered wrongly, and it says so.
    """
    import blockamr

    mf = field.mf[lev]
    ncomp = field.ncomp
    blocks = []
    for mfi in blockamr.MFIterator(mf):
        ng = mf.n_grow()
        lo = np.asarray([int(v) for v in mfi.valid_box().small_end()]) - ng
        blocks.append((lo, np.asarray(mf.copy_grown_to_host(mfi))))
    lo_all = np.min([lo for lo, _a in blocks], axis=0)
    hi_all = np.max([lo + np.asarray(a.shape[:3]) for lo, a in blocks], axis=0)
    dense = np.full(tuple(hi_all - lo_all) + (ncomp,), np.nan)
    for lo, arr in blocks:
        off = lo - lo_all
        dense[
            off[0] : off[0] + arr.shape[0],
            off[1] : off[1] + arr.shape[1],
            off[2] : off[2] + arr.shape[2],
        ] = arr[..., :ncomp]

    idx = data.donor.astype(np.int64) - lo_all  # (n, K, 3)
    if (idx < 0).any() or (idx >= np.asarray(dense.shape[:3])).any():
        raise ValueError(
            "a wall row's image point interpolates from a cell outside every local box of "
            "this level, so the wall diagnostic cannot read it. That happens when the body "
            "touches a non-periodic domain face; move it inside or refine there."
        )
    donor_value = dense[idx[..., 0], idx[..., 1], idx[..., 2], :]  # (n, K, ncomp)
    if np.isnan(donor_value).any():
        raise ValueError(
            "a wall row's image point interpolates from a cell no local box covers (the "
            "level's boxes do not tile the region the donors reach); the wall diagnostic "
            "cannot read it."
        )
    return np.einsum("nk,nkc->nc", data.weight, donor_value)


def _cat(blocks, ncol):
    blocks = [b for b in blocks if b.shape[0]]
    if blocks:
        return np.concatenate(blocks)
    return np.zeros((0, ncol) if ncol else (0,))
