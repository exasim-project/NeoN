# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Free-slip / symmetry wall ghost-fill (native BC code 3).

A slip wall imposes no penetration (the velocity component normal to the face is
reflected with a sign flip → zero at the face) and zero tangential shear (the
tangential components are copied → zero gradient). Contrasted against a no-slip
wall, which drives every component to zero at the face.
"""

import numpy as np

import blockamr
from blockamr.bc import BoundaryCondition, NeumannBC, SlipBC, noSlip
from blockamr.field import CellField
from blockamr.fillpatch import FillPatchWithBC
from blockamr.mesh import Mesh

N = 8
NG = 1
U_INT = (1.0, 0.5, 0.3)  # a generic interior velocity (u, v, w)


def _make_mesh():
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 0])  # non-periodic all round
    ba = blockamr.BoxArray(box)
    ba.max_size(N)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom), geom


def _fill_vector(mf, vec):
    for mfi in blockamr.MFIterator(mf):
        arr = np.array(mf.copy_grown_to_host(mfi), order="F")
        for c in range(3):
            arr[..., c] = vec[c]
        mf.copy_grown_from(mfi, np.asfortranarray(arr))


def _grown(mf):
    for mfi in blockamr.MFIterator(mf):
        return np.array(mf.copy_grown_to_host(mfi), order="F")


def test_slip_wall_reflects_normal_copies_tangential(blockamr_session):
    """y-face slip: normal component (v) flips sign, tangential (u,w) copied."""
    mesh, geom = _make_mesh()
    # slip on the y-faces (normal axis = 1); other faces irrelevant here
    bc = BoundaryCondition(
        lo=[NeumannBC(), SlipBC(), NeumannBC()],
        hi=[NeumannBC(), SlipBC(), NeumannBC()],
    )
    U = CellField(mesh, ncomp=3, ngrow=NG, name="U",
                  fill_patch=FillPatchWithBC(bc))
    _fill_vector(U.mf[0], U_INT)
    U.fill_patch(0, 0.0)

    arr = _grown(U.mf[0])
    ng = NG
    # y_lo ghost row: normal component v -> -0.5, tangential u,w copied
    assert np.isclose(arr[ng, 0, ng, 0], U_INT[0])   # u tangential: copied
    assert np.isclose(arr[ng, 0, ng, 1], -U_INT[1])  # v normal: reflect-odd
    assert np.isclose(arr[ng, 0, ng, 2], U_INT[2])   # w tangential: copied
    # y_hi ghost row: same behavior
    assert np.isclose(arr[ng, -1, ng, 0], U_INT[0])
    assert np.isclose(arr[ng, -1, ng, 1], -U_INT[1])
    assert np.isclose(arr[ng, -1, ng, 2], U_INT[2])


def test_slip_differs_from_noslip(blockamr_session):
    """Sanity: no-slip drives every component to reflect about zero, unlike slip
    which preserves the tangential components."""
    mesh, geom = _make_mesh()
    bc = BoundaryCondition(
        lo=[NeumannBC(), noSlip(), NeumannBC()],
        hi=[NeumannBC(), noSlip(), NeumannBC()],
    )
    U = CellField(mesh, ncomp=3, ngrow=NG, name="U",
                  fill_patch=FillPatchWithBC(bc))
    _fill_vector(U.mf[0], U_INT)
    U.fill_patch(0, 0.0)

    arr = _grown(U.mf[0])
    ng = NG
    # no-slip (fixedValue 0) -> ghost = -interior for ALL components
    assert np.isclose(arr[ng, 0, ng, 0], -U_INT[0])  # tangential reflected too
    assert np.isclose(arr[ng, 0, ng, 1], -U_INT[1])
    assert np.isclose(arr[ng, 0, ng, 2], -U_INT[2])
