# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Interpolate cell-centred velocity to face-centred flux field."""

import blockamr


def interpolate(U_cell, face_field):
    """Interpolate a CellField to a FaceField: ``phi_face[i+1/2] = 0.5*(U[i,d]+U[i+1,d])``.

    ncomp=1 uses component 0 for every direction.

    Parameters
    ----------
    U_cell : CellField
        Cell-centred velocity with ghosts already filled.
    face_field : FaceField
        Face-centred field to fill — MUTATED.
    """
    mesh = U_cell.mesh
    for lev in range(mesh.n_levels()):
        _interpolate_level(U_cell, face_field, lev)


def _interpolate_level(U_cell, face_field, lev):
    """Interpolate cell to face for a single level."""
    cell_ng = U_cell.ngrow
    cell_arrs = U_cell.mf[lev].grown_arrays()
    ncomp = U_cell.ncomp
    geom = U_cell.mesh.geom(lev)

    for d in range(3):
        face_mf = face_field[lev][d].mf
        face_valid_arrs = face_mf.arrays()

        comp = d if ncomp >= 3 else 0
        results = []
        for bi in range(len(face_valid_arrs)):
            c = cell_arrs[bi][:, :, :, comp]

            face_ng = face_mf.n_grow()
            nf = [int(face_valid_arrs[bi].shape[ax]) - 2 * face_ng for ax in range(3)]

            sl_lo = [slice(None)] * 3
            sl_hi = [slice(None)] * 3
            for ax in range(3):
                if ax == d:
                    sl_lo[ax] = slice(cell_ng - 1, cell_ng - 1 + nf[ax])
                    sl_hi[ax] = slice(cell_ng, cell_ng + nf[ax])
                else:
                    sl_lo[ax] = slice(cell_ng, cell_ng + nf[ax])
                    sl_hi[ax] = slice(cell_ng, cell_ng + nf[ax])

            results.append(0.5 * (c[tuple(sl_lo)] + c[tuple(sl_hi)]))

        face_mf.copy_arrays(results)
        face_mf.fill_boundary(geom)
