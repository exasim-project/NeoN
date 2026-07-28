# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import blockamr


class FillPatchCellConservative:
    """Conservative interpolation for cell-centred scalars (periodic)."""

    def __call__(self, mesh, field, lev, time, target_mf=None):
        mf = target_mf or field.mf[lev]
        if mf is None:
            return
        if lev == 0:
            blockamr.fill_patch_single_level(
                mf, time, [field.mf[0]], [time], mesh.geom(0), 0, field.ncomp
            )
        elif field.mf[lev - 1] is None:
            mf.fill_boundary(mesh.geom(lev))
        else:
            bcs = [blockamr.periodic_bcrec()] * field.ncomp
            fine_mf = field.mf[lev]
            if fine_mf is None:
                # New level — no old fine data, interpolate from coarse only
                blockamr.interp_from_coarse_level(
                    mf,
                    time,
                    field.mf[lev - 1],
                    0,
                    0,
                    field.ncomp,
                    mesh.geom(lev - 1),
                    mesh.geom(lev),
                    mesh.ref_ratio(lev - 1),
                    blockamr.cell_cons_interp(),
                    bcs,
                )
            else:
                blockamr.fill_patch_two_levels(
                    mf,
                    time,
                    [field.mf[lev - 1]],
                    [time],
                    [fine_mf],
                    [time],
                    0,
                    0,
                    field.ncomp,
                    mesh.geom(lev - 1),
                    mesh.geom(lev),
                    mesh.ref_ratio(lev - 1),
                    blockamr.cell_cons_interp(),
                    bcs,
                )


class FillPatchSingleLevel:
    """Single-level fallback: just FillBoundary."""

    def __call__(self, mesh, field, lev, time, target_mf=None):
        mf = target_mf or field.mf[lev]
        mf.fill_boundary(mesh.geom(lev))


class FillPatchWithBC:
    """Fill patch with explicit boundary conditions (non-periodic).

    Performs inter-box ghost exchange via fill_boundary, then fills
    domain-boundary ghosts according to the supplied BoundaryCondition.
    """

    def __init__(self, bc):
        from .bc import fill_ghost_cells
        self._bc = bc
        self._fill = fill_ghost_cells

    def __call__(self, mesh, field, lev, time, target_mf=None):
        mf = target_mf or field.mf[lev]
        mf.fill_boundary(mesh.geom(lev))
        self._fill(mf, mesh.geom(lev), self._bc)
