# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import neon.blockamr as blockamr


class FillPatchCellConservative:
    """Conservative interpolation for cell-centred scalars (periodic)."""

    def __call__(self, mesh, field, lev, time, target_mf=None):
        mf = target_mf or field.mf[lev]
        if lev == 0:
            blockamr.fill_patch_single_level(
                mf, time, [field.mf[0]], [time], mesh.geom(0), 0, field.ncomp
            )
        else:
            bcs = [blockamr.periodic_bcrec()] * field.ncomp
            blockamr.fill_patch_two_levels(
                mf,
                time,
                [field.mf[lev - 1]],
                [time],
                [field.mf[lev]],
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
