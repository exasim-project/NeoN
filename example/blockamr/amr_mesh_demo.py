#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""
AMR mesh demo: create a 2-level AMR mesh, initialise a Gaussian field,
regrid based on gradient tagging, and write a multi-level plotfile.

Usage:
    python amr_mesh_demo.py [--ncell 64] [--plotdir plt_amr]
"""

import argparse

import numpy as np

import blockamr
from blockamr.mesh import AmrMesh
from blockamr.field import CellField
from blockamr.fillpatch import FillPatchCellConservative


def gaussian(x, y, z, cx=0.5, cy=0.5, cz=0.5, sigma=0.1):
    r2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    return np.exp(-r2 / (2.0 * sigma**2))


def fill_gaussian(phi, mesh):
    """Fill phi with a Gaussian on all active levels."""
    for lev in range(mesh.n_levels()):
        if phi.mf[lev] is None:
            continue
        geom = mesh.geom(lev)
        dx = geom.cell_size()
        lo = geom.prob_lo()
        for mfi in blockamr.MFIterator(phi.mf[lev]):
            bx = mfi.valid_box()
            sml = bx.small_end()
            big = bx.big_end()
            nx = big[0] - sml[0] + 1
            ny = big[1] - sml[1] + 1
            nz = big[2] - sml[2] + 1

            x = np.linspace(
                lo[0] + (sml[0] + 0.5) * dx[0],
                lo[0] + (big[0] + 0.5) * dx[0],
                nx,
            )
            y = np.linspace(
                lo[1] + (sml[1] + 0.5) * dx[1],
                lo[1] + (big[1] + 0.5) * dx[1],
                ny,
            )
            z = np.linspace(
                lo[2] + (sml[2] + 0.5) * dx[2],
                lo[2] + (big[2] + 0.5) * dx[2],
                nz,
            )
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            vals = gaussian(X, Y, Z)
            phi.mf[lev].copy_from(mfi, vals[:, :, :, np.newaxis])


def tag_gradient(phi, threshold=0.1):
    """Return a tagging function that tags cells where |grad(phi)| > threshold.

    Uses vectorised numpy gradient and GPU-safe batch set_tags.
    """

    def _tag(lev, tags, time, ngrow):
        if phi.mf[lev] is None:
            return
        geom = phi.mesh.geom(lev)
        dx = geom.cell_size()

        # Collect all patch data up front (avoids nested MFIter)
        patch_data = {}
        for mfi in blockamr.MFIterator(phi.mf[lev]):
            lo = tuple(mfi.valid_box().small_end())
            patch_data[lo] = phi.mf[lev].copy_to_host(mfi)

        # Now iterate tags and apply gradient-based tagging
        for tbi in blockamr.TagBoxIterator(tags):
            bx = tbi.valid_box()
            sml = bx.small_end()
            big = bx.big_end()
            nx = big[0] - sml[0] + 1
            ny = big[1] - sml[1] + 1
            nz = big[2] - sml[2] + 1

            arr = patch_data.get(tuple(sml))
            if arr is None:
                continue

            # Compute gradient magnitude with numpy (vectorised)
            data = arr[:, :, :, 0]
            gx = np.zeros_like(data)
            gy = np.zeros_like(data)
            gz = np.zeros_like(data)
            if nx > 2:
                gx[1:-1, :, :] = np.abs(data[2:, :, :] - data[:-2, :, :]) / (
                    2.0 * dx[0]
                )
            if ny > 2:
                gy[:, 1:-1, :] = np.abs(data[:, 2:, :] - data[:, :-2, :]) / (
                    2.0 * dx[1]
                )
            if nz > 2:
                gz[:, :, 1:-1] = np.abs(data[:, :, 2:] - data[:, :, :-2]) / (
                    2.0 * dx[2]
                )
            grad_mag = gx + gy + gz

            # Build mask and tag in one GPU-safe batch
            mask = (grad_mag > threshold).astype(np.int32)
            tbi.set_tags(mask)

    return _tag


def run(ncell=32, plotdir="plt_amr"):
    box = blockamr.Box([0, 0, 0], [ncell - 1, ncell - 1, ncell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])

    amr_info = blockamr.AmrInfo()
    amr_info.max_level = 3
    amr_info.set_ref_ratio(0, 2)
    amr_info.set_max_grid_size(0, 32)
    amr_info.set_blocking_factor(0, 8)

    # Create mesh + field
    mesh = AmrMesh(geom, amr_info)
    phi = CellField(
        mesh, name="phi", ncomp=1, ngrow=2, fill_patch=FillPatchCellConservative()
    )

    # Initialize — creates level 0
    mesh.init_from_scratch(0.0)
    fill_gaussian(phi, mesh)

    print(
        f"After init: {mesh.n_levels()} level(s), finest_level={mesh.finest_level()}"
    )

    # Regrid — tag cells with gradient and add level 1
    mesh.regrid(0.0, tag=tag_gradient(phi, threshold=1.5))
    fill_gaussian(phi, mesh)
    mesh.regrid(0.0, tag=tag_gradient(phi, threshold=1.5))
    fill_gaussian(phi, mesh)
    mesh.regrid(0.0, tag=tag_gradient(phi, threshold=1.5))
    fill_gaussian(phi, mesh)

    print(
        f"After regrid: {mesh.n_levels()} level(s), finest_level={mesh.finest_level()}"
    )

    # Write multi-level plotfile
    mesh.write_plotfile(plotdir, phi, 0.0)
    print(f"Wrote plotfile to {plotdir}")


def main():
    parser = argparse.ArgumentParser(description="AMR mesh demo")
    parser.add_argument(
        "--ncell", type=int, default=32, help="Cells per side (level 0)"
    )
    parser.add_argument(
        "--plotdir", type=str, default="plt_amr", help="Plotfile directory"
    )
    args = parser.parse_args()

    blockamr.runtime(lambda: run(ncell=args.ncell, plotdir=args.plotdir))


if __name__ == "__main__":
    main()
