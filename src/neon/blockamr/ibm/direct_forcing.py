# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Direct-forcing IBM strategy: pin the velocity to a wall value in solid
cells each step. Mesh-owned data, per-field method (API doc §6)."""

from dataclasses import dataclass, field as _dc_field

import jax.numpy as jnp
import numpy as np

import neon.blockamr as blockamr


@dataclass
class DirectForcingData:
    """Precomputed per-(level, box) solid masks plus the reaction-force
    history. ``masks[lev][bi]`` is a boolean array over box ``bi``'s valid
    (non-ghost) cells, ``True`` inside the body. ``force_history`` is a list
    of ``(t, Fx, Fy, Fz)`` entries, one per :func:`DirectForcing.apply` call
    — it must survive a regrid (masks are spatial, force history is a time
    series), so mesh regrid rebuilds carry the same list object forward.
    """

    masks: list
    force_history: list = _dc_field(default_factory=list)


class DirectForcing:
    """Direct-forcing IBM strategy. Stateless — everything it needs travels
    through the ``mesh``/``data`` arguments, so the class itself (not an
    instance) is both what ``mesh.build_ibm([DirectForcing])`` expects and
    what ``IBM.lookup("directForcing")`` returns.
    """

    @staticmethod
    def build_data(mesh, body):
        """Per-(level, box) boolean masks: True in valid cells inside the body.

        A cell is solid when its centre's distance from the body axis
        (measured in the plane perpendicular to ``body.axis``) is below
        ``body.radius``. Ghost cells are not represented — they are excluded
        by construction (masks cover only the valid box) and are always
        passed through unchanged by :func:`apply`.
        """
        center = [float(c) for c in body.centre]
        radius = float(body.radius)
        axis = int(body.axis)
        plane = [a for a in range(3) if a != axis]

        masks = []
        for lev in range(mesh.n_levels()):
            geom = mesh.geom(lev)
            dx = [float(v) for v in geom.cell_size()]
            lo = [float(v) for v in geom.prob_lo()]
            # Zero-ghost scratch MultiFab purely for the box layout — mask
            # data is independent of any registered field's ghost width.
            scratch = blockamr.MultiFab(mesh.box_array(lev), mesh.dm(lev), 1, 0)
            boxes = [mfi.valid_box() for mfi in blockamr.MFIterator(scratch)]

            lev_masks = []
            for box in boxes:
                small = list(box.small_end())
                big = list(box.big_end())
                shape = [big[d] - small[d] + 1 for d in range(3)]
                gi = [np.arange(shape[d]) + small[d] for d in range(3)]
                cc = [lo[d] + (gi[d] + 0.5) * dx[d] for d in range(3)]
                mesh_c = np.meshgrid(cc[0], cc[1], cc[2], indexing="ij")
                d2 = (mesh_c[plane[0]] - center[plane[0]]) ** 2 + (
                    mesh_c[plane[1]] - center[plane[1]]
                ) ** 2
                lev_masks.append(jnp.asarray(d2 < radius * radius))
            masks.append(lev_masks)
        return DirectForcingData(masks=masks)

    @staticmethod
    def apply(cell_field, dt, t, data, u_body=(0.0, 0.0, 0.0)):
        """Reset the velocity in solid cells to the (stationary) wall value.

        Records the reaction force on the body (momentum removed per unit
        time, ``F = (rho/dt) * sum_solid(U_before - u_body) * cell_vol``,
        rho=1) in ``data.force_history``.
        """
        masks = data.masks
        u_vec = jnp.asarray(u_body).reshape(1, 1, 1, 3)
        force = jnp.zeros(3)
        mesh = cell_field.mesh
        for lev in range(mesh.n_levels()):
            mf = cell_field.mf[lev]
            ng = mf.n_grow()
            dx = [float(v) for v in mesh.geom(lev).cell_size()]
            cell_vol = dx[0] * dx[1] * dx[2]
            grown = mf.grown_arrays()
            results = []
            for bi, g in enumerate(grown):
                nx, ny, nz = (int(s) for s in g.shape[:3])
                m = masks[lev][bi][..., None]
                sl = (slice(ng, nx - ng), slice(ng, ny - ng), slice(ng, nz - ng))
                valid = g[sl[0], sl[1], sl[2], :]
                force = force + jnp.sum(jnp.where(m, valid - u_vec, 0.0), axis=(0, 1, 2)) * cell_vol
                new_valid = jnp.where(m, u_vec, valid)
                results.append(g.at[sl[0], sl[1], sl[2], :].set(new_valid))
            mf.copy_grown_arrays(results)
        fvec = force / dt
        data.force_history.append((t, float(fvec[0]), float(fvec[1]), float(fvec[2])))

    @staticmethod
    def force_history(data):
        """Accessor: the recorded reaction-force time series (see
        ``DirectForcingData.force_history``)."""
        return data.force_history
