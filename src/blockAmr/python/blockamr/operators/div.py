# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import numpy as np

import blockamr
from ..field import FaceField
from ..mesh import Mesh
from ..schemes.div_schemes import Upwind
from ..dsl.eqterm import EqTerm


class Div(EqTerm):
    """Divergence operator: div(U * phi), from a FaceField and a CellField."""

    kind = "spatial"
    _scheme_operator = "div"

    def __init__(self, face_field, cell_field, coeff=1.0, scheme=None):
        super().__init__(cell_field, coeff=coeff, coefficient=face_field, scheme=scheme or Upwind())
        self.face_field = face_field
        self.cell_field = cell_field
        self._scheme_explicit = scheme is not None

    @property
    def scheme_key(self):
        return (
            f"div({self._named(self.face_field, 'face-flux')},"
            f"{self._named(self.cell_field, 'cell')})"
        )


def build_face_fluxes(vel_func, box, dm, geom, ngrow, t, max_size=32, memory="default"):
    """Build a FaceField containing normal velocity at face centers."""
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    mesh = Mesh(ba, dm, geom)
    ff = FaceField(mesh, ncomp=1, ngrow=ngrow, memory=memory)
    update_face_fluxes(ff[0], vel_func, geom, t)
    return ff


def _fill_face_component(comp, d, vel_func, dx, prob_lo, t):
    """Fill one face-field component (direction *d*) with the normal velocity."""
    res = []
    for mfi in blockamr.MFIterator(comp.mf):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1

        coords = []
        for e in range(3):
            n = [nx, ny, nz][e]
            offset = 0.0 if e == d else 0.5
            coords.append(np.arange(n) * dx[e] + (prob_lo[e] + (lo[e] + offset) * dx[e]))

        X, Y, Z = np.meshgrid(*coords, indexing="ij")
        vel = vel_func(X, Y, Z, t)
        res.append(np.broadcast_to(vel[d], X.shape))

    comp.mf.copy_arrays(res)
    comp.fill_boundary()


def update_face_fluxes(face_fluxes, vel_func, geom, t):
    """Evaluate vel_func at face centers and store normal components."""
    dx = geom.cell_size()
    prob_lo = geom.prob_lo()

    for d in range(3):
        _fill_face_component(face_fluxes[d], d, vel_func, dx, prob_lo, t)


class FaceFluxUpdater:
    """Precomputes face-centre coordinates and batches the velocity evaluation.

    Boxes of equal shape are stacked and evaluated in one vectorised ``vel_func`` call.
    """

    def __init__(self, face_fluxes, vel_func, geom):
        self.face_fluxes = face_fluxes
        self._vel_func = vel_func

        dx = geom.cell_size()
        prob_lo = geom.prob_lo()

        self._groups = {}
        self._n_boxes = {}

        for d in range(3):
            shape_to_boxes = {}
            idx = 0
            for mfi in blockamr.MFIterator(face_fluxes[d].mf):
                bx = mfi.valid_box()
                lo = bx.small_end()
                hi = bx.big_end()
                nx = hi[0] - lo[0] + 1
                ny = hi[1] - lo[1] + 1
                nz = hi[2] - lo[2] + 1
                shape_key = (nx, ny, nz)

                coords = []
                for e in range(3):
                    n = [nx, ny, nz][e]
                    offset = 0.0 if e == d else 0.5
                    coords.append(np.arange(n) * dx[e] + (prob_lo[e] + (lo[e] + offset) * dx[e]))
                X, Y, Z = np.meshgrid(*coords, indexing="ij")
                shape_to_boxes.setdefault(shape_key, []).append((idx, X, Y, Z))
                idx += 1

            groups = []
            for entries in shape_to_boxes.values():
                indices = [e[0] for e in entries]
                all_X = np.stack([e[1] for e in entries])
                all_Y = np.stack([e[2] for e in entries])
                all_Z = np.stack([e[3] for e in entries])
                groups.append((indices, all_X, all_Y, all_Z))

            self._groups[d] = groups
            self._n_boxes[d] = idx

    def _batched_vel(self, all_X, all_Y, all_Z, t):
        vel = self._vel_func(all_X, all_Y, all_Z, t)
        return tuple(np.broadcast_to(v, all_X.shape) for v in vel)

    def update(self, t):
        """Evaluate velocity at time *t* and write into face fluxes."""
        for d in range(3):
            results = [None] * self._n_boxes[d]

            for indices, all_X, all_Y, all_Z in self._groups[d]:
                all_u, all_v, all_w = self._batched_vel(all_X, all_Y, all_Z, t)
                vel_d = (all_u, all_v, all_w)[d]
                for i, box_idx in enumerate(indices):
                    results[box_idx] = vel_d[i]

            self.face_fluxes[d].mf.copy_arrays(results)
            self.face_fluxes[d].fill_boundary()


class AmrFaceFluxUpdater:
    """Manages FaceFluxUpdater instances per AMR level."""

    def __init__(self, face_vel, vel_func, mesh):
        self._face_vel = face_vel
        self._vel_func = vel_func
        self._mesh = mesh
        self._updaters = {}
        self._rebuild()

    def _rebuild(self):
        self._updaters = {}
        for lev in range(self._mesh.n_levels()):
            if self._face_vel[lev] is not None:
                self._updaters[lev] = FaceFluxUpdater(
                    self._face_vel[lev], self._vel_func, self._mesh.geom(lev)
                )

    def update(self, t):
        """Evaluate velocity at time *t* across all AMR levels."""
        if len(self._updaters) != self._mesh.n_levels():
            self._rebuild()
        for lev in self._updaters:
            self._updaters[lev].update(t)
