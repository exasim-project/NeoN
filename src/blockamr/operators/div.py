# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import jax.numpy as jnp
import numpy as np

import blockamr
from blockamr.field import FaceField
from blockamr.schemes.div_schemes import Upwind


class Div:
    """Divergence operator: div(U * phi).

    Accepts a pre-built FaceField containing face fluxes.
    Stencil computation delegated to scheme.compute().
    """

    def __init__(self, face_fluxes, field, coeff=1.0, scheme=None):
        self.field = field
        self.coeff = coeff
        self.scheme = scheme or Upwind()
        self._name = "Div"

        if isinstance(face_fluxes, FaceField):
            self.face_fluxes = face_fluxes
        else:
            # Backward compat: accept vel_func, build fluxes at t=0
            ngrow = self.scheme.stencil_width
            self.face_fluxes = build_face_fluxes(
                face_fluxes, self.field.box, self.field.dm,
                self.field.geom, ngrow=ngrow, t=0.0,
                max_size=self.field.max_size,
            )

    def __rmul__(self, scalar):
        obj = Div.__new__(Div)
        obj.face_fluxes = self.face_fluxes
        obj.field = self.field
        obj.coeff = self.coeff * scalar
        obj.scheme = self.scheme
        obj._name = "Div"
        return obj

    def build_kernel(self, mfi, t):
        """Return a scheme functor bound to this mfi's fluxes."""
        w = self.scheme.stencil_width
        dh = jnp.array(self.field.geom.cell_size())
        fluxes = []
        for dim in range(3):
            flux = jnp.array(
                self.face_fluxes[dim].mf.grown_array(mfi)[:, :, :, 0]
            )
            sl = [slice(None)] * 3
            sl[dim] = slice(w, -w) if w > 0 else slice(None)
            fluxes.append(flux[tuple(sl)])
        return self.scheme.build_kernel(fluxes, dh, coeff=self.coeff)


def build_face_fluxes(vel_func, box, dm, geom, ngrow, t, max_size=32):
    """Build a FaceField containing normal velocity at face centers."""
    ff = FaceField(box, dm, geom, ncomp=1, ngrow=ngrow, max_size=max_size)
    update_face_fluxes(ff, vel_func, geom, t)
    return ff


def _fill_face_component(comp, d, vel_func, dx, prob_lo, t):
    """Fill one face-field component (direction *d*) with the normal velocity."""
    for mfi in blockamr.MFIterator(comp.mf):
        arr = comp.mf.array(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        nx, ny, nz = arr.shape[:3]

        coords = []
        for e in range(3):
            n = [nx, ny, nz][e]
            if e == d:
                # Face positions (integer grid points, no +0.5)
                coords.append(
                    np.array([prob_lo[e] + (lo[e] + i) * dx[e] for i in range(n)])
                )
            else:
                # Cell centers (+0.5)
                coords.append(
                    np.array(
                        [prob_lo[e] + (lo[e] + i + 0.5) * dx[e] for i in range(n)]
                    )
                )

        X, Y, Z = np.meshgrid(*coords, indexing="ij")
        vel = vel_func(X, Y, Z, t)
        arr[:, :, :, 0] = vel[d]

    comp.fill_boundary()


def update_face_fluxes(face_fluxes, vel_func, geom, t):
    """Evaluate vel_func at face centers and store normal components."""
    dx = geom.cell_size()
    prob_lo = geom.prob_lo()

    for d in range(3):
        _fill_face_component(face_fluxes[d], d, vel_func, dx, prob_lo, t)
