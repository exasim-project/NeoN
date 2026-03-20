# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from jax import Array

import blockamr
from blockamr.field import FaceField
from blockamr.schemes.div_schemes import Upwind


@dataclass(frozen=True)
class BoxFluxData:
    """Raw FAB pointers for one box — face fluxes, 4D Fortran-order."""

    flux_x: Array  # (nx+1+2ng, ny+2ng, nz+2ng, 1) — x-face flux
    flux_y: Array  # (nx+2ng, ny+1+2ng, nz+2ng, 1) — y-face flux
    flux_z: Array  # (nx+2ng, ny+2ng, nz+1+2ng, 1) — z-face flux
    dh: Array  # (3,) — cell sizes
    stencil_width: int  # ghost width for stencil trim


def _box_flux_data_flatten(bfd):
    """JAX pytree flatten: arrays are children, stencil_width is aux."""
    return (bfd.flux_x, bfd.flux_y, bfd.flux_z, bfd.dh), bfd.stencil_width


def _box_flux_data_unflatten(aux, children):
    """JAX pytree unflatten."""
    return BoxFluxData(*children, stencil_width=aux)


jax.tree_util.register_pytree_node(
    BoxFluxData, _box_flux_data_flatten, _box_flux_data_unflatten
)


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
        self.face_fluxes = face_fluxes

    def __rmul__(self, scalar):
        obj = Div.__new__(Div)
        obj.face_fluxes = self.face_fluxes
        obj.field = self.field
        obj.coeff = self.coeff * scalar
        obj.scheme = self.scheme
        obj._name = "Div"
        return obj

    def build_kernel(self, mfi, t):
        """Return a scheme functor bound to this mfi's raw FAB data."""
        flux_data = BoxFluxData(
            flux_x=self.face_fluxes[0].mf.grown_array(mfi),
            flux_y=self.face_fluxes[1].mf.grown_array(mfi),
            flux_z=self.face_fluxes[2].mf.grown_array(mfi),
            dh=jnp.array(self.field.geom.cell_size()),
            stencil_width=self.scheme.stencil_width,
        )
        return self.scheme.build_kernel(flux_data, coeff=self.coeff)


def build_face_fluxes(vel_func, box, dm, geom, ngrow, t, max_size=32, memory="default"):
    """Build a FaceField containing normal velocity at face centers."""
    ff = FaceField(box, dm, geom, ncomp=1, ngrow=ngrow, max_size=max_size, memory=memory)
    update_face_fluxes(ff, vel_func, geom, t)
    return ff


def _fill_face_component(comp, d, vel_func, dx, prob_lo, t):
    """Fill one face-field component (direction *d*) with the normal velocity."""
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
            coords.append(
                jnp.array([prob_lo[e] + (lo[e] + i + offset) * dx[e] for i in range(n)])
            )

        X, Y, Z = jnp.meshgrid(*coords, indexing="ij")
        vel = vel_func(X, Y, Z, t)
        comp.mf.copy_from(mfi, vel[d])

    comp.fill_boundary()


def update_face_fluxes(face_fluxes, vel_func, geom, t):
    """Evaluate vel_func at face centers and store normal components."""
    dx = geom.cell_size()
    prob_lo = geom.prob_lo()

    for d in range(3):
        _fill_face_component(face_fluxes[d], d, vel_func, dx, prob_lo, t)


class FaceFluxUpdater:
    """Precomputes face-centre coordinates and batches velocity evaluation via JAX.

    Replaces the per-box Python loop in update_face_fluxes with:
      1. Coordinates computed once at construction (static geometry).
      2. Velocity evaluated in a single jax.vmap + jax.jit call per dimension.
      3. Only copy_from (C++ writeback) loops over boxes.
    """

    def __init__(self, face_fluxes, vel_func, geom):
        self.face_fluxes = face_fluxes
        self._vel_func = vel_func

        dx = geom.cell_size()
        prob_lo = geom.prob_lo()

        # Precompute and stack coordinate meshgrids per dimension.
        # _batched_coords[d] = (all_X, all_Y, all_Z) each (n_boxes, nx, ny, nz)
        self._batched_coords = {}

        for d in range(3):
            Xs, Ys, Zs = [], [], []
            for mfi in blockamr.MFIterator(face_fluxes[d].mf):
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
                    coords.append(
                        jnp.array(
                            [prob_lo[e] + (lo[e] + i + offset) * dx[e] for i in range(n)]
                        )
                    )
                X, Y, Z = jnp.meshgrid(*coords, indexing="ij")
                Xs.append(X)
                Ys.append(Y)
                Zs.append(Z)

            self._batched_coords[d] = (jnp.stack(Xs), jnp.stack(Ys), jnp.stack(Zs))

        # JIT + vmap: single XLA kernel evaluates velocity on all boxes at once.
        @jax.jit
        def _batched_vel(all_X, all_Y, all_Z, t):
            return jax.vmap(lambda x, y, z: vel_func(x, y, z, t))(all_X, all_Y, all_Z)

        self._batched_vel = _batched_vel

    def update(self, t):
        """Evaluate velocity at time *t* and write into face fluxes."""
        for d in range(3):
            all_X, all_Y, all_Z = self._batched_coords[d]
            all_u, all_v, all_w = self._batched_vel(all_X, all_Y, all_Z, t)
            all_vel = (all_u, all_v, all_w)

            i = 0
            for mfi in blockamr.MFIterator(self.face_fluxes[d].mf):
                self.face_fluxes[d].mf.copy_from(mfi, all_vel[d][i])
                i += 1

            self.face_fluxes[d].fill_boundary()
