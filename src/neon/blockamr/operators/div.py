# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from jax import Array

import neon.blockamr as blockamr
from ..field import FaceField, CellField
from ..mesh import Mesh
from ..schemes.div_schemes import Upwind


@dataclass(frozen=True)
class BoxFluxData:
    """Raw FAB pointers for one box — face fluxes, 4D Fortran-order."""

    flux_x: Array  # (nx+1+2ng, ny+2ng, nz+2ng, 1) — x-face flux
    flux_y: Array  # (nx+2ng, ny+1+2ng, nz+2ng, 1) — y-face flux
    flux_z: Array  # (nx+2ng, ny+2ng, nz+1+2ng, 1) — z-face flux
    dh: Array  # (3,) — cell sizes
    stencil_width: int  # ghost cells needed by the stencil
    ngrow: int  # MultiFab ghost cells — controls output trim to valid region


def _box_flux_data_flatten(bfd):
    """JAX pytree flatten: arrays are children, (stencil_width, ngrow) is aux."""
    return (bfd.flux_x, bfd.flux_y, bfd.flux_z, bfd.dh), (bfd.stencil_width, bfd.ngrow)


def _box_flux_data_unflatten(aux, children):
    """JAX pytree unflatten."""
    stencil_width, ngrow = aux
    return BoxFluxData(*children, stencil_width=stencil_width, ngrow=ngrow)


jax.tree_util.register_pytree_node(
    BoxFluxData, _box_flux_data_flatten, _box_flux_data_unflatten
)


class Div:
    """Divergence operator: div(U * phi).

    Accepts a FaceField (multi-level) and a CellField (multi-level).
    Stencil computation delegated to scheme.compute().
    """

    def __init__(self, face_field, cell_field, coeff=1.0, scheme=None):
        self.face_field = face_field   # FaceField (unified)
        self.cell_field = cell_field   # CellField (unified)
        self.coeff = coeff
        self.scheme = scheme or Upwind()
        self._name = "Div"
        # Keep .field for DSL compatibility (Ddt extracts field from here)
        self.field = cell_field

    def __rmul__(self, scalar):
        obj = Div.__new__(Div)
        obj.face_field = self.face_field
        obj.cell_field = self.cell_field
        obj.field = self.cell_field
        obj.coeff = self.coeff * scalar
        obj.scheme = self.scheme
        obj._name = "Div"
        return obj

    def build_kernel(self, mfi, t, lev=0):
        """Return a scheme functor bound to this mfi's raw FAB data."""
        face_lev = self.face_field[lev]
        ngrow = self.cell_field.mf[lev].n_grow()
        flux_data = BoxFluxData(
            flux_x=face_lev[0].mf.grown_array(mfi),
            flux_y=face_lev[1].mf.grown_array(mfi),
            flux_z=face_lev[2].mf.grown_array(mfi),
            dh=jnp.array(self.cell_field.mesh.geom(lev).cell_size()),
            stencil_width=self.scheme.stencil_width,
            ngrow=ngrow,
        )
        return self.scheme.build_kernel(flux_data, coeff=self.coeff)


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
            coords.append(
                jnp.arange(n) * dx[e] + (prob_lo[e] + (lo[e] + offset) * dx[e])
            )

        X, Y, Z = jnp.meshgrid(*coords, indexing="ij")
        vel = vel_func(X, Y, Z, t)
        res.append(vel[d])

    comp.mf.copy_arrays(res)
    comp.fill_boundary()


def update_face_fluxes(face_fluxes, vel_func, geom, t):
    """Evaluate vel_func at face centers and store normal components."""
    dx = geom.cell_size()
    prob_lo = geom.prob_lo()

    for d in range(3):
        _fill_face_component(face_fluxes[d], d, vel_func, dx, prob_lo, t)


class FaceFluxUpdater:
    """Precomputes face-centre coordinates and batches velocity evaluation via JAX.

    Groups boxes by shape so that boxes with the same dimensions are stacked
    and evaluated in a single jax.vmap + jax.jit call. This handles AMR grids
    with variable box sizes at refinement boundaries.

    Workflow:
      1. Coordinates computed once at construction (static geometry).
      2. Velocity evaluated per shape-group via vmap+jit (one kernel per shape).
      3. Results written back via copy_arrays (single sync).
    """

    def __init__(self, face_fluxes, vel_func, geom):
        self.face_fluxes = face_fluxes
        self._vel_func = vel_func

        dx = geom.cell_size()
        prob_lo = geom.prob_lo()

        # _groups[d] = [(box_indices, all_X, all_Y, all_Z), ...]
        # Each entry groups boxes with the same shape for stacking.
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
                    coords.append(
                        jnp.arange(n) * dx[e]
                        + (prob_lo[e] + (lo[e] + offset) * dx[e])
                    )
                X, Y, Z = jnp.meshgrid(*coords, indexing="ij")
                shape_to_boxes.setdefault(shape_key, []).append((idx, X, Y, Z))
                idx += 1

            groups = []
            for entries in shape_to_boxes.values():
                indices = [e[0] for e in entries]
                all_X = jnp.stack([e[1] for e in entries])
                all_Y = jnp.stack([e[2] for e in entries])
                all_Z = jnp.stack([e[3] for e in entries])
                groups.append((indices, all_X, all_Y, all_Z))

            self._groups[d] = groups
            self._n_boxes[d] = idx

        @jax.jit
        def _batched_vel(all_X, all_Y, all_Z, t):
            return jax.vmap(lambda x, y, z: vel_func(x, y, z, t))(all_X, all_Y, all_Z)

        self._batched_vel = _batched_vel

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
    """Manages FaceFluxUpdater instances per AMR level, recreates after regrid."""

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
