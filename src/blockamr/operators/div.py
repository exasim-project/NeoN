# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import jax.numpy as jnp

import blockamr
from ..field import FaceField
from ..flattened_boxes import FlattenedFaceBoxes
from ..mesh import Mesh
from ..schemes.div_schemes import Upwind
from ..dsl.eqterm import EqTerm


class Div(EqTerm):
    """Divergence operator: div(U * phi).

    Accepts a FaceField (multi-level) and a CellField (multi-level).
    """

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

    def build_kernel_3d(self, ctx, t):
        """Build a 3D spatial kernel from TiledContext.

        Extracts face data from self.face_field, constructs FlatFaceBoxed
        with per-box offsets (box_id is a dummy — replaced per-tile inside
        the Pallas kernel by parallel_for).

        Returns an eqx.Module with __call__(box_id, i, j, k, phi) → scalar.
        """
        import jax.numpy as jnp
        from ..flat_refs import FlatFaceBoxed
        from ..flattened_boxes import FlattenedFaceBoxes

        lev = ctx.lev
        face_lev = self.face_field[lev]
        ng_face = face_lev[0].mf.n_grow()
        face_fb = FlattenedFaceBoxes.from_face_field(self.face_field, lev)
        face_metas = [face_lev[d].mf.fab_metadata() for d in range(3)]
        n_boxes = len(face_metas[0])

        # Per-box face offsets and strides
        all_offs = []
        all_strs = []
        for d in range(3):
            offs = []
            strs = []
            for b in range(n_boxes):
                _, fNx, fNy, _, _ = face_metas[d][b]
                offs.append(
                    int(face_metas[d][b][0]) + ng_face + fNx * ng_face + fNx * fNy * ng_face
                )
                strs.append([1, fNx, fNx * fNy])
            all_offs.append(offs)
            all_strs.append(strs)

        mb = 1
        while mb < n_boxes:
            mb <<= 1
        for d in range(3):
            while len(all_offs[d]) < mb:
                all_offs[d].append(all_offs[d][0])
                all_strs[d].append(all_strs[d][0])

        box_id = jnp.int32(0)
        face = FlatFaceBoxed(
            face_fb.bufs[0],
            face_fb.bufs[1],
            face_fb.bufs[2],
            jnp.array(all_offs[0], dtype=jnp.int32),
            jnp.array(all_offs[1], dtype=jnp.int32),
            jnp.array(all_offs[2], dtype=jnp.int32),
            jnp.array(all_strs[0], dtype=jnp.int32),
            jnp.array(all_strs[1], dtype=jnp.int32),
            jnp.array(all_strs[2], dtype=jnp.int32),
            box_id,
        )
        return self.scheme.build_spatial_kernel(face=face, dh=ctx.dh, coeff=self.coeff)

    def build_kernel(self, bucket, t):
        """Build a cell-level kernel for a bucket of boxes.

        Returns an eqx.Module kernel with __call__(phi) → scalar
        and for_box(bucket, box_idx) for per-box rebinding.

        Caches face_offsets across calls — only the face buffer data
        (traced) changes each timestep; the offset layout is stable
        until regrid.
        """
        lev = bucket.lev
        face_fb = FlattenedFaceBoxes.from_face_field(self.face_field, lev)

        # Cache face offsets keyed on (box_indices, max_boxes, lev)
        cache_key = (bucket.box_indices, bucket.max_boxes, lev)
        if not hasattr(self, "_face_offset_cache"):
            self._face_offset_cache = {}
        if cache_key not in self._face_offset_cache:
            n_pad = bucket.max_boxes - len(bucket.box_indices)
            face_offsets = tuple(
                jnp.array(
                    [int(face_fb.offsets[d][mf_idx]) for mf_idx in bucket.box_indices]
                    + [0] * n_pad,
                    dtype=jnp.int32,
                )
                for d in range(3)
            )
            ng_face = self.face_field[lev][0].mf.n_grow()
            self._face_offset_cache[cache_key] = (face_offsets, ng_face)

        face_offsets, ng_face = self._face_offset_cache[cache_key]

        return self.scheme.build_kernel(
            face_bufs=face_fb.bufs,
            face_offsets=face_offsets,
            Nx=bucket.Nx_arr,
            Ny=bucket.Ny_arr,
            Nz=bucket.Nz_arr,
            ng=bucket.ng,
            dh=bucket.dh_arr,
            coeff=self.coeff,
            ncomp=self.cell_field.ncomp,
            ng_face=ng_face,
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
            coords.append(jnp.arange(n) * dx[e] + (prob_lo[e] + (lo[e] + offset) * dx[e]))

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
    """Precomputes face-centre coordinates and batches velocity evaluation.

    Groups boxes by shape so that boxes with the same dimensions are stacked
    and evaluated in a single jax.vmap + jax.jit call.
    """

    def __init__(self, face_fluxes, vel_func, geom):
        import jax

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
                    coords.append(jnp.arange(n) * dx[e] + (prob_lo[e] + (lo[e] + offset) * dx[e]))
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
