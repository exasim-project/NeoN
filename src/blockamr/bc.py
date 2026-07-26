# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Non-periodic boundary conditions for block-structured meshes.

Provides Dirichlet and Neumann ghost-cell filling for CellFields
on domains with solid walls.
"""

import blockamr


class DirichletBC:
    """Dirichlet BC: ghost = 2*value - interior (linear extrapolation to face)."""

    def __init__(self, value=0.0):
        self.value = value

    def fill(self, arr, axis, side, ngrow):
        """Fill ghost cells on one side of one axis.

        Parameters
        ----------
        arr : ndarray
            The full grown array (interior + ghosts), shape (nx+2ng, ny+2ng, nz+2ng, ncomp).
        axis : int
            Axis index (0=x, 1=y, 2=z).
        side : int
            0 = low side, 1 = high side.
        ngrow : int
            Number of ghost cells.
        """
        n = arr.shape[axis] - 2 * ngrow  # interior cells along this axis
        for g in range(ngrow):
            if side == 0:
                ghost_idx = ngrow - 1 - g
                interior_idx = ngrow + g
            else:
                ghost_idx = ngrow + n + g
                interior_idx = ngrow + n - 1 - g
            src = _take(arr, interior_idx, axis)
            val = 2.0 * self.value - src
            _put(arr, ghost_idx, axis, val)


class NeumannBC:
    """Zero-gradient (Neumann) BC: ghost = interior."""

    def fill(self, arr, axis, side, ngrow):
        n = arr.shape[axis] - 2 * ngrow
        for g in range(ngrow):
            if side == 0:
                ghost_idx = ngrow - 1 - g
                interior_idx = ngrow + g
            else:
                ghost_idx = ngrow + n + g
                interior_idx = ngrow + n - 1 - g
            _put(arr, ghost_idx, axis, _take(arr, interior_idx, axis))


class SlipBC:
    """Free-slip / symmetry wall (native BC code 3).

    No penetration + zero tangential shear: the velocity component *normal* to
    the face is reflected with a sign flip (ghost = -interior → zero normal
    velocity at the face) while the *tangential* components are copied
    (ghost = interior → zero gradient). cf. OpenFOAM ``slip`` / ``symmetry``.
    """

    def fill(self, arr, axis, side, ngrow):
        n = arr.shape[axis] - 2 * ngrow
        ncomp = arr.shape[-1] if arr.ndim == 4 else 1
        for g in range(ngrow):
            if side == 0:
                ghost_idx = ngrow - 1 - g
                interior_idx = ngrow + g
            else:
                ghost_idx = ngrow + n + g
                interior_idx = ngrow + n - 1 - g
            src = _take(arr, interior_idx, axis)
            if arr.ndim == 4 and ncomp > 1:
                val = src.copy()
                # component normal to this face (== axis) reflects with -sign
                val[..., axis] = -val[..., axis]
                _put(arr, ghost_idx, axis, val)
            else:
                _put(arr, ghost_idx, axis, src)


class BoundaryCondition:
    """Per-face BC specification for a box domain.

    Parameters
    ----------
    lo : list of BC objects (length 3)
        Boundary conditions for x_lo, y_lo, z_lo faces.
    hi : list of BC objects (length 3)
        Boundary conditions for x_hi, y_hi, z_hi faces.
    """

    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi


def fill_ghost_cells(mf, geom, bc):
    """Fill ghost cells of a MultiFab according to BoundaryCondition.

    Uses a GPU-native C++ kernel — no host round-trip.
    """
    ngrow = mf.n_grow()
    if ngrow == 0:
        return

    bc_types, bc_values = _bc_to_native_spec(bc, geom)
    mf.fill_domain_boundary(geom, bc_types, bc_values)


def _bc_to_native_spec(bc, geom):
    """Convert a BoundaryCondition to (bc_types, bc_values) for the C++ binding.

    bc_types: [lo_x, hi_x, lo_y, hi_y, lo_z, hi_z]  — 0=skip, 1=dirichlet, 2=neumann
    bc_values: [[vals_lo_x], [vals_hi_x], ...]  — per-component wall values
    """
    is_per = geom.is_periodic()
    bc_types = []
    bc_values = []

    for d in range(3):
        if is_per[d]:
            bc_types.extend([0, 0])
            bc_values.extend([[0.0], [0.0]])
        else:
            bc_types.append(_bc_type_code(bc.lo[d]))
            bc_values.append(_bc_wall_values(bc.lo[d]))
            bc_types.append(_bc_type_code(bc.hi[d]))
            bc_values.append(_bc_wall_values(bc.hi[d]))

    return bc_types, bc_values


def _bc_type_code(bc_obj):
    """Return 1 for Dirichlet, 2 for Neumann, 3 for slip/symmetry."""
    if isinstance(bc_obj, (DirichletBC, VectorDirichletBC)):
        return 1
    elif isinstance(bc_obj, NeumannBC):
        return 2
    elif isinstance(bc_obj, SlipBC):
        return 3
    return 0


def _bc_wall_values(bc_obj):
    """Return list of per-component wall values."""
    if isinstance(bc_obj, VectorDirichletBC):
        return list(bc_obj.vec)
    elif isinstance(bc_obj, DirichletBC):
        return [bc_obj.value]
    return [0.0]


class VectorDirichletBC:
    """Dirichlet BC with per-component values: ghost[c] = 2*vec[c] - interior[c].

    cf. OpenFOAM: fixedValue uniform (ux uy uz)
    """

    def __init__(self, vec):
        self.vec = vec  # [ux, uy, uz]

    def fill(self, arr, axis, side, ngrow):
        ncomp = arr.shape[-1] if arr.ndim == 4 else 1
        if ncomp == 1 or arr.ndim == 3:
            DirichletBC(self.vec[0] if isinstance(self.vec, (list, tuple)) else self.vec).fill(
                arr, axis, side, ngrow)
            return
        for c in range(min(ncomp, len(self.vec))):
            DirichletBC(self.vec[c]).fill(arr[:, :, :, c], axis, side, ngrow)


def fixedValue(vec):
    """Dirichlet BC with vector value [ux, uy, uz]."""
    return VectorDirichletBC(vec)


def noSlip():
    """No-slip wall: fixedValue([0, 0, 0])."""
    return fixedValue([0, 0, 0])


def slip():
    """Free-slip / symmetry wall: :class:`SlipBC`."""
    return SlipBC()


def pressure_domain_bc(u_bc, geom):
    """Per-face pressure ``LinOpBCType`` derived from a velocity ``VectorBC``.

    Standard incompressible pressure/velocity BC pairing (cf. OpenFOAM
    ``inlet: U fixedValue / p zeroGradient``, ``outlet: U zeroGradient / p
    fixedValue``):

    * periodic axis                        → ``Periodic``
    * velocity Neumann face (outflow)      → pressure ``Dirichlet`` (pins the
      otherwise-singular reference and lets flow leave the domain)
    * velocity Dirichlet face (inlet/wall) → pressure ``Neumann`` (zeroGradient)

    Returns ``(lo_bc, hi_bc)`` — two length-3 lists of
    ``blockamr.LinOpBCType`` for the lo/hi faces of each axis, ready for
    ``MLLinOp.set_domain_bc(lo_bc, hi_bc)``.
    """
    bc_type = blockamr.LinOpBCType
    is_per = geom.is_periodic()

    def face_bc(face_obj):
        return bc_type.Dirichlet if isinstance(face_obj, NeumannBC) else bc_type.Neumann

    lo_bc = []
    hi_bc = []
    for d in range(3):
        if is_per[d]:
            lo_bc.append(bc_type.Periodic)
            hi_bc.append(bc_type.Periodic)
        else:
            lo_bc.append(face_bc(u_bc.lo[d]))
            hi_bc.append(face_bc(u_bc.hi[d]))
    return lo_bc, hi_bc


class VectorBC(BoundaryCondition):
    """Per-face vector BC specification.

    cf. OpenFOAM boundaryField:
        lid    { type fixedValue; value uniform (1 0 0); }
        walls  { type noSlip; }

    Usage:
        VectorBC(
            xlo=noSlip(), xhi=noSlip(),
            ylo=noSlip(), yhi=fixedValue([1, 0, 0]),
            zlo=noSlip(), zhi=noSlip(),
        )
    """

    def __init__(self, xlo=None, xhi=None, ylo=None, yhi=None, zlo=None, zhi=None):
        lo = [xlo or noSlip(), ylo or noSlip(), zlo or noSlip()]
        hi = [xhi or noSlip(), yhi or noSlip(), zhi or noSlip()]
        super().__init__(lo, hi)


def _take(arr, idx, axis):
    """Extract a slice along axis at index idx."""
    slices = [slice(None)] * arr.ndim
    slices[axis] = idx
    return arr[tuple(slices)]


def _put(arr, idx, axis, val):
    """Set a slice along axis at index idx."""
    slices = [slice(None)] * arr.ndim
    slices[axis] = idx
    arr[tuple(slices)] = val
