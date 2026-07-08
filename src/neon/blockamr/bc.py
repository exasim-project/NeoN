# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Non-periodic boundary conditions for block-structured meshes.

Provides Dirichlet and Neumann ghost-cell filling for CellFields
on domains with solid walls.
"""

import neon.blockamr as blockamr


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

    bc_types: list of length 6, one entry per face in the order
        [lo_x, hi_x, lo_y, hi_y, lo_z, hi_z]. Each entry is either:
            * a single int (legacy: same code for all components), or
            * a list of 3 ints (per-component code, used by SlipWallBC).
        Codes: 0=skip (leave alone), 1=Dirichlet, 2=Neumann.
    bc_values: 6 lists of per-component wall values (used only for
        Dirichlet components).
    """
    is_per = geom.is_periodic()
    bc_types = []
    bc_values = []

    for d in range(3):
        if is_per[d]:
            bc_types.extend([0, 0])
            bc_values.extend([[0.0], [0.0]])
        else:
            bc_types.append(_bc_type_code(bc.lo[d], normal_dir=d))
            bc_values.append(_bc_wall_values(bc.lo[d]))
            bc_types.append(_bc_type_code(bc.hi[d], normal_dir=d))
            bc_values.append(_bc_wall_values(bc.hi[d]))

    return bc_types, bc_values


def _bc_type_code(bc_obj, normal_dir=0):
    """Return the per-face BC code(s) the C++ binding accepts.

    For most BC types this is a single int (1=Dirichlet, 2=Neumann, 0=skip).
    For ``SlipWallBC`` this is a *list* of 3 ints — Dirichlet 0 on the
    component normal to the wall (so U_n = 0 → no penetration), Neumann
    on the two tangential components.
    """
    if isinstance(bc_obj, SlipWallBC):
        return [2 if c != normal_dir else 1 for c in range(3)]
    if isinstance(bc_obj, (DirichletBC, VectorDirichletBC)):
        return 1
    elif isinstance(bc_obj, NeumannBC):
        return 2
    return 0


def _bc_wall_values(bc_obj):
    """Return list of per-component wall values."""
    if isinstance(bc_obj, SlipWallBC):
        # Normal component is Dirichlet 0; tangential are Neumann (value
        # ignored) — uniform [0,0,0] is the right encoding.
        return [0.0, 0.0, 0.0]
    if isinstance(bc_obj, VectorDirichletBC):
        return list(bc_obj.vec)
    elif isinstance(bc_obj, DirichletBC):
        return [bc_obj.value]
    return [0.0]


class SlipWallBC:
    """Free-slip / symmetry wall: zero normal velocity, zero gradient
    on tangential components.

    The "normal" direction is determined by which face this BC is
    attached to (xlo/xhi → x is normal, ylo/yhi → y is normal, etc.),
    so the same ``SlipWallBC()`` instance can be reused on any face.
    The translation to per-component Dirichlet/Neumann codes happens
    in ``_bc_type_code`` based on the face's normal direction.

    cf. OpenFOAM ``slip``.
    """
    pass


def slipWall():
    """Free-slip wall: zero normal velocity, zero gradient on tangential."""
    return SlipWallBC()


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
