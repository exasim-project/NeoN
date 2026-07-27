# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Cell classification against the immersed bodies (task B1).

What every cell of a level is, relative to ``mesh.bodies``: its signed stencil
:func:`depth <cell_depth>`, its owning patch, and the union signed distance.
This is the method-agnostic layer — nothing here knows an IBM method, an
operator or a scheme (``plans/IBM/overview.md`` §3).

Everything below :class:`BoxGrid` is **pure numpy** and takes explicit per-box
index ranges, so the classification is testable without the compiled extension.
Only :func:`box_grids` touches ``blockamr``.

Assumptions and conventions
---------------------------

* **Signed distance.** ``s > 0`` is fluid; ``s <= 0`` is not. A cell is
  non-fluid when *any* body contains it (the union sdf, ``min_b s_b``).
* **Depth** is signed, per cell, and clamped to ``±MAX_DEPTH``
  (``plans/IBM/overview.md`` §4). A fluid cell's depth is the number of cells
  along the nearest of the six axis rays to the first non-fluid cell, so
  ``depth = 1`` means a face neighbour is non-fluid. A non-fluid cell's depth
  is ``0`` when a face neighbour is fluid and negative going deeper. The band
  of a width-``w`` cross-stencil scheme is then exactly ``depth <= w``.
* **Classification is analytic and grown.** The sdf is evaluated on the valid
  box grown by ``MAX_DEPTH`` cells, never read back from a MultiFab, so the
  rays of a cell on a box edge are available analytically.
* **The owner is the nearest containing surface**, not the deepest: among the
  bodies that contain the cell, the one with the smallest ``|s|``. Ties go to
  the lowest patch id. For a fluid cell — which no body contains — it is the
  nearest surface. ``patch`` is the index of that body in ``sorted(bodies)``;
  that ordering is the diagnostics/force key.
* **Indices are not clamped to the box.** Ray and donor indices routinely live
  in a neighbour's halo, and they are emitted **unwrapped**, so they may also
  step outside the *domain*: that index addresses the halo cell a fill step
  populated, which is exactly what the kernel must read. Their fluid/solid
  state is judged at the **wrapped** position in a periodic direction (that
  halo cell *is* the wrapped cell, body and all); in a non-periodic direction
  the halo holds a physical boundary value and the analytic body is simply
  evaluated at the extended coordinate. Nothing is clamped either way, and the
  interpolation geometry uses the unwrapped position (:mod:`.geometry`).
"""

from dataclasses import dataclass

import numpy as np

#: Depth is clamped to this magnitude: one array serves every stencil width.
MAX_DEPTH = 4


@dataclass(frozen=True)
class BoxGrid:
    """Index range and coordinate metadata of one local box.

    ``lo``/``hi`` are the **inclusive** valid-box corners in global index
    space; ``domain_lo``/``domain_hi`` are the same for the whole level.
    """

    lo: tuple[int, int, int]
    hi: tuple[int, int, int]
    dx: tuple[float, float, float]
    prob_lo: tuple[float, float, float]
    domain_lo: tuple[int, int, int]
    domain_hi: tuple[int, int, int]
    periodic: tuple[bool, bool, bool]


def box_grids(mesh, lev):
    """The level's local boxes as :class:`BoxGrid`, in ``MFIterator`` order."""
    # Deferred so the geometry core above stays importable (and testable)
    # without the compiled extension.
    import blockamr

    geom = mesh.geom(lev)
    domain = geom.domain()
    common = {
        "dx": tuple(float(v) for v in geom.cell_size()),
        "prob_lo": tuple(float(v) for v in geom.prob_lo()),
        "domain_lo": tuple(int(v) for v in domain.small_end()),
        "domain_hi": tuple(int(v) for v in domain.big_end()),
        "periodic": tuple(bool(v) for v in geom.is_periodic()),
    }
    # Zero-ghost scratch MultiFab purely for the box layout, as in
    # DirectForcing.build_data — the classification is independent of any
    # field's ghost width.
    scratch = blockamr.MultiFab(mesh.box_array(lev), mesh.dm(lev), 1, 0)
    grids = []
    for mfi in blockamr.MFIterator(scratch):
        box = mfi.valid_box()
        grids.append(
            BoxGrid(
                lo=tuple(int(v) for v in box.small_end()),
                hi=tuple(int(v) for v in box.big_end()),
                **common,
            )
        )
    return grids


def _patches(bodies):
    """Patch names in the id order the rows use, plus the matching bodies."""
    names = sorted(bodies)
    return names, [bodies[name] for name in names]


# ---------------------------------------------------------------------------
# geometry core (pure numpy)
# ---------------------------------------------------------------------------


def _valid_shape(grid):
    """Cell counts of the box's valid region."""
    return tuple(grid.hi[d] - grid.lo[d] + 1 for d in range(3))


def _valid_index(grid):
    """Global indices of the box's valid cells, shape ``(nx, ny, nz, 3)``."""
    return _index_grid(grid, 0)


def _index_grid(grid, ghost):
    """Global indices of the valid box grown by ``ghost`` cells."""
    return np.stack(
        np.meshgrid(
            *[np.arange(grid.lo[d] - ghost, grid.hi[d] + ghost + 1) for d in range(3)],
            indexing="ij",
        ),
        axis=-1,
    )


def _index_coords(idx, grid):
    """Cell centres of the cells ``idx`` (``(..., 3)`` ints).

    Indices outside the domain are wrapped in a periodic direction — that halo
    cell *is* the wrapped one, so the body geometry it sees is the wrapped
    geometry. In a non-periodic direction there is nothing to wrap onto and the
    analytic body is simply evaluated at the extended coordinate.
    """
    coords = np.empty(idx.shape, dtype=float)
    for d in range(3):
        i = idx[..., d]
        if grid.periodic[d]:
            lo, hi = grid.domain_lo[d], grid.domain_hi[d]
            i = lo + np.mod(i - lo, hi - lo + 1)
        coords[..., d] = grid.prob_lo[d] + (i + 0.5) * grid.dx[d]
    return coords


def _sdf_stack(body_list, x, y, z):
    """Signed distance of every body at the given points, shape ``(nb, ...)``."""
    if not body_list:
        return np.full((0,) + np.shape(x), np.inf)
    return np.stack([np.broadcast_to(b.sdf(x, y, z), np.shape(x)) for b in body_list])


def _fluid_at_index(idx, grid, body_list):
    """True where the cell ``idx`` is a fluid cell (all bodies' ``s > 0``)."""
    coords = _index_coords(idx, grid)
    s = _sdf_stack(body_list, coords[..., 0], coords[..., 1], coords[..., 2]).min(axis=0)
    return s > 0.0


def _classify(grid, body_list):
    """``(ghost, solid, owner, s_owner)`` over the box's valid cells.

    The sdf is evaluated on the box grown by one cell so that face-neighbours
    of edge cells are available analytically.
    """
    if not body_list:
        # no bodies: every cell is fluid, so there are no rows of any kind
        shape = _valid_shape(grid)
        none = np.zeros(shape, dtype=bool)
        return none, none, np.zeros(shape, dtype=np.int64), np.zeros(shape)

    idx = _index_grid(grid, 1)
    coords = _index_coords(idx, grid)
    s_all = _sdf_stack(body_list, coords[..., 0], coords[..., 1], coords[..., 2])
    fluid = s_all.min(axis=0) > 0.0

    core = ~fluid[1:-1, 1:-1, 1:-1]
    nbr_fluid = (
        fluid[2:, 1:-1, 1:-1]
        | fluid[:-2, 1:-1, 1:-1]
        | fluid[1:-1, 2:, 1:-1]
        | fluid[1:-1, :-2, 1:-1]
        | fluid[1:-1, 1:-1, 2:]
        | fluid[1:-1, 1:-1, :-2]
    )
    ghost = core & nbr_fluid
    solid = core & ~nbr_fluid

    inner = s_all[:, 1:-1, 1:-1, 1:-1]
    # nearest containing surface owns the cell; ties -> lowest patch id
    owner = np.argmin(np.where(inner <= 0.0, np.abs(inner), np.inf), axis=0)
    s_owner = np.take_along_axis(inner, owner[np.newaxis], axis=0)[0]
    return ghost, solid, owner, s_owner


def _ray_neighbour(fluid, axis, step, ghost):
    """The grown mask ``fluid`` sampled ``step`` cells along ``axis``."""
    window = []
    for d in range(3):
        n = fluid.shape[d] - 2 * ghost
        lo = ghost + (step if d == axis else 0)
        window.append(slice(lo, lo + n))
    return fluid[tuple(window)]


def cell_depth(grid, body_list):
    """Signed clamped stencil depth of every valid cell of one box.

    ``+k`` is a fluid cell whose nearest non-fluid cell is ``k`` axis-ray steps
    away, ``0`` is a non-fluid cell with a fluid face neighbour, ``-k`` is a
    non-fluid cell whose nearest fluid cell is ``k + 1`` steps away; the whole
    range is clamped to ``±MAX_DEPTH``.
    """
    shape = _valid_shape(grid)
    if not body_list:
        return np.full(shape, MAX_DEPTH, dtype=np.int8)

    ghost = MAX_DEPTH
    fluid = _fluid_at_index(_index_grid(grid, ghost), grid, body_list)
    here = _ray_neighbour(fluid, 0, 0, ghost)

    # Rays are walked outward-in so that a shorter one overwrites a longer one;
    # the initial values are the clamp, i.e. "no state change within reach".
    to_non_fluid = np.full(shape, MAX_DEPTH, dtype=np.int64)
    to_fluid = np.full(shape, MAX_DEPTH + 1, dtype=np.int64)
    for k in range(MAX_DEPTH, 0, -1):
        for axis in range(3):
            for step in (k, -k):
                neighbour = _ray_neighbour(fluid, axis, step, ghost)
                to_non_fluid = np.where(neighbour, to_non_fluid, k)
                to_fluid = np.where(neighbour, k, to_fluid)
    return np.where(here, to_non_fluid, 1 - to_fluid).astype(np.int8)


def _owner(s_all):
    """``(owning body id, its signed distance)`` from the per-body sdf stack."""
    contained = np.where(s_all <= 0.0, np.abs(s_all), np.inf)
    owner = np.where(
        np.isinf(contained).all(axis=0),
        # a fluid cell is contained by no body; the nearest surface owns it
        np.argmin(s_all, axis=0),
        np.argmin(contained, axis=0),
    )
    return owner, np.take_along_axis(s_all, owner[np.newaxis], axis=0)[0]


def classify_box(grid, names, body_list):
    """``(depth, patch, sdf)`` over the valid cells of one box.

    ``sdf`` is the union ``min_b s_b``; ``patch`` is the owning body's id and
    ``depth`` the signed clamped stencil depth. Raises if a body is not
    resolvable on this mesh (:func:`_check_adjacent`).
    """
    depth = cell_depth(grid, body_list)
    shape = _valid_shape(grid)
    if not body_list:
        return depth, np.zeros(shape, dtype=np.int8), np.full(shape, np.inf)

    index = _valid_index(grid)
    coords = _index_coords(index, grid)
    s_all = _sdf_stack(body_list, coords[..., 0], coords[..., 1], coords[..., 2])
    owner, s_owner = _owner(s_all)

    wall_layer = depth == 0
    _check_adjacent(np.abs(s_owner[wall_layer]), index[wall_layer], owner[wall_layer], names, grid)
    _check_resolvable_gap(grid, body_list, names, owner, index)
    return depth, owner.astype(np.int8), s_all.min(axis=0)


def _cell_name(idx):
    return "[" + ", ".join(str(int(v)) for v in idx) + "]"


def _check_resolvable_gap(grid, body_list, names, owner, index):
    """Refuse a fluid channel that no cell centre samples.

    Two adjacent cells are both non-fluid, and yet the **face between them** is
    in the fluid: a channel runs between the two cell centres and is narrower
    than the distance between them, so nothing in the discrete field ever sees
    it. Two bodies less than a cell apart are the case this exists for — the
    mesh merges them into one solid region, silently, and returns a plausible
    field.

    It is deliberately a statement about the *fluid*, not about the distance
    between two surfaces: two bodies that overlap have no channel between them
    at all, however close their surfaces come, and that configuration is
    legitimate (it is how a compound solid is built out of primitives).

    This check, not Invariant F, is where a thin gap is caught under this
    design. The previous design mirrored a value from inside the body into the
    fluid, and a sub-cell channel showed up as a mirror with no fluid cell to
    interpolate from; nothing is reconstructed inside a body any more (a
    non-fluid cell is an ``nnz = 0`` row), so that symptom is gone and the
    geometry has to be judged directly.
    """
    grown = _index_coords(_index_grid(grid, 1), grid)
    solid = _union_sdf(body_list, grown) <= 0.0
    core = (slice(1, -1),) * 3
    here = solid[core]
    coords = grown[core]
    for axis in range(3):
        for step in (1, -1):
            window = []
            for d in range(3):
                lo = 1 + (step if d == axis else 0)
                window.append(slice(lo, lo + here.shape[d]))
            face = coords.copy()
            face[..., axis] += 0.5 * step * grid.dx[axis]
            bad = here & solid[tuple(window)] & (_union_sdf(body_list, face) > 0.0)
            if bad.any():
                r = tuple(np.argwhere(bad)[0])
                raise ValueError(
                    f"IBM cell {_cell_name(index[r])} on patch '{names[owner[r]]}' and its "
                    f"neighbour are both non-fluid, but the face between them is in the fluid: "
                    f"a fluid channel runs between the two cell centres and is thinner than the "
                    f"cell size {grid.dx[axis]:.6g}, so no cell centre samples it and the mesh "
                    "cannot resolve it. Refine the mesh there or move the bodies apart."
                )


def _union_sdf(body_list, coords):
    """``min_b s_b`` at the given points, shape ``coords.shape[:-1]``."""
    return _sdf_stack(body_list, coords[..., 0], coords[..., 1], coords[..., 2]).min(axis=0)


def _check_adjacent(dist, target, own, names, grid):
    """Enforce the module's ``|s_G| < dx`` invariant on every ``depth = 0`` cell.

    The bound is rigorous for a true signed distance (1-Lipschitz): such a cell
    ``G`` has some face neighbour ``N`` at distance ``dx_d`` with ``s_N > 0``, so
    ``|s_G| = -s_G < s_N - s_G <= dx_d``. A cell that violates it was classified
    against a function that is not the distance to the surface it is closed
    against — the cell is not adjacent to any surface, and every wall
    construction built on it is meaningless.
    """
    cell = max(grid.dx)
    bad = dist >= cell
    if bad.any():
        r = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"IBM ghost cell {_cell_name(target[r])} on patch '{names[own[r]]}' is "
            f"{dist[r]:.6g} from that surface, which is not less than the cell size "
            f"{cell:.6g}: the cell is not adjacent to the surface it was classified "
            "against, so mirroring it across that surface is meaningless. The body is "
            "incompatible with this mesh — most often because it is not periodic while "
            "the domain is, so its solid region meets fluid across the periodic seam."
        )
