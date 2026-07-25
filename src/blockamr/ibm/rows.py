# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Wall-row builders — the geometry half of the IBM (tasks T9, T10).

Both builders return the arrays of the frozen row contract
(``plans/IBM/ibm-row-format.md`` §2), ready to be splatted into ``WallTable``::

    {"target", "donor", "w", "ndonor", "b", "gamma", "patch", "box_offset"}

Rows are grouped per local box in ``MFIterator`` order; ``box_offset`` is the
CSR-style boundary array (rows of local box ``i`` are
``[box_offset[i], box_offset[i+1])``).

Structure: everything below ``BoxGrid`` is **pure numpy** and takes explicit
per-box index ranges, so the geometry is testable without the compiled
extension. Only :func:`box_grids` touches ``blockamr``.

Assumptions and conventions
---------------------------

* **Signed distance.** ``s > 0`` is fluid; ``s <= 0`` is not. A cell is *ghost*
  when it is non-fluid and at least one of its six face-neighbours is fluid,
  and *deep solid* otherwise (design §2).
* **Classification is analytic and grown.** The sdf is evaluated on the valid
  box grown by one cell, never read back from a MultiFab, so a ghost cell
  sitting on a box edge is still detected.
* **Multiple bodies.** A cell is non-fluid if *any* body contains it (the union
  sdf, ``min_b s_b``). The row's owner is the body with the **smallest**
  ``|s|`` among those that contain the cell — the nearest surface wins, not the
  deepest. Ties go to the lowest patch id. ``patch[r]`` is the index of that
  body in ``sorted(bodies)``; that ordering is the diagnostics/force key.
* **Donor indices are not clamped to the box.** Donors routinely live in a
  neighbour's halo, which is exactly why the schedule fills boundaries before
  ``P`` (contract §6). Indices are emitted **unwrapped**, so they may also step
  outside the *domain*: that index addresses the halo cell the fill step just
  populated, which is exactly what the kernel must read. Their fluid/solid
  state is judged at the wrapped position in a periodic direction (that halo
  cell *is* the wrapped cell, body and all); in a non-periodic direction the
  halo holds a physical boundary value and the analytic body is simply
  evaluated at the extended coordinate. Nothing is clamped either way.
* **Halo width — not bounded by a constant in code.** ``|s_G| < max(dx)`` is
  enforced (:func:`_check_adjacent`) and the Invariant-D ladder moves the image
  point at most ``|s_G| + |n̂ ⊙ dx|`` from the ghost centre, so no donor sits
  further than ``2 max(dx)`` away *in physical space*. In index units of
  direction ``d`` that is ``2 max(dx)/dx_d + 1`` cells, which is a constant only
  for isotropic cells — there it is **3**. Measured reach outside the valid box:
  0–2 cells for a cylinder or a tilted plane on cubic cells, but **7** on the
  quasi-2D ``32 x 32 x 4`` grid (``dz = 8 dx``). Nothing clamps this; the field's
  ghost width must cover whatever the geometry actually produces.
"""

from dataclasses import dataclass

import numpy as np

from .bc import broadcast_gamma

#: Fixed donor stride of the row format (trilinear).
K = 8

#: Relative tolerance for the ``p == q`` Robin degeneracy (design §1.3).
_DEGENERACY_EPS = 1e-12

#: The eight trilinear corner offsets, in donor-slot order.
_OFFSETS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.int64)


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


# ---------------------------------------------------------------------------
# public builders
# ---------------------------------------------------------------------------


def ghost_cell_rows(mesh, lev, bodies, ibm_bc, ncomp):
    """Sharp-interface reconstruction rows for every non-fluid cell of ``lev``.

    ``bodies`` and ``ibm_bc`` are patch-keyed dicts with identical keys.
    """
    return ghost_cell_rows_on_grids(box_grids(mesh, lev), bodies, ibm_bc, ncomp)


def direct_forcing_rows(mesh, lev, bodies, u_body, ncomp):
    """Direct-forcing rows: every solid cell is pinned to ``u_body``.

    Not wired yet: this is the T6 seam (``plans/IBM/ibm-tasks.md``) — ``DirectForcing``
    still builds its own jnp mask, and nothing in ``src/`` or ``test/`` calls this.
    """
    return direct_forcing_rows_on_grids(box_grids(mesh, lev), bodies, u_body, ncomp)


def ghost_cell_rows_on_grids(grids, bodies, ibm_bc, ncomp):
    """:func:`ghost_cell_rows` on explicit :class:`BoxGrid` descriptions."""
    names, body_list = _patches(bodies)
    _check_bc_keys(names, ibm_bc)
    robin = [ibm_bc[name].robin() for name in names]
    gammas = (
        np.stack([broadcast_gamma(r[2], ncomp) for r in robin]) if names else np.zeros((0, ncomp))
    )
    per_box = [_ghost_rows_for_box(grid, names, body_list, robin, gammas, ncomp) for grid in grids]
    return _concat(per_box, ncomp)


def direct_forcing_rows_on_grids(grids, bodies, u_body, ncomp):
    """:func:`direct_forcing_rows` on explicit :class:`BoxGrid` descriptions."""
    _names, body_list = _patches(bodies)
    gamma = broadcast_gamma(u_body, ncomp)
    per_box = [_solid_rows_for_box(grid, body_list, gamma) for grid in grids]
    return _concat(per_box, ncomp)


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
    # DirectForcing.build_data — the rows are independent of any field's
    # ghost width.
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


# ---------------------------------------------------------------------------
# patches and bcs
# ---------------------------------------------------------------------------


def _patches(bodies):
    """Patch names in the id order the rows use, plus the matching bodies."""
    names = sorted(bodies)
    return names, [bodies[name] for name in names]


def _check_bc_keys(names, ibm_bc):
    for name in names:
        if name not in ibm_bc:
            raise ValueError(f"no ibm_bc entry for immersed patch '{name}'")
    for name in ibm_bc:
        if name not in names:
            raise ValueError(f"ibm_bc entry '{name}' has no matching body in mesh.bodies")


# ---------------------------------------------------------------------------
# geometry core (pure numpy)
# ---------------------------------------------------------------------------


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
        shape = tuple(grid.hi[d] - grid.lo[d] + 1 for d in range(3))
        none = np.zeros(shape, dtype=bool)
        return none, none, np.zeros(shape, dtype=np.int64), np.zeros(shape)

    idx = np.stack(
        np.meshgrid(*[np.arange(grid.lo[d] - 1, grid.hi[d] + 2) for d in range(3)], indexing="ij"),
        axis=-1,
    )
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


def _trilinear_donors(points, grid):
    """The 8 cells surrounding each point and their trilinear weights."""
    plo = np.asarray(grid.prob_lo)
    dx = np.asarray(grid.dx)
    t = (np.asarray(points, dtype=float) - plo) / dx - 0.5
    base = np.floor(t).astype(np.int64)
    frac = t - base
    idx = base[:, np.newaxis, :] + _OFFSETS[np.newaxis]
    corner = np.where(
        _OFFSETS[np.newaxis] == 0, 1.0 - frac[:, np.newaxis, :], frac[:, np.newaxis, :]
    )
    return idx, corner.prod(axis=2)


def _donor_coords(idx, grid):
    """Cell centres of ``idx``, **unwrapped**.

    This is the position the trilinear stencil was built around, so it is what
    the interpolation geometry must use. :func:`_index_coords` wraps instead,
    which is what the fluid/solid lookup wants (that halo cell *is* the wrapped
    cell) and what the geometry must not do.
    """
    return np.asarray(grid.prob_lo) + (idx + 0.5) * np.asarray(grid.dx)


def _containing_cell(points, grid):
    """The cell each point lies in (not the same as the donor base index)."""
    plo = np.asarray(grid.prob_lo)
    dx = np.asarray(grid.dx)
    return np.floor((np.asarray(points, dtype=float) - plo) / dx).astype(np.int64)


def _normals(points, owner, body_list):
    """Per-point unit normal of the owning body, shape ``(n, 3)``."""
    out = np.zeros((points.shape[0], 3), dtype=float)
    for b, body in enumerate(body_list):
        sel = owner == b
        if sel.any():
            p = points[sel]
            out[sel] = body.normal(p[:, 0], p[:, 1], p[:, 2])
    return out


def _cell_name(idx):
    return "[" + ", ".join(str(int(v)) for v in idx) + "]"


def _check_adjacent(dist, target, own, names, grid):
    """Enforce the module's ``|s_G| < dx`` invariant on every ghost row.

    The bound is rigorous for a true signed distance (1-Lipschitz): a ghost cell
    ``G`` has some face neighbour ``N`` at distance ``dx_d`` with ``s_N > 0``, so
    ``|s_G| = -s_G < s_N - s_G <= dx_d``. A row that violates it was classified
    ghost by a function that is not the distance to the surface the mirror is
    built against — the "ghost cell" is not adjacent to any surface and the
    mirror construction is meaningless, so the row must not be emitted.
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


def _ghost_rows_for_box(grid, names, body_list, robin, gammas, ncomp):
    """Reconstruction rows for the ghost cells, zero rows for the deep solid."""
    ghost, solid, owner, s_owner = _classify(grid, body_list)
    lo = np.asarray(grid.lo)
    target = np.argwhere(ghost) + lo
    ng = target.shape[0]
    if ng == 0:
        return _stack_rows(_empty(ncomp), _solid_block(solid, owner, lo, ncomp))

    own = owner[ghost]
    s_g = s_owner[ghost]
    x_g = _index_coords(target, grid)
    n_hat = _normals(x_g, own, body_list)

    # mirror image point, then the Invariant D ladder (design §2.1)
    dist = np.abs(s_g)  # wall distance of the ghost cell...
    _check_adjacent(dist, target, own, names, grid)
    d_ip = dist.copy()  # ...and of the image point, on the fluid side
    x_ip = x_g - 2.0 * s_g[:, np.newaxis] * n_hat
    donor, c = _trilinear_donors(x_ip, grid)

    # Step 2 of the ladder is the *primary* path: the mirror stencil is kept
    # only when all eight of its donors are fluid. Dropping donors (step 1)
    # moves the effective interpolation point off x_IP by O(dx) and costs the
    # reconstruction its linear exactness, so a single one-cell push-out is
    # tried first — it lowers theta below 1/2, which also shrinks the row
    # amplification b = 1/(alpha(1-theta)) rather than growing it.
    retry = ~_fluid_at_index(donor, grid, body_list).all(axis=1)
    if retry.any():
        sel = np.flatnonzero(retry)
        step = np.linalg.norm(n_hat * np.asarray(grid.dx), axis=1)
        pushed = x_g + (dist + step)[:, np.newaxis] * n_hat
        push_ok = _fluid_at_index(_containing_cell(pushed[sel], grid), grid, body_list)
        mirror_ok = _fluid_at_index(_containing_cell(x_ip[sel], grid), grid, body_list)
        if not (push_ok | mirror_ok).all():
            # step 3: no image point lands in fluid — never fall back silently
            r = sel[np.flatnonzero(~(push_ok | mirror_ok))[0]]
            raise ValueError(
                f"IBM ghost cell {_cell_name(target[r])} on patch '{names[own[r]]}' has no "
                "fluid cell to interpolate from: the cell holding the mirror image point "
                "is not fluid, and neither is the cell holding the one-cell-out fallback "
                "point — the fluid on the far side of the surface is under one cell deep "
                "there. Refine the mesh there or move the bodies apart."
            )
        take = sel[push_ok]
        x_ip[take] = pushed[take]
        d_ip[take] = step[take]
        donor[take], c[take] = _trilinear_donors(pushed[take], grid)

    fluid = _fluid_at_index(donor, grid, body_list)
    # step 1, now only the fallback for the rows the push-out could not make
    # whole: drop non-fluid donors and renormalise the survivors. The cell the
    # image point lies in is always one of the eight donors and carries at least
    # 1/8 of the weight, and the ladder above just proved it is fluid — so the
    # surviving sum is bounded away from zero.
    c = np.where(fluid, c, 0.0)
    c /= c.sum(axis=1)[:, np.newaxis]
    dropped = ~fluid.all(axis=1)
    if dropped.any():
        # Renormalising moved the point the row actually interpolates at off
        # x_IP, so Delta must be measured from *that* point — otherwise b and
        # theta describe an interpolation the row does not perform, and the row
        # is only first-order accurate (e.g. FixedGradient's b = -Delta is then
        # wrong by O(dx)).
        x_eff = (c[dropped][..., np.newaxis] * _donor_coords(donor[dropped], grid)).sum(axis=1)
        d_ip[dropped] = ((x_eff - x_g[dropped]) * n_hat[dropped]).sum(axis=1) - dist[dropped]

    # A donor whose weight is exactly zero contributes nothing but is still
    # dereferenced by the kernel (which loops k < ndonor unconditionally), so a
    # zero-fraction direction would make it read a possibly uninitialised halo
    # cell. Drop those slots too; it also shortens the kernel's inner loop.
    live = fluid & (c != 0.0)

    # The wall sits a fraction ``theta`` of the way from G to IP, so
    # phi_w = (1-theta) phi_G + theta phi_IP and dphi/dn|_w = (phi_IP-phi_G)/Delta.
    # Design §1.3 writes the mirror case, theta = 1/2 and p = alpha/2; the
    # push-out fallback of §2.1 moves the image point off the mirror, and only
    # the general theta keeps the reconstruction linear-exact there. For a
    # mirrored image point ``theta`` is exactly 0.5 and this reduces, bitwise,
    # to the formula and to the FixedValue/FixedGradient rows of contract §5.
    delta = dist + d_ip
    theta = dist / delta
    alpha = np.array([r[0] for r in robin])[own]
    beta = np.array([r[1] for r in robin])[own]
    p = alpha * (1.0 - theta)
    q = beta / delta
    den = p - q
    degenerate = np.abs(den) <= _DEGENERACY_EPS * (np.abs(p) + np.abs(q))
    if degenerate.any():
        r = int(np.flatnonzero(degenerate)[0])
        raise ValueError(
            f"IBM ghost cell {_cell_name(target[r])} on patch '{names[own[r]]}' has a "
            f"degenerate Robin condition (p == q, i.e. alpha*(1-theta)*Delta == beta, "
            f"with Delta={delta[r]:.6g} and theta={theta[r]:.6g}): the surface condition "
            "is ill posed at that wall distance."
        )

    w = -((alpha * theta + q) / den)[:, np.newaxis] * c
    b = 1.0 / den
    gamma = gammas[own]

    # live donors first, dead slots padded with the row's own target (never
    # read: their weight is zero, and reading one's own target cannot race)
    order = np.argsort(~live, axis=1, kind="stable")
    live_s = np.take_along_axis(live, order, axis=1)
    w = np.where(live_s, np.take_along_axis(w, order, axis=1), 0.0)
    donor = np.where(
        live_s[..., np.newaxis],
        np.take_along_axis(donor, order[..., np.newaxis], axis=1),
        target[:, np.newaxis, :],
    )
    ndonor = live.sum(axis=1)

    rows = {
        "target": target,
        "donor": donor,
        "w": w,
        "ndonor": ndonor,
        "b": b,
        "gamma": gamma,
        "patch": own,
    }
    return _stack_rows(rows, _solid_block(solid, owner, lo, ncomp))


def _solid_block(solid, owner, lo, ncomp):
    """Deep-solid rows: ``ndonor = 0``, ``b = 0``, ``gamma = 0`` (contract §5)."""
    target = np.argwhere(solid) + lo
    n = target.shape[0]
    return {
        "target": target,
        "donor": np.repeat(target[:, np.newaxis, :], K, axis=1),
        "w": np.zeros((n, K)),
        "ndonor": np.zeros(n, dtype=np.int64),
        "b": np.zeros(n),
        "gamma": np.zeros((n, ncomp)),
        "patch": owner[solid],
    }


def _solid_rows_for_box(grid, body_list, gamma):
    """Direct-forcing rows: every non-fluid cell takes the body value."""
    ghost, solid, owner, _s = _classify(grid, body_list)
    non_fluid = ghost | solid
    lo = np.asarray(grid.lo)
    target = np.argwhere(non_fluid) + lo
    n = target.shape[0]
    return {
        "target": target,
        "donor": np.repeat(target[:, np.newaxis, :], K, axis=1),
        "w": np.zeros((n, K)),
        "ndonor": np.zeros(n, dtype=np.int64),
        "b": np.ones(n),
        "gamma": np.repeat(gamma[np.newaxis], n, axis=0),
        "patch": owner[non_fluid],
    }


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def _empty(ncomp):
    return {
        "target": np.zeros((0, 3), dtype=np.int64),
        "donor": np.zeros((0, K, 3), dtype=np.int64),
        "w": np.zeros((0, K)),
        "ndonor": np.zeros(0, dtype=np.int64),
        "b": np.zeros(0),
        "gamma": np.zeros((0, ncomp)),
        "patch": np.zeros(0, dtype=np.int64),
    }


_ROW_KEYS = ("target", "donor", "w", "ndonor", "b", "gamma", "patch")
_INT_KEYS = ("target", "donor", "ndonor", "patch")


def _stack_rows(*blocks):
    return {key: np.concatenate([blk[key] for blk in blocks]) for key in _ROW_KEYS}


def _concat(per_box, ncomp):
    """Concatenate per-box row blocks and add the CSR ``box_offset``."""
    blocks = per_box if per_box else [_empty(ncomp)]
    out = _stack_rows(*blocks)
    counts = [blk["target"].shape[0] for blk in per_box]
    out["box_offset"] = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    for key in _ROW_KEYS:
        dtype = np.int32 if key in _INT_KEYS else np.float64
        out[key] = np.ascontiguousarray(out[key], dtype=dtype)
    return out
