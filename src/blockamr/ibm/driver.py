# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The per-``evaluate`` IBM driver — the wall flow (B36).

One ``evaluate`` under the v2 architecture (``plans/IBM/design.md`` §3,
``plans/IBM/api.md`` §2.3)::

    fill_patch(phi)                       once, before the terms
    interior kernel over the whole valid box   the *same* call as the no-IBM path
    wall_<op>_<method>(...) per term      after every sweep: Overwrite, then Add
    pin_solid(out, ct, 0.0)               the SOLID mask

The non-fluid pin (design §7) is **not** in that list: it is a classification
write, applied once per ``(field, method, lev, grid_version)`` when this driver
is built (B25, ``plans/IBM/review.md`` §4 Q3), so an ``evaluate`` after the
field's first one never writes the field at all.

The interior sweep is untouched — same kernel, same arguments — so the bulk is
bitwise the plain operator's *structurally* rather than by care
(``plans/IBM/overview.md`` §5). The one exception is W1: a width-2 interior
scheme runs its ``_ibm`` sibling, which falls back to its own width-1 formula at
a cell whose stencil would read a ``SOLID`` cell (design §5). That sibling is
selected here, by handing the marker to the backend, and nowhere else.

Two paths, one composition rule
-------------------------------

A ``(operator, method)`` pair that is **compiled** is called through its
``build_cpp_kernel()`` wrapper with the canonical twelve arguments (design
§4.4); one that is not — ``source x ghostCell``, and any pair a test registers
from Python — still emits v1 :class:`~blockamr.ibm.band_rows.BandRows` over the
band. Both write the same cells, and they compose the same way: the first term
to write uses ``Overwrite``, every later one ``Add``.

That "same cells" is what fixes the row path's band width once a compiled pair
is in the equation. A wall sweep writes exactly the ``WALL`` cells, so a row
term composing with it must emit over ``band(1)`` — the wall layer and the
solid cells — and not over the equation's widest band, whose deeper fluid cells
the interior sweep already owns and no wall sweep overwrites. With no compiled
pair in the equation the v1 rule stands unchanged: one band, the widest term's.

The SOLID mask (OPEN-C)
-----------------------

v1 carried every solid cell as an ``nnz = 0, c = 0`` row, so its first
``Overwrite`` term wrote exactly ``0.0`` there. A wall sweep returns before the
sink at ``m != WALL``, which would leave the interior sweep's value — computed
from pinned neighbours and read by nothing — inside the body. Design §7's four
lines are already compiled as ``blockamr.pin_solid``, so the mask is that call
on the *result*, once per level, after the terms.

``ibm is None`` — the no-key path, ``noIbm`` and an empty band — is one branch
*outside* any kernel, in the caller, which is what makes bitwise equality with
the plain operator structural rather than maintained.
"""

import numpy as np

import blockamr

from ..schemes.boundary import resolve
from .band import CROSS
from .band_rows import band_table
from .bc import robin_data
from .classify import _patches


class WallEvaluation:
    """The wall flow for one field, one ``evaluate``.

    Built per ``evaluate``. Holds one boundary scheme per term, resolved up
    front so a missing ``(operator, method)`` pair raises before any kernel
    launches, and — for the pairs that are compiled — the
    :class:`~blockamr.cpp_kernels.CppWallKernel` that names the binding.
    """

    def __init__(self, method, name, cell_field, spatial_ops):
        self.method = method
        self.name = name
        self.field = cell_field
        self.ibm = cell_field.mesh.ibm
        self.terms = list(spatial_ops)
        self.schemes = {term: resolve(_operator_of(term), name, term.scheme) for term in self.terms}
        self.kernels = {term: _wall_kernel(self.schemes[term]) for term in self.terms}
        #: True when at least one term is on a compiled pair — the v2 flow.
        self.on_pairs = any(kernel is not None for kernel in self.kernels.values())
        self.width, self.shape = equation_band(self.terms)
        #: The ghost width the marker and the packed geometry are built at: the
        #: widest interior stencil, since W1's siblings read the marker at their
        #: own reach (``MARKER_NGROW`` is the classification's floor, not a size).
        self.ngrow = max(1, self.width)
        #: The band width a *row* term is asked for — see the module docstring.
        self.row_width = 1 if self.on_pairs else self.width
        # Classification time: the driver is built before the level loop —
        # before the first fill_patch and before any sweep (design §7,
        # review §4 Q3). Once per (field, method, lev, grid_version); every
        # later evaluate is a read.
        for lev in range(cell_field.mesh.n_levels()):
            self.ibm.ensure_pinned(cell_field, method, lev)

    def interior_cell_type(self, lev):
        """The marker the interior sweep degrades against (W1), or ``None``.

        ``None`` for a method with no compiled pair: there is no v2 marker on
        that path, and a width-2 term there is corrected by its rows.
        """
        if not self.on_pairs:
            return None
        return self.ibm.cell_type(self.method, lev, self.ngrow)

    def evaluate_level(self, impl, terms, cell_field, lev, t):
        """One level: the untouched interior sweep, then the wall sweep."""
        return impl.evaluate(terms, cell_field, lev, t, ibm=self)

    def source_level(self, impl, terms, cell_field, lev, t):
        """The step-side twin of :meth:`evaluate_level`: interior sweep and
        wall sweep — returning the accumulated source MultiFab instead of host
        arrays, so ``solve()``'s time schemes can feed it to ``euler_update``
        (the R4 seam between operator and update)."""
        return impl.source(terms, cell_field, lev, t, ibm=self)

    def apply(self, out_mf, lev, t):
        """Write the wall cells of the accumulated result, then mask the solid.

        Called by the backend once the interior sweep of **every** term has
        run: a later term's sweep writes the whole valid box, so a correction
        applied between two sweeps would be overwritten by the second one.
        """
        ncomp = self.field.ncomp
        views = self._views(lev, t, ncomp) if self.on_pairs else None
        first = True
        for term in self.terms:
            kernel = self.kernels[term]
            scheme = self.schemes[term]
            if kernel is None:
                first = self._apply_rows(out_mf, term, scheme, lev, t, ncomp, first)
                continue
            kernel(
                out=out_mf,
                phi=self.field.mf[lev],
                t=t,
                coeff=scheme.wall_coeff(term, t),
                ncomp=ncomp,
                mode=blockamr.WallMode.Overwrite if first else blockamr.WallMode.Add,
                constant_scale=1.0,  # R2: the affine apply (design §4.4)
                **views,
                **scheme.wall_extras(term, lev),
            )
            first = False
        if self.on_pairs:
            # OPEN-C: design §7's four lines, on the result rather than the field.
            blockamr.pin_solid(out_mf, views["cell_type"], 0.0, ncomp)

    # -- internals ----------------------------------------------------------

    def _views(self, lev, t, ncomp):
        """The device views every compiled pair on this level shares."""
        return {
            "cell_type": self.ibm.cell_type(self.method, lev, self.ngrow),
            "geom_ibm": self.ibm.geometry_fab(lev, self.ngrow),
            "method_data": self.ibm.wall_data(self.method, lev, self.ngrow),
            "robin": self._robin(lev, t, ncomp),
            "geom": self.field.mesh.geom(lev),
        }

    def _robin(self, lev, t, ncomp):
        """The per-patch ``(alpha, beta, gamma(t))`` tables, in patch order.

        Rebuilt per ``apply`` because ``gamma`` may be a schedule; it is
        ``npatch`` numbers, so the rebuild is free next to the sweep it feeds.
        """
        names, _bodies = _patches(self.ibm.bodies)
        return robin_data(names, self.field.ibm_bc, ncomp, self._wall_points(lev), t)

    def _wall_points(self, lev):
        """``patch -> (n, 3)`` wall foot points, for a callable datum only.

        The same points, in the same order, v1's ``_band_closure`` hands a
        schedule: the surface feet of the level's ``depth == 1`` cells. Built
        lazily, so a constant datum reads no geometry at all.
        """
        cache = {}

        def points(patch):
            if not cache:
                geometries = self.ibm.geometry(lev)
                at_wall = [g.depth == 1 for g in geometries]
                cache["point"] = _stack([g.wall_point[s] for g, s in zip(geometries, at_wall)], 3)
                cache["patch"] = _stack([g.patch[s] for g, s in zip(geometries, at_wall)], None)
            return cache["point"][cache["patch"] == patch]

        return points

    def _apply_rows(self, out_mf, term, scheme, lev, t, ncomp, first):
        """v1's band-row apply for a pair that is not compiled. Returns the
        new ``first`` flag."""
        rows = scheme.rows(term, self.ibm, lev, ncomp, t, self.row_width)
        if rows.nrows == 0:
            return first
        version = self.ibm.grid_version
        blockamr.apply_band_rows(
            out_mf,
            self.field.mf[lev],
            band_table(rows, version),
            ncomp,
            blockamr.BandMode.Overwrite if first else blockamr.BandMode.Add,
            1.0,  # constant_scale: the affine apply (row-contract §4)
            version,
        )
        return False


def _stack(blocks, ncol):
    """Concatenate per-box arrays, with the empty case's shape spelled out."""
    blocks = [b for b in blocks if b.shape[0]]
    if blocks:
        return np.concatenate(blocks)
    return np.zeros((0, ncol) if ncol else (0,))


def _wall_kernel(scheme):
    """The compiled pair a boundary scheme names, or ``None``.

    The exact peer of the interior dispatch (``cpp_backend``): the scheme owns
    its kernel, and the driver has no ``(operator, method)`` table of its own.
    """
    build = getattr(scheme, "build_cpp_kernel", None)
    return None if build is None else build()


def _operator_of(term):
    """The registry name of a term's operator (``"laplacian"``, ``"div"``, ...)."""
    operator = getattr(term, "_scheme_operator", None)
    if operator is None:
        raise ValueError(
            f"term {type(term).__name__!r} names no operator, so no boundary scheme "
            "can be resolved for it; an IBM method applies to the operators of "
            "SCHEME_REGISTRY."
        )
    return operator


def equation_band(terms):
    """``(width, shape)`` of the band an equation's terms share.

    The width is the widest term's, so the band contains every term's own band
    (they are nested: ``band(w) = {depth <= w}``). The shape is the widest
    *stencil* shape any term declares — a corner-reading scheme needs the
    Chebyshev band, and the cross band would under-select along the diagonals
    for it (design §4).

    Still the rule for a row-only equation, and still what sizes the marker's
    ghost region on the v2 path: W1's siblings read the marker at their own
    stencil reach.
    """
    widths = [_band_width(term) for term in terms]
    shapes = {_band_shape(term) for term in terms}
    return max(widths, default=1), (CROSS if shapes <= {CROSS} else (shapes - {CROSS}).pop())


def _band_width(term):
    """The stencil width the term's own band would be taken at."""
    return int(getattr(term.scheme, "stencil_width", 1))


def _band_shape(term):
    """The stencil shape the term's scheme declares (design §4)."""
    return getattr(term.scheme, "stencil_shape", CROSS)
