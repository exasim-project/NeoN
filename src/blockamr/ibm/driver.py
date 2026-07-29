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

One path, one composition rule
------------------------------

Every registered ``(operator, method)`` pair is **compiled** and is called
through its ``build_cpp_kernel()`` wrapper with the canonical twelve arguments
(design §4.4), plus whatever that pair appends past the twelfth. They compose
the way they always did: the first term to write uses ``Overwrite``, every
later one ``Add``, and every one of them writes exactly the ``WALL`` cells — so
the set they touch is identical by construction and there is nothing to
negotiate. The band, its width and the row path went with ``source x
ghostCell``'s own kernel, which was the last thing holding them.

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
        #: The ghost width the marker and the packed geometry are built at: the
        #: widest interior stencil, since W1's siblings read the marker at their
        #: own reach (``MARKER_NGROW`` is the classification's floor, not a size).
        self.ngrow = wall_ngrow(self.terms)
        # Classification time: the driver is built before the level loop —
        # before the first fill_patch and before any sweep (design §7,
        # review §4 Q3). Once per (field, method, lev, grid_version); every
        # later evaluate is a read.
        for lev in range(cell_field.mesh.n_levels()):
            self.ibm.ensure_pinned(cell_field, method, lev, self.ngrow)

    def interior_cell_type(self, lev):
        """The marker the interior sweep degrades against (W1)."""
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
        views = self._views(lev, t, ncomp)
        first = True
        for term in self.terms:
            kernel = self.kernels[term]
            scheme = self.schemes[term]
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


def _stack(blocks, ncol):
    """Concatenate per-box arrays, with the empty case's shape spelled out."""
    blocks = [b for b in blocks if b.shape[0]]
    if blocks:
        return np.concatenate(blocks)
    return np.zeros((0, ncol) if ncol else (0,))


def _wall_kernel(scheme):
    """The compiled pair a boundary scheme names.

    The exact peer of the interior dispatch (``cpp_backend``): the scheme owns
    its kernel, and the driver has no ``(operator, method)`` table of its own.
    A scheme without one raises here rather than falling back — there is no row
    path left to fall back to, and a wall condition dropped in silence is the
    failure this design most needs to make loud (design §6).
    """
    build = getattr(scheme, "build_cpp_kernel", None)
    if build is None:
        raise NotImplementedError(
            f"the boundary scheme {type(scheme).__name__!r} for the pair "
            f"('{scheme.operator}', '{scheme.method}') names no compiled kernel: it has no "
            "build_cpp_kernel(). Every registered pair is a compiled kernel under v2 "
            "(plans/IBM/design.md §4.4); the numpy row path it would otherwise have used is "
            "deleted."
        )
    return build()


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


def wall_ngrow(terms):
    """The ghost width one equation's markers and packed geometry are built at.

    The widest interior stencil of the equation's terms, floored at 1: W1's
    marker-aware siblings read the marker at their **own** stencil reach
    (design §5), so a width-2 term needs two ghost cells of it. It is not a band
    width and there is no shape in it — the band went with ``source x
    ghostCell``'s kernel; this is an allocation size and nothing else.
    """
    return max([_stencil_width(term) for term in terms], default=1)


def _stencil_width(term):
    """The interior stencil width the term's scheme declares."""
    return max(1, int(getattr(term.scheme, "stencil_width", 1)))
