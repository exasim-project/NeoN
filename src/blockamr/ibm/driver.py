# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The per-``evaluate`` IBM driver — the band flow (B6, B10).

One ``evaluate`` under the row architecture (``plans/IBM/design.md`` §3)::

    fill_patch(phi)                       once, before the terms
    interior kernel over the whole valid box   the *same* call as the no-IBM path
    apply_band_rows(out, phi, table, Overwrite)  per term, after the sweeps

The non-fluid pin (design §7) is **not** in that list: it is a classification
write, applied once per ``(field, method, lev, grid_version)`` when this driver
is built (B25, ``plans/IBM/review.md`` §4 Q3), so an ``evaluate`` after the
field's first one never writes the field at all.

The interior sweep is untouched — same kernel, same arguments — so the bulk is
bitwise the plain operator's *structurally* rather than by care
(``plans/IBM/overview.md`` §5), and the whole IBM correction is a list of rows
over the band.

The composition rule (design.md §6)
----------------------------------

The band is a property of the **equation**, not of one term: it is
``band(W)`` for ``W = max_t w_t`` over the equation's terms, and every term's
boundary scheme emits rows over that one set. Then the applies compose the way
the interior sweeps do — the first term writes, the rest add — and a cell in
``band(W)`` carries ``sum_t rows_t`` exactly.

Terms of different widths (a width-1 laplacian beside a width-2 div) are what
this rule exists for. The narrow term's rows extend past its own
``band(w_t)``, and there they are its **plain interior formula**: outside its
band every cell of its stencil is fluid, so the row is the operator's own value
and the wall never enters it. The alternative — correcting each term on its own
band, in ``Add`` mode, with the row carrying ``row - interior`` — would keep
the sweep's bits outside each band, but it needs *every* term's interior
formula as a fixed row, which a limiter (``vanLeer``) does not have. This rule
is total.

``ibm is None`` — the no-key path, ``noIbm`` and an empty band — is one branch
*outside* any kernel, in the caller, which is what makes bitwise equality with
the plain operator structural rather than maintained.
"""

import blockamr

from ..schemes.boundary import resolve
from .band import CROSS
from .band_rows import band_table


class BandEvaluation:
    """The band flow for one field, one ``evaluate``.

    Built per ``evaluate`` because the rows are (v1: design §8 — rebuild every
    evaluate, cache on a coefficient generation later). Holds one boundary
    scheme per term, resolved up front so a missing ``(operator, method)`` pair
    raises before any kernel launches.
    """

    def __init__(self, method, name, cell_field, spatial_ops):
        self.method = method
        self.name = name
        self.field = cell_field
        self.ibm = cell_field.mesh.ibm
        self.terms = list(spatial_ops)
        self.schemes = {term: resolve(_operator_of(term), name, term.scheme) for term in self.terms}
        self.width, self.shape = equation_band(self.terms)
        # Classification time, v1: the driver is built before the level loop —
        # before the first fill_patch and before any sweep — and the band has
        # already been classified by then (design §7, review §4 Q3). Once per
        # (field, method, lev, grid_version); every later evaluate is a read.
        for lev in range(cell_field.mesh.n_levels()):
            self.ibm.ensure_pinned(cell_field, method, lev)

    def evaluate_level(self, impl, terms, cell_field, lev, t):
        """One level: the untouched interior sweep, then the rows."""
        return impl.evaluate(terms, cell_field, lev, t, ibm=self)

    def source_level(self, impl, terms, cell_field, lev, t):
        """The step-side twin of :meth:`evaluate_level`: interior sweep and
        band rows — returning the accumulated source MultiFab instead of host
        arrays, so ``solve()``'s time schemes can feed it to ``euler_update``
        (the R4 seam between operator and update)."""
        return impl.source(terms, cell_field, lev, t, ibm=self)

    def apply(self, out_mf, lev, t):
        """Overwrite the band cells of the accumulated result with the rows.

        Called by the backend once the interior sweep of **every** term has
        run: a later term's sweep writes the whole valid box, so a correction
        applied between two sweeps would be overwritten by the second one.

        The first term's rows replace what the sweep left in the band; every
        further term adds to them, which is the same accumulation the interior
        sweeps do into the scratch source. Every term's rows cover the *same*
        set — the equation's band — which is what makes that exact.
        """
        ncomp = self.field.ncomp
        version = self.ibm.grid_version
        mode = blockamr.BandMode.Overwrite
        for term in self.terms:
            rows = self.schemes[term].rows(term, self.ibm, lev, ncomp, t, self.width)
            if rows.nrows == 0:
                continue
            blockamr.apply_band_rows(
                out_mf,
                self.field.mf[lev],
                band_table(rows, version),
                ncomp,
                mode,
                1.0,  # constant_scale: the affine apply (row-contract §4)
                version,
            )
            mode = blockamr.BandMode.Add


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
    for it (design §4). One equation, one band, one composition rule.
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
