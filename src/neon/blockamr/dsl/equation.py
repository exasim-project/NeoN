# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Unified lazy Equation: EqTerms + the schemes (fvSchemes) to discretise them.

Nothing evaluates at build time — the terms stay a DSL tree, leaving room
for optimize() to fuse/reorder before dispatch. The linear-solver settings
are NOT here: they are 'how to solve', passed to solve() as ``solution``.
"""

from .eqterm import EqTerm


class Equation:
    """A lazy PDE equation.

    Built from EqTerms via composition (``exp.ddt(U) + exp.div(phi, U)``)
    or directly: ``Equation(*terms, schemes=...)``.

    State:
      explicit_terms : list of temporal/spatial EqTerms
      implicit_lhs   : the implicit EqTerm (matrix side), or None
      rhs            : the RHS term of an implicit system (set via ``==``)
      schemes        : dict of scheme NAMES keyed by scheme_key
                       (e.g. ``{"div(phi,U)": "vanLeer"}``); resolved to
                       scheme objects at discretise time
    """

    def __init__(self, *terms, schemes=None):
        self.explicit_terms = []
        self.implicit_lhs = None
        self.rhs = None
        self.schemes = dict(schemes) if schemes is not None else {}
        for term in terms:
            self._absorb(term)

    def _absorb(self, term):
        if isinstance(term, Equation):
            self.explicit_terms.extend(term.explicit_terms)
            if term.implicit_lhs is not None:
                self.implicit_lhs = term.implicit_lhs
            if term.rhs is not None:
                self.rhs = term.rhs
            if not self.schemes and term.schemes:
                self.schemes = dict(term.schemes)
        elif isinstance(term, EqTerm):
            if term.kind == "implicit":
                self.implicit_lhs = term
            else:
                self.explicit_terms.append(term)
        else:
            raise TypeError(f"Equation terms must be EqTerm or Equation, got {type(term)}")

    @property
    def temporal_ops(self):
        return [t for t in self.explicit_terms if t.kind == "temporal"]

    @property
    def spatial_ops(self):
        return [t for t in self.explicit_terms if t.kind == "spatial"]

    @property
    def required_ngrow(self):
        """Minimum ghost-cell count required by the widest stencil."""
        return max(
            (getattr(getattr(op, "scheme", None), "stencil_width", 1) for op in self.spatial_ops),
            default=1,
        )

    def __add__(self, other):
        if isinstance(other, (EqTerm, Equation)):
            return Equation(self, other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, EqTerm):
            return self.__add__(-1.0 * other)
        return NotImplemented

    def optimize(self):
        """Return an optimised equation.

        Identity today — this is the seam for kernel fusion, stencil
        caching and term reordering, and the reason terms are held lazily.
        """
        return self

    def solve(self, *, dt=None, t=None, solution=None):
        """Discretise and solve: optimize() then delegate to the free solve().

        solution : the field's fvSolution.solvers[field] block — the linear
            solver + tolerances (MLMG rtol/atol/maxIter/bottomSolver) and the
            field's IBM method. Discretisation schemes are NOT passed here —
            they are the equation's own ``schemes`` (bound at construction).
        """
        from .solve import solve as _solve

        eqn = self.optimize()
        _solve(eqn, t=t, dt=dt, solution=solution)
