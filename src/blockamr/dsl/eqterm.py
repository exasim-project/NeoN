# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""EqTerm — base class for a single PDE term in the fvm DSL.

Every ``exp.*`` / ``imp.*`` operator returns an ``EqTerm``. A term knows:

- its ``kind`` — ``"temporal"`` (ddt), ``"spatial"`` (div/laplacian/grad/
  source) or ``"implicit"`` (matrix side of an MLMG solve);
- its operand ``field`` (and optional ``coefficient`` operand: phi
  face-field / gamma / sigma);
- its ``scheme_key`` — the OpenFOAM-style string it looks up in the
  equation's schemes dict (e.g. ``"div(phi,U)"``, ``"ddt"``).

The ``scheme`` slot holds the resolved scheme *object*; ``exp.*`` operators
resolve it purely from the equation's ``schemes`` dict at discretise time
(inside ``solve()``). Constructing an operator class directly (e.g.
``Div(ff, phi, scheme=Upwind())``) can still pin an explicit scheme object,
which then wins over the dict — used by low-level scheme-accuracy tests that
bypass the DSL's dict-driven flow.

Terms compose lazily and immutably — composition returns NEW objects,
nothing evaluates until ``solve()``:

- ``term + term`` / ``term - term`` -> ``Equation``
- ``scalar * term`` / ``term * scalar`` / ``-term`` -> scaled copy
- ``term == rhs`` -> implicit ``Equation`` (lhs == rhs)

.. warning::
   ``__eq__`` is overridden to build equations (OpenFOAM heritage), so
   ``__hash__`` is identity-based (``object.__hash__``) and terms must
   never be stored in equality-based containers (dict keys, sets) or
   compared for equality.
"""

import copy


class EqTerm:
    """One PDE term: kind + operand field + scalar coeff + scheme slot."""

    kind = None  # "temporal" | "spatial" | "implicit"
    _scheme_operator = None  # registry table name ("div", ...); None = no scheme
    _scheme_explicit = False  # True when a scheme object was passed at the call site

    def __init__(self, field, coeff=1.0, coefficient=None, scheme=None):
        self.field = field
        self.coeff = coeff
        self.coefficient = coefficient
        self.scheme = scheme

    @property
    def scheme_key(self):
        """OpenFOAM-style key into the equation's schemes dict."""
        raise NotImplementedError

    @staticmethod
    def _named(field, role):
        """Return field.name or raise a clear error when it has no name."""
        name = getattr(field, "name", "")
        if not name:
            raise ValueError(
                f"Cannot build a scheme key: the {role} operand has no name. "
                "Construct DSL fields with name=..."
            )
        return name

    def _scheme_key_or_none(self):
        """scheme_key, or None when an operand field has no name."""
        try:
            return self.scheme_key
        except ValueError:
            return None

    def _scaled(self, factor):
        """Return a copy of this term with coeff scaled by *factor*."""
        new = copy.copy(self)
        new.coeff = self.coeff * factor
        return new

    def __mul__(self, scalar):
        if isinstance(scalar, EqTerm):
            return NotImplemented
        return self._scaled(scalar)

    __rmul__ = __mul__

    def __neg__(self):
        return self._scaled(-1.0)

    def __add__(self, other):
        from .equation import Equation

        if isinstance(other, (EqTerm, Equation)):
            return Equation(self, other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, EqTerm):
            return self.__add__(other._scaled(-1.0))
        return NotImplemented

    def __eq__(self, rhs):
        from .equation import Equation

        eqn = Equation(self)
        eqn.rhs = rhs
        return eqn

    # __eq__ builds equations; keep identity hashing (see module docstring).
    __hash__ = object.__hash__
