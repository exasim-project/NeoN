# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The ``(operator, method)`` boundary-scheme registry and its resolver (B5).

A *boundary scheme* is the one place an operator and an IBM method meet
(``plans/IBM/design.md`` §6). Under v2 it **names a compiled kernel** and does
no arithmetic at all::

    class GhostCellLaplacian:
        operator = "laplacian"
        method = "ghostCell"

        def __init__(self, interior_scheme): ...
        def build_cpp_kernel(self) -> CppWallKernel: ...
        def wall_coeff(self, term, t): ...
        def wall_extras(self, term, lev): ...

The table is keyed by ``(operator, method)`` and **not** by the interior
scheme: that one is handed to the constructor so a pair that needs to know the
interior face rule can read it (``div``'s D1 degrade, in ``wall_extras``), which
keeps the table at ``n_operators x n_methods``.

Two rules, and both are about failing loudly rather than plausibly:

* **No fallback.** A missing pair raises, naming the pair and listing the
  registered ones. Falling back to the interior scheme would silently drop the
  wall condition and return a field that looks right (design §6, A8).
* **An interior scheme must declare its stencil shape.** The declaration
  outlived the band it was invented for: with W1 the shape is what says whether
  a scheme reads its corners, and a scheme that does not say cannot be reasoned
  about at all. It stays required, and stays checked here, because a silent
  default is exactly the failure this rule exists to prevent.

``noIbm`` and the absent-``"ibm"`` path never reach here: they produce no band
and the band sweep does not launch.
"""

#: ``(operator, method)`` -> boundary scheme class. Populated by
#: :func:`register`; read only by :func:`resolve` and :func:`pairs`.
BOUNDARY_SCHEMES = {}


def register(scheme_cls):
    """Register ``scheme_cls`` under its own ``(operator, method)`` pair.

    Usable as a decorator. Re-registering the same class is a no-op; a
    different class for a pair that is already taken raises, because the two
    would differ only in which import ran last.
    """
    key = (scheme_cls.operator, scheme_cls.method)
    taken = BOUNDARY_SCHEMES.get(key)
    if taken is not None and taken is not scheme_cls:
        raise ValueError(
            f"boundary scheme {key} is already registered to "
            f"{taken.__name__!r}; {scheme_cls.__name__!r} cannot take it."
        )
    BOUNDARY_SCHEMES[key] = scheme_cls
    return scheme_cls


#: The two stencil shapes an interior scheme may declare. Rehomed here when
#: ``ibm/band.py`` was deleted: the band that used to switch on them is gone,
#: the declaration is not (see the module docstring), and this is the one file
#: that reads it.
#: Axis-ray stencil.
CROSS = "cross"

#: Corner-reading stencil.
BOX = "box"

SHAPES = (BOX, CROSS)


def pairs():
    """The registered ``(operator, method)`` pairs, sorted."""
    return sorted(BOUNDARY_SCHEMES)


def resolve(operator, method, interior_scheme):
    """The boundary scheme for ``(operator, method)``, built on ``interior_scheme``.

    Raises, naming the pair and listing the registered ones, when the pair is
    absent — there is no fallback to the interior scheme.
    """
    _check_stencil_shape(interior_scheme)
    scheme_cls = BOUNDARY_SCHEMES.get((operator, method))
    if scheme_cls is None:
        raise ValueError(
            f"no boundary scheme for the pair ('{operator}', '{method}'); "
            f"registered pairs: {pairs()}. A missing pair is not filled in by the "
            "interior scheme: that would drop the wall condition silently."
        )
    return scheme_cls(interior_scheme)


def _check_stencil_shape(interior_scheme):
    """Reject an interior scheme that does not say which band it needs."""
    shape = getattr(interior_scheme, "stencil_shape", None)
    name = type(interior_scheme).__name__
    if shape is None:
        raise ValueError(
            f"interior scheme '{name}' declares no stencil_shape, so the band it "
            f"needs is unknown; declare one of {list(SHAPES)} on the scheme class. "
            "There is no default: a cross band under a corner-reading scheme "
            "under-selects along the diagonals, which is a wrong answer in the band "
            "with a correct bulk."
        )
    if shape not in SHAPES:
        raise ValueError(
            f"interior scheme '{name}' declares stencil_shape '{shape}'; "
            f"the shapes are {list(SHAPES)}."
        )


# The shipped pairs register themselves on import, at the bottom so
# ``register`` exists by the time they run.
from . import ghost_cell as _ghost_cell  # noqa: F401
