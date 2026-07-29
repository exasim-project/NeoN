# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Assemble a linear system from operators and solve it, format-agnostically.

Three layers, each knowing only the one below: a matrix FORMAT
(:class:`MFFaceCoeffs` matrix-free, :class:`CsrMatrix` assembled), OPERATORS that
discretise a term into coefficients (:func:`laplacian`), and a :class:`Solver`
that is handed a :class:`LinearSystem` -- a matrix and an rhs, pure data -- and
cannot tell which format it got.

Use this instead of ``FaceCoeffSolver`` when the coefficients come from a
discretisation rather than from arrays you already have; use ``FaceCoeffSolver``
when you want GMG (see the limitations below).

The system is NON-OWNING: the matrix and the rhs must outlive it, and the rhs an
operator writes IS the field you passed in. Operators ACCUMULATE, so a reused
system is ``zero()``-ed first. The executor is given to the MATRIX and reaches the
solve from there -- ``Solver`` has none of its own, so the two cannot disagree.

Two things are NOT reachable through this surface today, and it does not pretend
otherwise:

* **GMG.** ``precond="gmg"``/``"gmg_kokkos"`` and ``solver="gmg"/"ir"/"mpir"``
  raise from :meth:`Solver.solve`, because the GMG hierarchy is built from the
  coefficient FIELDS rather than from the ``LinOp`` this path solves through. The
  error says so; it is a real limitation, not a bug to route around. Use
  ``blockamr.FaceCoeffSolver`` for a GMG-preconditioned solve.
* **Only one operator exists**, :func:`laplacian`. There is no ``ddt``, so the
  cell-centred diagonal SOURCE is written directly with
  ``Matrix.diagonal_source(alpha)``.

Example
-------
>>> mat = MFFaceCoeffs.symmetric(ba, dm, geom, executor=exec, bc=["dirichlet"] * 6)
>>> system = LinearSystem(mat, rhs)
>>> system += laplacian(gamma, geom, bc=["dirichlet"] * 6)
>>> stats = Solver(SolverConfig(solver="cg", rtol=1e-10)).solve(system, sol)
>>> stats["num_iters"]
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ._blockamr import LinearSystem, Matrix, Operator
from ._blockamr import Solver as _Solver
from ._blockamr import la_laplacian as _la_laplacian
from .solver_config import SolverConfig

__all__ = [
    "CsrMatrix",
    "LinearSystem",
    "MFFaceCoeffs",
    "Matrix",
    "Operator",
    "Solver",
    "SolverConfig",
    "laplacian",
]

_PERIODIC = ["periodic"] * 6


class MFFaceCoeffs:
    """Matrix-free face-coefficient format: no matrix is ever assembled.

    Use it when the mat-vec is cheaper than storing the matrix, which is the whole
    point of the comparison this component exists for; :class:`CsrMatrix` is the
    assembled alternative and takes the identical operator calls, so switching is
    one line.

    Not instantiable: ``symmetric`` / ``asymmetric`` hand back a :class:`Matrix`,
    the erasure everything above the format speaks. The returned matrix OWNS its
    coefficient fields and starts zeroed.

    Example
    -------
    >>> mat = MFFaceCoeffs.symmetric(ba, dm, geom, executor=exec)
    """

    @staticmethod
    def symmetric(
        ba: Any,
        dm: Any,
        geom: Any,
        executor: Optional[Any] = None,
        bc: Optional[Sequence[str]] = None,
    ) -> Matrix:
        """Allocate diag + upper; `lower` aliases `upper` and is reported empty."""
        return Matrix.mf_symmetric(ba, dm, geom, executor=executor, bc=list(bc or _PERIODIC))

    @staticmethod
    def asymmetric(
        ba: Any,
        dm: Any,
        geom: Any,
        executor: Optional[Any] = None,
        bc: Optional[Sequence[str]] = None,
    ) -> Matrix:
        """Additionally allocate `lower`; a convecting operator needs it."""
        return Matrix.mf_asymmetric(ba, dm, geom, executor=executor, bc=list(bc or _PERIODIC))


class CsrMatrix:
    """Assembled format: the same coefficients, held as an explicit Ginkgo CSR.

    Use it as the baseline :class:`MFFaceCoeffs` is measured against. Single-box
    meshes only, which is ``assembleFaceCoeffCsr``'s restriction rather than a new
    one. Assembly is lazy and re-run after any write through the coefficients.

    Not instantiable, for the same reason as :class:`MFFaceCoeffs`.

    Example
    -------
    >>> mat = CsrMatrix.symmetric(ba, dm, geom, executor=exec)
    """

    @staticmethod
    def symmetric(
        ba: Any,
        dm: Any,
        geom: Any,
        executor: Optional[Any] = None,
        bc: Optional[Sequence[str]] = None,
    ) -> Matrix:
        """Allocate diag + upper; `lower` aliases `upper` and is reported empty."""
        return Matrix.csr_symmetric(ba, dm, geom, executor=executor, bc=list(bc or _PERIODIC))

    @staticmethod
    def asymmetric(
        ba: Any,
        dm: Any,
        geom: Any,
        executor: Optional[Any] = None,
        bc: Optional[Sequence[str]] = None,
    ) -> Matrix:
        """Additionally allocate `lower`; a convecting operator needs it."""
        return Matrix.csr_asymmetric(ba, dm, geom, executor=executor, bc=list(bc or _PERIODIC))


def laplacian(
    gamma: Any,
    geom: Any,
    bc: Optional[Sequence[str]] = None,
    bc_data: Optional[Any] = None,
) -> Operator:
    """The implicit diffusion term: ``system += laplacian(gamma, geom, bc=bc)``.

    Writes ``-gammaFace/dx**2`` onto every interior face, so a system of this term
    alone is ``-div(gamma grad phi)`` -- the positive-definite sign every caller in
    this component already writes by hand, and the opposite of OpenFOAM's
    ``fvm::laplacian``. On a non-periodic domain face the coefficient is folded
    onto the diagonal source instead, and with ``bc_data`` onto the rhs too, which
    MUTATES the rhs the system holds.

    `geom` is an argument because a :class:`LinearSystem` carries no geometry: it
    is a matrix and an rhs and nothing else, by design.

    `gamma` and `bc_data` are held by POINTER and read when ``+=`` runs, so both
    must outlive this operator.

    Example
    -------
    >>> system += laplacian(gamma, geom, bc=["dirichlet"] * 6)
    """
    return _la_laplacian(gamma, geom, bc=list(bc or _PERIODIC), bc_data=bc_data)


class Solver(_Solver):
    """Solves a :class:`LinearSystem`; holds a :class:`SolverConfig` and nothing else.

    Use it for any Krylov method; ``blockamr.FaceCoeffSolver`` is the alternative
    when a GMG preconditioner is wanted, which this path cannot supply (see the
    module docstring). There is deliberately no factory: which method the config
    names is decided once, inside, where the string is already parsed.

    Stateless with respect to the system -- the same solver can solve many, and it
    reads the executor off each system's matrix rather than carrying one.

    Example
    -------
    >>> stats = Solver(SolverConfig(solver="cg", rtol=1e-10)).solve(system, sol)
    """

    def __init__(self, config: SolverConfig) -> None:
        super().__init__(**config.kwargs())
