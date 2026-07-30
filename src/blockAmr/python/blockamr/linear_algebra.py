# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Assemble a linear system from operators and solve it, format-agnostically.

Three layers, each knowing only the one below: a matrix FORMAT (:class:`MFFaceCoeffs`
matrix-free, :class:`CsrMatrix` assembled), OPERATORS that discretise a term into
coefficients (:func:`laplacian`), and a :class:`Solver` handed a
:class:`LinearSystem` -- a matrix and an rhs, pure data -- that cannot tell which
format it got. Use ``FaceCoeffSolver`` instead when you want GMG as the SOLVER.

A format is allocated on a ``blockamr.MeshLevel(ba, dm, geom)`` -- one AMR level's
layout as a single object, NOT the multi-level ``blockamr.Mesh``.

The system is NON-OWNING: the matrix and the rhs must OUTLIVE it, and the rhs an
operator writes IS the field you passed in. Operators ACCUMULATE, so a reused system
is ``zero()``-ed first. The executor is given to the MATRIX and reaches the solve from
there, so the two cannot disagree.

PRECONDITIONERS ARE THE MATRIX'S, not the solver's: ``precond="gmg"`` /
``"gmg_kokkos"`` / ``"mlmg"`` are built by the FORMAT from the coefficients it holds,
and shaped by :class:`~blockamr.solver_config.GmgConfig` nested on the config::

    Solver(SolverConfig(solver="cg", precond="gmg_kokkos",
                        gmg=GmgConfig(pre_sweeps=2, post_sweeps=2))).solve(system, sol)

:class:`MFFaceCoeffs` builds all four; :class:`CsrMatrix` takes ``"none"`` and
``"mlmg"`` only and :meth:`Solver.solve` raises for the rest, naming the format and
the precond.

Two things this surface deliberately does not reach:

* **GMG as the SOLVER.** ``solver="gmg"/"ir"/"mpir"`` RAISE from
  :meth:`Solver.solve` -- they drive the hierarchy directly rather than
  preconditioning a Krylov method with it, which is a different object with a
  different stopping test. Use ``blockamr.FaceCoeffSolver`` for those.
* **Only one operator exists**, :func:`laplacian`. There is no ``ddt``, so the
  cell-centred diagonal SOURCE is written directly with
  ``Matrix.diagonal_source(alpha)``.

Example
-------
>>> mesh = blockamr.MeshLevel(ba, dm, geom)
>>> mat = MFFaceCoeffs.symmetric(mesh, executor=exec, bc=["dirichlet"] * 6)
>>> system = LinearSystem(mat, rhs)
>>> system += laplacian(gamma, geom, bc=["dirichlet"] * 6)
>>> stats = Solver(SolverConfig(solver="cg", precond="gmg", rtol=1e-10)).solve(system, sol)
>>> stats["num_iters"]
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ._blockamr import LinearSystem, Matrix, Operator
from ._blockamr import Solver as _Solver
from ._blockamr import la_laplacian as _la_laplacian
from .solver_config import GmgConfig, SolverConfig

__all__ = [
    "CsrMatrix",
    "GmgConfig",
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

    :class:`CsrMatrix` is the assembled alternative and takes identical operator calls,
    so switching is one line. Not instantiable: ``symmetric`` / ``asymmetric`` hand back
    a :class:`Matrix`, which OWNS its coefficient fields and starts zeroed.

    Example
    -------
    >>> mat = MFFaceCoeffs.symmetric(blockamr.MeshLevel(ba, dm, geom), executor=exec)
    """

    @staticmethod
    def symmetric(
        mesh: Any,
        executor: Optional[Any] = None,
        bc: Optional[Sequence[str]] = None,
    ) -> Matrix:
        """Allocate diag + upper; `lower` aliases `upper` and is reported empty."""
        return Matrix.mf_symmetric(mesh, executor=executor, bc=list(bc or _PERIODIC))

    @staticmethod
    def asymmetric(
        mesh: Any,
        executor: Optional[Any] = None,
        bc: Optional[Sequence[str]] = None,
    ) -> Matrix:
        """Additionally allocate `lower`; a convecting operator needs it."""
        return Matrix.mf_asymmetric(mesh, executor=executor, bc=list(bc or _PERIODIC))


class CsrMatrix:
    """Assembled format: the same coefficients, held as an explicit Ginkgo CSR.

    The baseline :class:`MFFaceCoeffs` is measured against. SINGLE-BOX meshes only
    (``assembleFaceCoeffCsr``'s restriction). Assembly is lazy and re-run after any
    write through the coefficients. Not instantiable, as :class:`MFFaceCoeffs`.

    Example
    -------
    >>> mat = CsrMatrix.symmetric(blockamr.MeshLevel(ba, dm, geom), executor=exec)
    """

    @staticmethod
    def symmetric(
        mesh: Any,
        executor: Optional[Any] = None,
        bc: Optional[Sequence[str]] = None,
    ) -> Matrix:
        """Allocate diag + upper; `lower` aliases `upper` and is reported empty."""
        return Matrix.csr_symmetric(mesh, executor=executor, bc=list(bc or _PERIODIC))

    @staticmethod
    def asymmetric(
        mesh: Any,
        executor: Optional[Any] = None,
        bc: Optional[Sequence[str]] = None,
    ) -> Matrix:
        """Additionally allocate `lower`; a convecting operator needs it."""
        return Matrix.csr_asymmetric(mesh, executor=executor, bc=list(bc or _PERIODIC))


def laplacian(
    gamma: Any,
    geom: Any,
    bc: Optional[Sequence[str]] = None,
    bc_data: Optional[Any] = None,
) -> Operator:
    """The implicit diffusion term: ``system += laplacian(gamma, geom, bc=bc)``.

    ``gamma`` is the physical diffusivity. Each face coefficient is written as
    ``-gammaFace/dx**2``, NEGATIVE, so a system of this term alone is
    ``-div(gamma grad phi)`` -- positive-definite, and the opposite sign to OpenFOAM's
    ``fvm::laplacian``. A non-periodic domain face gets the boundary cell's own gamma;
    the homogeneous boundary condition is applied by the matrix, per level, from that
    coefficient. With ``bc_data`` the inhomogeneous constant is written onto the rhs,
    MUTATING the rhs the system holds.

    `geom` is an argument because a :class:`LinearSystem` carries no geometry.
    `gamma` and `bc_data` are held by POINTER and read when ``+=`` runs, so both must
    OUTLIVE this operator.

    Example
    -------
    >>> system += laplacian(gamma, geom, bc=["dirichlet"] * 6)
    """
    return _la_laplacian(gamma, geom, bc=list(bc or _PERIODIC), bc_data=bc_data)


class Solver(_Solver):
    """Solves a :class:`LinearSystem`; holds a :class:`SolverConfig` and nothing else.

    Any Krylov method, preconditioned or not; ``FaceCoeffSolver`` is the alternative
    when the V-cycle is wanted as the SOLVER. No factory: which method the config names
    is decided once, inside, where the string is already parsed.

    STATELESS with respect to the system -- the same solver can solve many, reading the
    executor and the preconditioner off each system's matrix per solve.

    Example
    -------
    >>> stats = Solver(SolverConfig(solver="cg", precond="gmg")).solve(system, sol)
    """

    def __init__(self, config: SolverConfig) -> None:
        super().__init__(**config.kwargs())
