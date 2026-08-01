# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Assemble a linear system from operators and solve it.

Three layers, each knowing only the one below: the MATRIX
(:class:`MFFaceCoeffs`, matrix-free), OPERATORS that discretise a term into
coefficients (:func:`laplacian`), and a :class:`Solver` handed a
:class:`LinearSystem` -- a matrix and an rhs, pure data. Use ``FaceCoeffSolver``
instead when you want GMG as the SOLVER.

The matrix is allocated on a ``blockamr.MeshLevel(ba, dm, geom)`` -- one AMR level's
layout as a single object, NOT the multi-level ``blockamr.Mesh``.

The system is NON-OWNING: the matrix and the rhs must OUTLIVE it, and the rhs an
operator writes IS the field you passed in. Operators ACCUMULATE, so a reused system
is ``zero()``-ed first. The executor, the layout and the domain BCs are all given to
the MATRIX and reach the operator and the solve from there, so none of them can
disagree.

PRECONDITIONERS ARE THE MATRIX'S, not the solver's: ``precond="gmg"`` /
``"gmg_kokkos"`` / ``"mlmg"`` are built from the coefficients the matrix holds,
and shaped by :class:`~blockamr.solver_config.GmgConfig` nested on the config::

    Solver(SolverConfig(solver="cg", precond="gmg_kokkos",
                        gmg=GmgConfig(pre_sweeps=2, post_sweeps=2))).solve(system, sol)

:class:`MFFaceCoeffs` builds all four.

Two things this surface deliberately does not reach:

* **GMG as the SOLVER.** ``solver="gmg"/"ir"/"mpir"`` RAISE from
  :meth:`Solver.solve` -- they drive the hierarchy directly rather than
  preconditioning a Krylov method with it, which is a different object with a
  different stopping test. Use ``blockamr.FaceCoeffSolver`` for those.
* **Only one operator exists**, :func:`laplacian`. There is no ``ddt``, so the
  cell-centred diagonal SOURCE is written directly with
  ``MFFaceCoeffs.diagonal_source(alpha)``.

Example
-------
>>> mesh = blockamr.MeshLevel(ba, dm, geom)
>>> mat = MFFaceCoeffs.symmetric(mesh, executor=exec, bc=["dirichlet"] * 6)
>>> system = LinearSystem(mat, rhs)
>>> system += laplacian(gamma)
>>> stats = Solver(SolverConfig(solver="cg", precond="gmg", rtol=1e-10)).solve(system, sol)
>>> stats["num_iters"]
"""

from __future__ import annotations

from typing import Any, Optional

from ._blockamr import Laplacian, LinearSystem, MFFaceCoeffs
from ._blockamr import Solver as _Solver
from ._blockamr import la_laplacian as _la_laplacian
from .solver_config import GmgConfig, SolverConfig

__all__ = [
    "GmgConfig",
    "Laplacian",
    "LinearSystem",
    "MFFaceCoeffs",
    "Solver",
    "SolverConfig",
    "laplacian",
]


def laplacian(
    gamma: Any,
    bc_data: Optional[Any] = None,
) -> Laplacian:
    """The implicit diffusion term: ``system += laplacian(gamma)``.

    ``gamma`` is the physical diffusivity. Each face coefficient is written as
    ``-gammaFace/dx**2``, NEGATIVE, so a system of this term alone is
    ``-div(gamma grad phi)`` -- positive-definite, and the opposite sign to OpenFOAM's
    ``fvm::laplacian``. A non-periodic domain face gets the boundary cell's own gamma;
    the homogeneous boundary condition is applied by the matrix, per level, from that
    coefficient. With ``bc_data`` the inhomogeneous constant is written onto the rhs,
    MUTATING the rhs the system holds.

    The mesh and the domain BCs are read off the system's MATRIX, so neither is an
    argument: the operator has nowhere to keep a copy that could disagree with the
    coefficients it writes. `gamma` and `bc_data` are held by POINTER and read when
    ``+=`` runs, so both must OUTLIVE this operator.

    Example
    -------
    >>> system += laplacian(gamma)
    """
    return _la_laplacian(gamma, bc_data=bc_data)


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
