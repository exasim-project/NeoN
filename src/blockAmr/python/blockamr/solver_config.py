# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Validated configuration for the solvers: the GMG V-cycle knobs, and la.Solver.

``GmgConfig`` collects the ``precond="gmg"`` V-cycle knobs of
``FaceCoeffSolver`` (see ``src/blockAmr/bindings/ginkgoSolve.cpp``) into one
validated pydantic model. ``kwargs()`` returns the ``gmg_*`` / ``precond_cycles``
constructor keyword arguments to splat into ``FaceCoeffSolver(...)``.

``SolverConfig`` does the same for ``blockamr.linear_algebra.Solver``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GmgConfig(BaseModel):
    """Knobs for the native matrix-free GMG V-cycle preconditioner.

    Defaults mirror the C++ binding's (2+2 red-black Gauss-Seidel sweeps, 16
    coarsest sweeps, unlimited coarsening down to a 2-cell bottom, omega=1.1) —
    a measured V-cycle shape rather than a historical one. Preconditioned CG on
    the periodic Helmholtz takes 8 iterations at 64^3, 128^3 and 256^3 alike;
    the previous 8/4/1.0 shape took 11/11/12, so this is both ~1.4x faster at
    256^3 and mesh-independent, which it was not. See ``ginkgoSolve.cpp`` for
    the per-knob measurements and the omega turnover curve.

    Notes
    -----
    ``pre_sweeps`` and ``post_sweeps`` should be equal for a CG-safe symmetric
    preconditioner: with unequal counts the post-smoother is no longer the exact
    adjoint of the pre-smoother, so the V-cycle is non-symmetric and CG may
    stall or diverge (the C++ solver warns in that case).

    ``omega`` != 1.0 breaks that symmetry too, by a smaller amount that the
    measurements say is worth paying up to ~1.1 and not beyond. Set it back to
    1.0 for a bit-for-bit self-adjoint V-cycle.

    The two do not compose: with ``pre_sweeps != post_sweeps`` the default
    ``omega=1.1`` stops CG converging at all, where either breaker alone is fine.
    If you set unequal sweeps, set ``omega=1.0`` with them.
    """

    model_config = ConfigDict(frozen=True)

    pre_sweeps: int = Field(default=2, ge=0)
    post_sweeps: int = Field(default=2, ge=0)
    coarsest_sweeps: int = Field(default=16, ge=1)
    max_levels: int = Field(default=0, ge=0)  # 0 = auto / unlimited coarsening
    min_bottom: int = Field(default=2, ge=2)
    smoother: Literal["rbgs", "chebyshev"] = "rbgs"
    cycles: int = Field(default=1, ge=1)  # V-cycles per preconditioner apply
    # RB-SOR relaxation: sol <- sol + omega * (gs - sol). Ignored by
    # smoother="chebyshev". Must stay in (0, 2) to be a convergent relaxation.
    omega: float = Field(default=1.1, gt=0.0, lt=2.0)
    # V-cycle hierarchy precision: "fp64" (default; byte-for-byte the built-in
    # behaviour), "fp32" (single-precision V-cycle, outer CG/operator stay
    # double — halves the bandwidth-bound V-cycle traffic) or "bf16" (quarters
    # it; stored in bfloat16, still computed in fp32).
    #
    # "bf16" needs ``precond="gmg_kokkos"`` — the shipped GMG hierarchy carries
    # fp64/fp32 only, and the solver raises for the other precond values. It is a
    # measured negative result rather than a recommended setting: 1.36x off the
    # V-cycle, but psi's storage error reaches the coarse grid multiplied by
    # ||A|| ~ 6/dx^2, so the cycle weakens as n^2 and the CG iteration count more
    # than doubles already at 64^3 (11 -> 25) and reaches 273 vs 12 at 256^3.
    # There is no size at which it wins. See solvers/bf16.hpp.
    precision: Literal["fp64", "fp32", "bf16"] = "fp64"
    # Storage type of the COEFFICIENTS alone (alpha and the face arrays); "" means
    # the same as ``precision``, which is what every level did before this existed.
    # May not be wider than ``precision``, and needs ``precond="gmg_kokkos"``.
    #
    # This is the half of the bf16 experiment above that survives. Rounding psi is
    # amplified — the cycle restricts ``b - A psi``, so psi's storage error reaches
    # the coarse grid times ``||A|| ~ 6/dx^2``. Rounding a COEFFICIENT only perturbs
    # the preconditioner's operator by the same ~0.4%; the operator CG applies and
    # the residual it stops on stay fp64, so it can cost iterations but never
    # correctness.
    #
    # Measured at 256^3 with a varying b, fields/coeffs: fp32/bf16 takes the
    # V-cycle from 12.52 to 10.60 ms at a residual reduction of 0.70147 against
    # fp32/fp32's 0.70185 — same cycle, 1.18x cheaper, 9 CG iterations either way
    # (solve 213 -> 195 ms). fp64/bf16 is 1.11x SLOWER: narrow the coefficients
    # only once the fields are narrow.
    coeff_precision: Literal["", "fp64", "fp32", "bf16"] = ""

    def kwargs(self) -> dict:
        """Constructor kwargs to splat into ``FaceCoeffSolver(..., precond="gmg")``."""
        return {
            "gmg_pre_sweeps": self.pre_sweeps,
            "gmg_post_sweeps": self.post_sweeps,
            "gmg_coarsest_sweeps": self.coarsest_sweeps,
            "gmg_max_levels": self.max_levels,
            "gmg_min_bottom": self.min_bottom,
            "gmg_omega": self.omega,
            "gmg_smoother": self.smoother,
            "gmg_precision": self.precision,
            "gmg_coeff_precision": self.coeff_precision,
            "precond_cycles": self.cycles,
        }


class SolverConfig(BaseModel):
    """Everything ``blockamr.linear_algebra.Solver`` needs besides the system.

    Use it for the ``la`` surface (``Solver(SolverConfig(solver="cg")).solve(...)``);
    ``GmgConfig`` above is the separate, older model for the legacy
    ``FaceCoeffSolver`` V-cycle knobs, and the two do not overlap.

    Frozen, like ``GmgConfig``: a config is a value, and a solver holds the one it
    was built with.

    Notes
    -----
    ``solver`` and ``precond`` are plain strings, deliberately not ``Literal``.
    The spellings are parsed exactly once, in C++
    (``linearAlgebra/solverConfig.hpp``'s ``parseSolverKind`` /
    ``parsePrecondKind``, called by the ``Solver`` binding's constructor), and
    every dispatch below that compares the resulting enum. Repeating the list here
    would be a second parse that has to be kept in step with the first.

    ``precond`` other than ``"none"``, and ``solver`` in ``("gmg", "ir", "mpir")``,
    are accepted here and REFUSED by ``Solver.solve`` with an explanation: the GMG
    hierarchy is built from the coefficient fields rather than from a ``LinOp``, so
    it is not reachable through this path. Use ``FaceCoeffSolver`` for those.

    Examples
    --------
    >>> SolverConfig(solver="cg", rtol=1e-10).kwargs()["solver"]
    'cg'
    """

    model_config = ConfigDict(frozen=True)

    solver: str = "bicgstab"
    precond: str = "none"
    max_iter: int = Field(default=1000, ge=1)
    rtol: float = Field(default=1e-10, ge=0.0)
    atol: float = Field(default=0.0, ge=0.0)
    project_nullspace: bool = False
    norm: str = "l2"

    def kwargs(self) -> dict:
        """Constructor kwargs to splat into the ``_blockamr.Solver(...)`` binding."""
        return {
            "solver": self.solver,
            "precond": self.precond,
            "max_iter": self.max_iter,
            "rtol": self.rtol,
            "atol": self.atol,
            "project_nullspace": self.project_nullspace,
            "norm": self.norm,
        }
