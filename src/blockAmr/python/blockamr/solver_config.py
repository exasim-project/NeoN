# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Validated pydantic configuration for the solvers.

``GmgConfig`` collects the ``precond="gmg"`` V-cycle knobs of ``FaceCoeffSolver``
(see ``src/blockAmr/bindings/ginkgoSolve.cpp``); its ``kwargs()`` emits the ``gmg_*``
keys plus ``precond_cycles`` to splat into the constructor. ``SolverConfig`` does the
same for ``blockamr.linear_algebra.Solver``.

These models are a VALIDATED MIRROR, not a source: the one C++ default list is
``la::SolverConfig``/``la::GmgConfig`` in
``include/NeoN/blockAmr/linearAlgebra/solverConfig.hpp``, which every
``nb::arg`` default of both bindings now reads from. Every field default here must still
match it exactly -- nothing enforces that automatically, so change the C++ struct first
and this file second.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GmgConfig(BaseModel):
    """Knobs for the native matrix-free GMG V-cycle preconditioner.

    Defaults mirror the C++ binding's and are a measured V-cycle shape: 8 CG
    iterations on the periodic Helmholtz at 64^3, 128^3 and 256^3 alike, against
    11/11/12 for the previous 8/4/1.0. Per-knob measurements are in
    ``ginkgoSolve.cpp``.

    Notes
    -----
    CG needs a symmetric preconditioner. ``pre_sweeps`` != ``post_sweeps`` makes the
    post-smoother something other than the pre-smoother's adjoint, so CG may stall or
    diverge (the C++ solver warns); ``omega`` != 1.0 breaks symmetry more mildly, and
    1.0 gives a bit-for-bit self-adjoint V-cycle. The two do not compose — unequal
    sweeps at the default ``omega=1.1`` stops CG converging at all, so pass
    ``omega=1.0`` with them.
    """

    model_config = ConfigDict(frozen=True)

    pre_sweeps: int = Field(default=2, ge=0)
    post_sweeps: int = Field(default=2, ge=0)
    coarsest_sweeps: int = Field(default=16, ge=1)
    max_levels: int = Field(default=0, ge=0)  # 0 = auto / unlimited coarsening
    min_bottom: int = Field(default=2, ge=2)
    smoother: Literal["rbgs", "chebyshev"] = "rbgs"
    cycles: int = Field(default=1, ge=1)  # V-cycles per preconditioner apply
    # RB-SOR relaxation: sol <- sol + omega * (gs - sol). Ignored by smoother="chebyshev".
    omega: float = Field(default=1.1, gt=0.0, lt=2.0)
    # V-cycle hierarchy storage, validated; "bf16" needs ``precond="gmg_kokkos"`` (the
    # shipped hierarchy carries fp64/fp32 only, and the solver raises for the others).
    # Measured: report/blockamr-precision-measurements.md#why-fp32-is-the-default
    precision: Literal["fp64", "fp32", "bf16"] = "fp32"
    # Storage of the COEFFICIENTS alone (alpha and the face arrays); "" means the same as
    # ``precision``. Validated: may not be wider than ``precision``, needs "gmg_kokkos".
    # Measured: report/blockamr-precision-measurements.md#why-fp32-is-the-default
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

    ``GmgConfig`` is NESTED rather than respelled, so the ``la`` surface and the legacy
    ``FaceCoeffSolver`` cannot describe different V-cycles. Frozen: a config is a value.

    Notes
    -----
    ``solver`` and ``precond`` are deliberately not ``Literal`` — the spellings are
    parsed exactly once, in C++ (``linearAlgebra/solverConfig.hpp``), and repeating the
    list here would be a second parse to keep in step with the first.

    ``precond`` is built by the MATRIX from the coefficients it holds, so
    ``Solver.solve`` raises -- naming the format and the precond -- on a format that
    cannot build it (``CsrMatrix`` takes ``"none"``/``"mlmg"`` only).

    ``solver`` in ``("gmg", "ir", "mpir")`` is accepted here and REFUSED by
    ``Solver.solve``: those want the hierarchy as the SOLVER rather than as a
    preconditioner of one, a different object with a different stopping test. Use
    ``FaceCoeffSolver`` for them.

    Examples
    --------
    >>> SolverConfig(solver="cg", rtol=1e-10).kwargs()["solver"]
    'cg'
    >>> SolverConfig(precond="gmg", gmg=GmgConfig(pre_sweeps=4)).kwargs()["gmg_pre_sweeps"]
    4
    """

    model_config = ConfigDict(frozen=True)

    solver: str = "bicgstab"
    precond: str = "none"
    max_iter: int = Field(default=1000, ge=1)
    rtol: float = Field(default=1e-10, ge=0.0)
    atol: float = Field(default=0.0, ge=0.0)
    project_nullspace: bool = False
    norm: str = "l2"
    # Inert unless ``precond`` names a GMG variant, as the C++ GmgConfig is.
    gmg: GmgConfig = GmgConfig()

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
            # The same dict FaceCoeffSolver is splatted with: one spelling, two bindings.
            **self.gmg.kwargs(),
        }
