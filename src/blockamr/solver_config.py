# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Validated configuration for the native geometric-multigrid preconditioner.

``GmgConfig`` collects the ``precond="gmg"`` V-cycle knobs of
``FaceCoeffSolver`` (see ``src/bindings/blockAMR/ginkgo_solve.cpp``) into one
validated pydantic model. ``kwargs()`` returns the ``gmg_*`` / ``precond_cycles``
constructor keyword arguments to splat into ``FaceCoeffSolver(...)``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GmgConfig(BaseModel):
    """Knobs for the native matrix-free GMG V-cycle preconditioner.

    Defaults reproduce the built-in fixed behaviour (2+2 red-black
    Gauss-Seidel sweeps, 8 coarsest sweeps, unlimited coarsening).

    Notes
    -----
    ``pre_sweeps`` and ``post_sweeps`` should be equal for a CG-safe symmetric
    preconditioner: with unequal counts the post-smoother is no longer the exact
    adjoint of the pre-smoother, so the V-cycle is non-symmetric and CG may
    stall or diverge (the C++ solver warns in that case).
    """

    model_config = ConfigDict(frozen=True)

    pre_sweeps: int = Field(default=2, ge=0)
    post_sweeps: int = Field(default=2, ge=0)
    coarsest_sweeps: int = Field(default=8, ge=1)
    max_levels: int = Field(default=0, ge=0)  # 0 = auto / unlimited coarsening
    min_bottom: int = Field(default=4, ge=2)
    smoother: Literal["rbgs", "chebyshev"] = "rbgs"
    cycles: int = Field(default=1, ge=1)  # V-cycles per preconditioner apply
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

    def kwargs(self) -> dict:
        """Constructor kwargs to splat into ``FaceCoeffSolver(..., precond="gmg")``."""
        return {
            "gmg_pre_sweeps": self.pre_sweeps,
            "gmg_post_sweeps": self.post_sweeps,
            "gmg_coarsest_sweeps": self.coarsest_sweeps,
            "gmg_max_levels": self.max_levels,
            "gmg_min_bottom": self.min_bottom,
            "gmg_smoother": self.smoother,
            "gmg_precision": self.precision,
            "precond_cycles": self.cycles,
        }
